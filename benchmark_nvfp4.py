"""Real torchao NVFP4 W4A4 inference, validation accuracy and synchronized timing.

Run with .venv/bin/python after installing torchao==0.16.0 alongside torch==2.10.0.
Token-mixing matrices, embeddings, normalization and the classifier stay unquantized.
"""
import argparse
import copy
import itertools
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime,timezone

import torch
import torchao
from torchao.quantization import quantize_
from torchao.prototype.mx_formats.inference_workflow import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

from train import OUT,DEVICE,load_data
from evaluation import load_model,probabilities,calibrate,digest
from garment_experiment import details


def quantize_channels(model, last_blocks=None):
    count=len(model.net.blocks)
    def eligible(module,name):
        if not isinstance(module,torch.nn.Linear) or '.channel.' not in name:
            return False
        index=int(name.split('.blocks.')[1].split('.')[0])
        return (last_blocks is None or index>=count-last_blocks) and all(d%16==0 for d in module.weight.shape)
    names=[]
    for name,module in model.named_modules():
        if eligible(module,name):
            module.to(torch.bfloat16)
            names.append(name)
    quantize_(model,NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True),filter_fn=eligible)
    assert names and all(isinstance(model.get_submodule(n).weight,NVFP4Tensor) for n in names)
    return names


def storage_bytes(model):
    total=0
    for p in itertools.chain(model.parameters(),model.buffers()):
        if isinstance(p,NVFP4Tensor):
            total+=sum(t.numel()*t.element_size() for t in (p.qdata,p.scale,p.per_tensor_scale,p.act_per_tensor_scale) if t is not None)
        else:
            total+=p.numel()*p.element_size()
    return total


@torch.inference_mode()
def time_call(fn,repeats=7):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times=[]
    for _ in range(repeats):
        torch.cuda.synchronize()
        start=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    return dict(median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),samples_ms=times)


@torch.inference_mode()
def run(recipe_path,compile_models=False):
    assert DEVICE.type=='cuda' and torch.cuda.get_device_capability()[0]>=10
    recipe=json.loads(recipe_path.read_text())
    x,y,vx,vy,_,_=load_data()
    del x,y
    models=[]; compressed=[]; names=[]; members=[]
    for c in recipe['components']:
        assert digest(c['name'])==c['sha256']
        model=load_model(c['name']).requires_grad_(False)
        quantized=copy.deepcopy(model)
        selected=quantize_channels(quantized)
        models.append(model); compressed.append(quantized); names.append(selected)
        members.append(dict(name=c['name'],views=c['views'],quantized_linear_layers=selected,
                            reference_storage_bytes=storage_bytes(model),
                            all_bf16_parameter_bytes=sum(p.numel()*2 for p in model.parameters()),
                            nvfp4_storage_bytes=storage_bytes(quantized)))
    timing_models={'reference':models,'nvfp4':compressed}
    if compile_models:
        # Six distinct models and contiguous/non-contiguous views share Model.forward's code cache.
        torch._dynamo.config.recompile_limit=32
        # Compile both paths under the same setting; exclude compilation from steady-state timing.
        timing_models.update({k+'_compiled':[torch.compile(m,fullgraph=True,mode='default') for m in v]
                              for k,v in list(timing_models.items())})
    result=dict(started_at=datetime.now(timezone.utc).isoformat(),recipe=recipe,
                environment=dict(python=platform.python_version(),torch=torch.__version__,torchao=torchao.__version__,
                                 cuda=torch.version.cuda,device=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                                 gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,temperature.gpu','--format=csv,noheader']).decode().strip()),
                members=members,timing={},validation={},
                method='Channel hidden Linear layers use packed NVFP4 weights and dynamically quantized NVFP4 activations, block size 16 and FP8 block scales. Other layers remain original precision; reference uses original CUDA BF16 autocast.',
                limitations='Shared GPU; observed latency is descriptive. Dense mathematical MACs/FLOPs and logical parameter counts do not decrease with quantization. Input images are already resident on GPU; end-to-end timing includes view construction, all model calls, probability calibration/aggregation and output copy to CPU, excludes dataset loading, checkpoint loading and compilation.')
    sample=vx[:128]
    def ensemble(group,images):
        weights=torch.tensor([c.get('weight',1.) for c in recipe['components']])
        weights=weights/weights.sum()
        return sum(w*calibrate(probabilities(m,images,c['views']),c.get('temperature',1.)) for w,m,c in zip(weights,group,recipe['components']))
    # Confirm a hardware FP4 GEMM is executed; retain profiler operation/kernel names as evidence.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
        compressed[0](sample)
        torch.cuda.synchronize()
    result['fp4_profiler_operations']=sorted({e.name for e in prof.events() if 'scaled_mm' in e.name or 'e2m1' in e.name.lower() or 'fp4' in e.name.lower()})
    assert any('scaled_mm' in name for name in result['fp4_profiler_operations'])
    for key,group in timing_models.items():
        print('Benchmark:',key,flush=True)
        single=[time_call(lambda m=m:m(sample),repeats=11) for m in group]
        full=time_call(lambda:ensemble(group,sample))
        full['images_per_second']=len(sample)/(full['median_ms']/1000)
        result['timing'][key]=dict(single_view_batch128=single,full_recipe_batch128=full)
        (OUT/'nvfp4_benchmark.json').write_text(json.dumps(result,indent=2))
    for key,group in (('reference',models),('nvfp4',compressed)):
        print('Validation:',key,flush=True)
        p=ensemble(group,vx)
        result['validation'][key]=details(p,vy)
        torch.save(p,OUT/f'nvfp4_{key}_validation.pt')
        (OUT/'nvfp4_benchmark.json').write_text(json.dumps(result,indent=2))
    result['speedup_eager']=result['timing']['reference']['full_recipe_batch128']['median_ms']/result['timing']['nvfp4']['full_recipe_batch128']['median_ms']
    if compile_models:
        result['speedup_compiled']=result['timing']['reference_compiled']['full_recipe_batch128']['median_ms']/result['timing']['nvfp4_compiled']['full_recipe_batch128']['median_ms']
    result['completed_at']=datetime.now(timezone.utc).isoformat()
    result['complete']=True
    (OUT/'nvfp4_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('validation','speedup_eager','complete')},indent=2),flush=True)


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',type=Path,default=OUT/'final_recipe.json')
    parser.add_argument('--compile',action='store_true')
    args=parser.parse_args()
    run(args.recipe,args.compile)
