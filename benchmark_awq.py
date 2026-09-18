"""Native torchao AWQ W4A16, training-only calibration and full cascade benchmark.

Run: .venv/bin/python benchmark_awq.py (torchao 0.16.0, torch 2.10.0).
"""
import copy
import itertools
import json
import platform
import subprocess
from datetime import datetime, timezone

import torch
import torchao
from torchao.prototype.awq import AWQConfig
from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.quant_api import Int4PackingFormat
from torchao.quantization import Int4TilePackedTo4dTensor

from benchmark_nvfp4 import time_call
from evaluation import digest, load_model, recipe_probabilities
from garment_experiment import details
from train import OUT, load_data


def eligible(module, name):
    return isinstance(module, torch.nn.Linear) and '.channel.' in name


def storage_bytes(model):
    def size(t):
        if isinstance(t, Int4TilePackedTo4dTensor):
            names, _ = t.__tensor_flatten__()
            return sum(size(getattr(t, n)) for n in names)
        return t.numel() * t.element_size()
    return sum(size(t) for t in itertools.chain(model.parameters(), model.buffers()))


@torch.inference_mode()
def run():
    recipe = json.loads((OUT / 'final_recipe.json').read_text())
    x, y, vx, vy, _, _ = load_data()
    # Ten images per class, deterministic offsets within the TRAIN split only.
    ids = torch.cat([(y == c).nonzero().flatten()[:10] for c in range(10)])
    calibration = x[ids].clone()
    assert len(ids) == 100 and torch.equal(y[ids].bincount(), torch.full((10,), 10, device=y.device))
    result = dict(started_at=datetime.now(timezone.utc).isoformat(), recipe=recipe,
                  calibration_train_split_offsets=ids.cpu().tolist(),
                  method='Native torchao AWQ, 20 scale candidates, group32 asymmetric INT4 tile-packed channel weights and BF16 activations. Token matrices, embedding, head and normalization remain original precision. Plain INT4 and channel-BF16 controls use the identical layer subset.',
                  limitations='Shared GPU; eager execution only. Timing includes actual cascade routing, TTA, probability aggregation and CPU output, but excludes loading and calibration. Packed-byte counts include padding, group scales/zeros and AWQ input scales; allocator/serialization overhead excluded. Logical parameter count and mathematical dense FLOPs are unchanged. No test data are loaded.',
                  environment=dict(torch=torch.__version__, torchao=torchao.__version__, python=platform.python_version(), device=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                                   gpu_status=subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader']).decode().strip()),
                  members=[], validation={}, timing={})
    del x, y
    groups = {mode: {} for mode in ('reference', 'channel_bf16', 'plain_int4', 'awq_int4')}
    config = Int4WeightOnlyConfig(group_size=32, int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D, set_inductor_config=False)
    for c in recipe['components']:
        name = c['name']
        assert digest(name) == c['sha256']
        original = load_model(name).requires_grad_(False)
        groups['reference'][name] = original
        control = copy.deepcopy(original)
        selected = [n for n, m in control.named_modules() if eligible(m, n)]
        for n in selected:
            control.get_submodule(n).to(torch.bfloat16)
        groups['channel_bf16'][name] = control
        row = dict(name=name, quantized_layers=selected, storage_bytes={}, logical_parameters=sum(p.numel() for p in original.parameters()), all_bf16_parameter_bytes=sum(p.numel()*2 for p in original.parameters()))
        for mode in ('plain_int4', 'awq_int4'):
            model = copy.deepcopy(control)
            if mode == 'awq_int4':
                quantize_(model, AWQConfig(config, step='prepare', scale_search_space_size=20), filter_fn=eligible)
                model(calibration)
                # Observers see FP32 LayerNorm output before autocast. Match the
                # actual BF16 GEMM input for native AWQ's out-of-forward search.
                for n in selected:
                    observed = model.get_submodule(n)
                    observed.act_obs.inputs = [a.to(observed.weight.dtype) for a in observed.act_obs.inputs]
                quantize_(model, AWQConfig(config, step='convert'), filter_fn=eligible)
            else:
                quantize_(model, config, filter_fn=eligible)
            for n in selected:
                weight = model.get_submodule(n).weight
                assert isinstance(weight, Int4TilePackedTo4dTensor)
                assert (weight.act_pre_scale is not None) == (mode == 'awq_int4')
            assert sum(p.numel() for p in model.parameters()) == row['logical_parameters']
            assert torch.isfinite(model(calibration[:2])).all()
            groups[mode][name] = model
        for mode, models in groups.items():
            row['storage_bytes'][mode] = storage_bytes(models[name])
        result['members'].append(row)
        print('Calibrated', name, row['storage_bytes'], flush=True)
    del calibration
    sample = vx[:1024]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        groups['awq_int4'][recipe['components'][0]['name']](sample[:16])
        torch.cuda.synchronize()
    result['packed_profiler_operations'] = sorted({e.name for e in prof.events() if 'int4' in e.name.lower() or 'tinygemm' in e.name.lower()})
    assert any('_weight_int4pack_mm' in n for n in result['packed_profiler_operations'])
    for mode, models in groups.items():
        p, execution = recipe_probabilities(recipe, vx, models)
        result['validation'][mode] = dict(**details(p, vy), execution=execution)
        if mode == 'reference':
            assert result['validation'][mode]['correct'] == recipe['correct']
        torch.save(p, OUT / f'awq_{mode}_validation.pt')
        timing = time_call(lambda: recipe_probabilities(recipe, sample, models))
        timing['images_per_second'] = len(sample) * 1000 / timing['median_ms']
        result['timing'][mode] = timing
        print(mode, result['validation'][mode]['correct'], timing['median_ms'], flush=True)
        (OUT / 'awq_benchmark.json').write_text(json.dumps(result, indent=2))
    result['complete'] = True
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'awq_benchmark.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
