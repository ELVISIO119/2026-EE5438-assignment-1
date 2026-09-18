"""Import the user's earlier two Mixers without retraining or altering their source."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat
import torch
from torch import nn

from evaluation import digest, probabilities
from garment_experiment import details
from train import DEVICE, OUT, Model, SEED, load_data, model_cost

SOURCE=Path('/home/poffices_ssh/Assign01_Cai_Haochen_58561440')
NAMES={'legacy_clean':'mixer_p4_clean_finetune','legacy_muon':'mixer_p4_muon'}


@torch.inference_mode()
def run():
    x,y,vx,vy,mean,std=load_data(); del x,y
    # Only the two already-inspected model class definitions are executed, not notebook I/O.
    notebook=nbformat.read(SOURCE/'work/advanced_experiments.ipynb',4)
    cell=next(c.source for c in notebook.cells if c.cell_type=='code' and 'class MLPMixer(' in c.source)
    definitions=[node for node in ast.parse(cell).body if isinstance(node,ast.ClassDef) and node.name in ('MixerBlock','MLPMixer')]
    assert len(definitions)==2
    namespace=dict(torch=torch,nn=nn,mean=mean,std=std)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'<legacy model definitions>','exec'),namespace)
    evidence=[]; ensemble=[]
    for name,original in NAMES.items():
        source=SOURCE/'results'/f'{original}.pt'
        checkpoint=torch.load(source,map_location='cpu',weights_only=True)
        cfg=checkpoint['config']; assert checkpoint['seed']==SEED
        assert abs(checkpoint['mean']-mean)<1e-7 and abs(checkpoint['std']-std)<1e-7
        assert cfg['token_hidden']==128
        target_cfg=dict(name=name,model='mixer',patch=cfg['patch'],width=cfg['channels'],depth=cfg['depth'],
                        dropout=cfg['dropout'],bf16=True,epochs=cfg['epochs'],legacy_source=original)
        state={}
        for key,value in checkpoint['state_dict'].items():
            mapped='net.'+key.replace('.norm1.','.token_norm.').replace('.norm2.','.channel_norm.').replace('.token_mlp.','.token.').replace('.channel_mlp.','.channel.')
            assert mapped not in state
            state[mapped]=value
        model=Model(target_cfg,mean,std).to(DEVICE).eval()
        model.load_state_dict(state)
        oracle=namespace['MLPMixer'](**{k:cfg[k] for k in ('patch','channels','depth','token_hidden','dropout')}).to(DEVICE).eval()
        oracle.load_state_dict(checkpoint['state_dict'])
        for batch in vx.split(128):
            with torch.autocast(device_type=DEVICE.type,dtype=torch.bfloat16,enabled=DEVICE.type=='cuda'):
                expected=oracle(batch).float()
            assert torch.equal(model(batch),expected), 'Renamed checkpoint changed logits.'
        p=probabilities(model,vx)
        record=json.loads((SOURCE/'results'/f'{original}.json').read_text())
        assert abs(details(p,vy)['accuracy']-record['val_accuracy'])<1e-6
        params,macs=model_cost(model)
        target=dict(state_dict=state,config=target_cfg,mean=mean,std=std)
        path=OUT/f'{name}.pt'
        if path.exists():
            old=torch.load(path,map_location='cpu',weights_only=True)
            assert old['config']==target_cfg and all(torch.equal(old['state_dict'][k],v) for k,v in state.items())
        else:
            torch.save(target,path)
        imported=dict(record,config=target_cfg,macs=macs,legacy_config=cfg,
                      provenance=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                                      original_class_definitions_sha256=hashlib.sha256(ast.unparse(ast.Module(body=definitions,type_ignores=[])).encode()).hexdigest(),
                                      transformation='Tensor names only; no training. All 6,000 validation logits exactly equal the original class implementation.',
                                      imported_at=datetime.now(timezone.utc).isoformat()))
        (OUT/f'{name}.json').write_text(json.dumps(imported,indent=2))
        evidence.append(dict(name=name,source=original,sha256=digest(name),parameters=params,macs=macs,**details(p,vy)))
        ensemble.append(probabilities(model,vx,10))
    historical=json.loads((SOURCE/'results/final_recipe.json').read_text())
    measured=details(sum(ensemble)/len(ensemble),vy)
    assert abs(measured['accuracy']-historical['val_accuracy'])<1e-6
    (OUT/'legacy_import.json').write_text(json.dumps(dict(complete=True,checkpoints=evidence,dual_ten_view=measured,
        historical_recipe=historical,historical_test_metrics=json.loads((SOURCE/'results/final_test_metrics.json').read_text()),
        note='Earlier same-student, same-split models, not external pretraining. Test metrics are historical records, not a new evaluation. Source directory is read-only.'),indent=2))
    print(json.dumps(dict(checkpoints=evidence,dual_ten_view=measured),indent=2),flush=True)


if __name__=='__main__':
    run()
