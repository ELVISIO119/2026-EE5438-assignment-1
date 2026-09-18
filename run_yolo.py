"""Run the fixed detector-inspired MLP comparisons and matched clean refinement."""
import json
import subprocess
import sys
from pathlib import Path

from train import OUT

STEMS=['yolo_pyramid_control','yolo_pyramid_fusion','yolo_pyramid_aux','yolo_pyramid_csp']


def train(path,force=False):
    cfg=json.loads(path.read_text())
    result_path=OUT/f"{cfg['name']}.json"
    if result_path.exists() and not force:
        result=json.loads(result_path.read_text())
        assert result['complete'] and result['config']==cfg, f'Incomplete or changed run: {result_path}'
        assert all((OUT/f"{cfg['name']}{suffix}.pt").exists() for suffix in ('','_ema','_swa'))
        print('Reuse completed run:',cfg['name'],flush=True)
        return
    print('Start:',cfg['name'],'hard timeout: 1800 seconds',flush=True)
    subprocess.run([sys.executable,'train.py',str(path)],check=True,timeout=1800)
    assert json.loads(result_path.read_text())['complete']


def run(force=False):
    for name in STEMS:
        train(Path('configs')/f'{name}.json',force)
    for name in STEMS:
        candidates={name+s:json.loads((OUT/f'{name+s}.json').read_text()) for s in ('','_ema','_swa')}
        parent=min(candidates,key=lambda n:(-candidates[n]['val_accuracy'],candidates[n]['val_loss']))
        cfg=json.loads((Path('configs')/f'{name}.json').read_text())
        cfg.update(name=name+'_clean',epochs=30,lr=.00005,decay=.01,augmentation=False,label_smoothing=0.,init=parent,
                   hypothesis=f'Matched 30-epoch clean refinement of {name}; initialize its best raw-validation ordinary/EMA/SWA checkpoint. Same clean schedule and learning rate for all four architectures; auxiliary supervision remains training-only when configured.')
        path=Path('configs')/f"{cfg['name']}.json"
        if path.exists() and not force:
            assert json.loads(path.read_text())==cfg
        else:
            path.write_text(json.dumps(cfg,indent=2))
        train(path,force)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--retrain',action='store_true',help='Explicitly replace recorded outputs with new training runs.')
    run(parser.parse_args().retrain)
