"""Paired focal-gamma-two and mild upper-garment weighted-CE follow-up."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation import load_model, probabilities
from garment_experiment import details
from run_yolo import train
from select_yolo import run as select
from train import OUT, load_data

STEMS=['features_gray','loss_focal2','loss_weighted']


def run(force=False):
    for stem in STEMS:
        train(Path('configs')/f'{stem}.json',force if stem!='features_gray' else False)
    x,y,vx,vy,_,_=load_data(); del x,y
    rows=[]
    for stem in STEMS:
        for suffix in ('','_ema','_swa'):
            name=stem+suffix
            p=probabilities(load_model(name),vx,1)
            pred=p.argmax(1); labels=vy.cpu()
            row=dict(name=name,**details(p,vy),classes=[])
            for cls in range(10):
                tp=int(((pred==cls)&(labels==cls)).sum())
                row['classes'].append(dict(label=cls,correct=tp,recall=tp/int((labels==cls).sum()),
                    precision=tp/max(1,int((pred==cls).sum()))))
            rows.append(row)
    selected=[min((r for r in rows if r['name'] in (stem,stem+'_ema',stem+'_swa')),
                  key=lambda r:(-r['correct'],r['loss'])) for stem in STEMS]
    control=selected[0]
    supported=[r['name'] for r in selected[1:] if r['correct']>control['correct'] and
               r['shirt_f1']>=control['shirt_f1'] and r['shirt_recall']>=control['shirt_recall'] and
               all(a>=b for a,b in zip(r['validation_half_correct'],control['validation_half_correct']))]
    result=dict(complete=True,checkpoints=rows,selected_families=selected,supported_for_local_augmentation=supported,
                completed_at=datetime.now(timezone.utc).isoformat(),
                rule='Advance local-augmentation stacking only if the best raw-validation checkpoint improves total correct and preserves Shirt recall/F1 and both development-half counts relative to matched CE. Otherwise retain separate negative results.',
                limitations='Single seed, repeated validation reuse. Class weights emphasize difficulty, not imbalance. No claim that recall must improve or that other classes remain unchanged. No new test evaluation.')
    (OUT/'loss_experiment.json').write_text(json.dumps(result,indent=2))
    select(STEMS,prefix='loss')
    print(json.dumps(dict(selected=selected,supported_for_local_augmentation=supported),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--retrain',action='store_true')
    run(parser.parse_args().retrain)
