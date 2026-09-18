"""Freeze an accuracy-first, lower-expected-FLOP cascade using validation only."""
import argparse
import json
from datetime import datetime,timezone

import pandas as pd
import torch
from train import OUT,load_data
from evaluation import load_model,probabilities,recipe_probabilities,early_mask,digest
from garment_experiment import details


def run(source):
    recipe=json.loads(source.read_text())
    assert 'cascade' not in recipe
    x,y,vx,vy,_,_=load_data()
    del x,y
    vy=vy.cpu()
    original,_=recipe_probabilities(recipe,vx)
    base=details(original,vy)
    assert base['correct']==recipe['correct']
    rows=[dict(**base,macs=recipe['macs'],early_exit=0,cascade=None)]
    for name in dict.fromkeys(c['name'] for c in recipe['components']):
        record=json.loads((OUT/f'{name}.json').read_text())
        model=load_model(name)
        for views in (1,2):
            p=probabilities(model,vx,views)
            for threshold in (.9,.95,.975,.99,.995,.999,.9999):
                cascade=dict(name=name,views=views,threshold=threshold,classes=[1,5,7,8,9],
                             macs=record['macs']*views,sha256=digest(name))
                mask=early_mask(p,cascade)
                mixed=original.clone(); mixed[mask]=p[mask]
                macs=cascade['macs']+(1-float(mask.sum())/len(mask))*recipe['macs']
                rows.append(dict(**details(mixed,vy),macs=macs,early_exit=int(mask.sum()),cascade=cascade))
        del model
    acceptable=[r for r in rows if r['shirt_f1']>=base['shirt_f1'] and
                all(a>=b for a,b in zip(r['validation_half_correct'],base['validation_half_correct']))]
    selected=min(acceptable,key=lambda r:(-r['correct'],r['macs'],r['loss']))
    frozen=dict(recipe,**{k:v for k,v in selected.items() if k not in ('cascade','early_exit')})
    if selected['cascade']:
        frozen.update(cascade=selected['cascade'],full_macs=recipe['macs'],
                      worst_case_macs=recipe['macs']+selected['cascade']['macs'],
                      name=recipe['name']+f"; early exit {selected['cascade']['name']} at {selected['cascade']['threshold']}")
    # Recompute actual routing: repacking hard examples can change floating-point kernels.
    actual,execution=recipe_probabilities(frozen,vx)
    measured=details(actual,vy)
    assert measured['correct']==selected['correct'],(measured,selected)
    assert measured['shirt_f1']>=base['shirt_f1']
    assert all(a>=b for a,b in zip(measured['validation_half_correct'],base['validation_half_correct']))
    frozen.update(**measured,frozen_at=datetime.now(timezone.utc).isoformat(),candidates=len(rows),
                  objective='accuracy_then_expected_cost',
                  rule='Maximize validation correct count, then minimize expected MACs and NLL. No class label is available to runtime gating. Guard Shirt F1 and correct counts on both development halves.',
                  cost_note='Logical parameter count is unchanged: first-stage weights are shared with an ensemble member. MACs are validation-average conditional cost; full and worst-case costs are also reported.',
                  validation_execution=execution)
    (OUT/'cascade_recipe.json').write_text(json.dumps(frozen,indent=2))
    (OUT/'cascade_reference.json').write_text(json.dumps(dict(recipe=recipe,validation=base),indent=2))
    pd.DataFrame(rows).to_csv(OUT/'cascade_candidates.csv',index=False)
    torch.save(actual,OUT/'cascade_validation.pt')
    print(json.dumps(frozen,indent=2),flush=True)
    with open('JOURNAL.md','a') as journal:
        journal.write(f"\n## {frozen['frozen_at']} - Confidence-gated inference\n\n"
                      f"Tested {len(rows)} validation recipes. Selected {frozen['correct']}/6000 correct, "
                      f"Shirt F1 {frozen['shirt_f1']:.6f}, {execution['early_exit']} early exits, "
                      f"{execution['macs_per_image']:,.0f} validation-average MACs and {execution['worst_case_macs']:,} worst-case MACs/image. "
                      "Recomputed the actual routed pipeline before freezing; no test labels were loaded. "
                      "The gate uses only model confidence and predicted non-garment class, and shares existing weights.\n")


if __name__=='__main__':
    from pathlib import Path
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',type=Path,default=OUT/'spatial_recipe.json')
    run(parser.parse_args().recipe)
