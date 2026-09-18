"""Time the frozen cascade and its full reference on the same 1,024 validation inputs."""
import json
from datetime import datetime,timezone

import torch
from torch.utils.benchmark import Timer
from train import OUT,load_data
from evaluation import load_model,recipe_probabilities


@torch.inference_mode()
def run():
    recipes={'full':json.loads((OUT/'cascade_reference.json').read_text())['recipe'],
             'cascade':json.loads((OUT/'cascade_recipe.json').read_text())}
    x,y,vx,vy,_,_=load_data()
    del x,y,vy
    sample=vx[:1024]
    models={c['name']:load_model(c['name']) for c in recipes['full']['components']}
    result=dict(environment=dict(torch=torch.__version__,device=torch.cuda.get_device_name()),
                examples=len(sample),timing={},method='Native torch.utils.benchmark.Timer CUDA synchronization; same GPU-resident validation inputs, all original-precision models preloaded; includes views, routing, aggregation and CPU output, excludes load/compile. Shared GPU, descriptive measurement.')
    for key,recipe in recipes.items():
        fn=lambda:recipe_probabilities(recipe,sample,models)
        for _ in range(3):
            fn()
        measurement=Timer(stmt='fn()',globals={'fn':fn}).blocked_autorange(min_run_time=2.)
        _,execution=fn()
        result['timing'][key]=dict(median_ms=measurement.median*1000,iqr_ms=measurement.iqr*1000,
                                  samples_seconds=measurement.raw_times,number_per_run=measurement.number_per_run,
                                  images_per_second=len(sample)/measurement.median,execution=execution)
    result['speedup']=result['timing']['full']['median_ms']/result['timing']['cascade']['median_ms']
    result['completed_at']=datetime.now(timezone.utc).isoformat()
    (OUT/'cascade_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    run()
