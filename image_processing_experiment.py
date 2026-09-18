"""Bounded validation-only input-processing comparison against the frozen cascade."""
import json
from datetime import datetime, timezone

import torch
from torch.nn import functional as F

from evaluation import recipe_probabilities, load_model, digest
from garment_experiment import details
from train import OUT, load_data


def transform(x, name):
    if name == 'original':
        return x
    if name.startswith('gamma_'):
        return x.pow(float(name.split('_')[1]))
    if name.startswith('contrast_'):
        return (x * float(name.split('_')[1])).clamp(0, 1)
    if name == 'center_half_pixel':
        # Bounded center-of-intensity correction; no label-dependent processing.
        mass = x.sum((2, 3)).clamp_min(1e-8)
        axis = torch.arange(28, device=x.device, dtype=x.dtype)
        cx = (x.sum(2) * axis).sum(2) / mass
        cy = (x.sum(3) * axis).sum(2) / mass
        theta = torch.eye(2, 3, device=x.device).repeat(len(x), 1, 1)
        theta[:, 0, 2] = (cx[:, 0] - 13.5).clamp(-.5, .5) * 2 / 28
        theta[:, 1, 2] = (cy[:, 0] - 13.5).clamp(-.5, .5) * 2 / 28
        return F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
    raise ValueError(name)


@torch.inference_mode()
def run():
    recipe = json.loads((OUT / 'final_recipe.json').read_text())
    x, y, vx, vy, _, _ = load_data()
    del x, y
    models = {c['name']: load_model(c['name']) for c in recipe['components']}
    assert all(digest(c['name']) == c['sha256'] for c in recipe['components'])
    names = ['original', 'gamma_0.95', 'gamma_1.05', 'contrast_0.95', 'contrast_1.05', 'center_half_pixel']
    result = dict(started_at=datetime.now(timezone.utc).isoformat(), recipe=recipe,
                  protocol='Six prespecified input pipelines; frozen weights, views, temperatures and gate. Validation only; no test data. Selection guards both validation halves and Shirt F1. Input processing precedes existing TTA.', candidates=[])
    for name in names:
        transformed = transform(vx, name)
        assert transformed.shape == vx.shape and torch.isfinite(transformed).all()
        assert transformed.min() >= 0 and transformed.max() <= 1
        p, execution = recipe_probabilities(recipe, transformed, models)
        row = dict(name=name, **details(p, vy), execution=execution)
        result['candidates'].append(row)
        torch.save(p, OUT / f'processing_{name}_validation.pt')
        print(json.dumps(row), flush=True)
        (OUT / 'image_processing.json').write_text(json.dumps(result, indent=2))
    baseline = result['candidates'][0]
    assert baseline['correct'] == recipe['correct']
    eligible = [r for r in result['candidates'] if r['correct'] > baseline['correct']
                and r['shirt_f1'] >= baseline['shirt_f1']
                and all(a >= b for a, b in zip(r['validation_half_correct'], baseline['validation_half_correct']))]
    result['eligible_improvements'] = [r['name'] for r in eligible]
    result['complete'] = True
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    (OUT / 'image_processing.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    # Check shift direction with an off-center impulse and preservation of blank input.
    sample = torch.zeros(2, 1, 28, 28)
    sample[0, 0, 13, 15] = 1
    shifted = transform(sample, 'center_half_pixel')
    assert shifted[0, 0, 13, 14] > 0 and shifted[1].sum() == 0
    assert torch.equal(transform(sample, 'original'), sample)
    run()
