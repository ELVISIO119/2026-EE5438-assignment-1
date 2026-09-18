"""Seeded, resumable configuration-space search; candidate selection never loads test data."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluation import digest, load_model, probabilities
from garment_experiment import details
from hybrid_experiment import model_names, predict
from refine_cascade import guarded, select_candidates
from train import OUT, SEED, load_data

# Six gates, two relative mixture logits, three log-temperature multipliers.
# This is a compact configuration space, NOT a learned image/weight latent space.
LOW = np.array([.96, .70, .65, .65, .75, .50, -.8, -.8, -.3, -.3, -.3])
HIGH = np.array([.9999, .97, .97, .97, .99, .95, .8, .8, .3, .3, .3])
START = (np.array([.99, .85, .90, .90, .90, .70, 0, 0, 0, 0, 0])-LOW)/(HIGH-LOW)
PROTOCOL = ('11-dimensional bounded configuration-vector evolutionary search. Fixed five checkpoints, '
            'architecture, front equal weights, views and view batching. Seeded populations use 25% global '
            'uniform exploration, 25% baseline-local exploration and 50% Gaussian elite sampling. '
            'Cache-only vectorized screening; no test access. Preserve reference correct count, Shirt F1, '
            'both development-half correct counts, parameters and at least 1% average MAC reduction. '
            'Five diverse cached finalists receive actual routed validation and paired latency checks; '
            'require at least 5% lower median latency on both first/last 1024 validation images. '
            'Final ranking: correct count, MACs, first-slice latency, NLL. Reused validation is development '
            'data, not an independent holdout; searching more configurations can overfit it.')


def decode(z, base):
    v = LOW + np.asarray(z)*(HIGH-LOW)
    recipe = {k: copy.deepcopy(base[k]) for k in
              ('front', 'back', 'threshold', 'classes', 'parameters', 'pre_exit', 'class_thresholds')}
    recipe['pre_exit']['threshold'] = float(v[0])
    recipe['class_thresholds'] = [float(v[1] if c in (0, 2, 4, 6) else v[2] if c == 3 else v[3]) for c in range(10)]
    recipe['back']['cascade']['threshold'] = float(v[4])
    recipe['back']['view_exit']['threshold'] = float(v[5])
    weights = np.array([c['weight'] for c in base['back']['components']])*np.exp([v[6], v[7], 0])
    weights /= weights.sum()
    for i, c in enumerate(recipe['back']['components']):
        c['weight'] = float(weights[i])
        c['temperature'] *= float(np.exp(v[8+i]))
    return recipe


def score_batch(z, base, cache, labels, return_probabilities=False):
    """Broadcast each population chunk over cached validation predictions, with charged routes."""
    v = (LOW + np.asarray(z)*(HIGH-LOW)).astype(np.float32)
    members = base['back']['components']
    weights = np.array([c['weight'] for c in members], np.float32)[None, :]*np.exp(
        np.column_stack((v[:, 6:8], np.zeros(len(v), np.float32))))
    weights /= weights.sum(1, keepdims=True)
    initial = np.zeros((len(v), len(labels), 10), np.float32)
    full = np.zeros_like(initial)
    for i, c in enumerate(members):
        temp = c['temperature']*np.exp(v[:, 8+i])
        for target, views in ((initial, 10), (full, c['views'])):
            logits = np.log(cache[c['name'], views].clip(1e-12))[None]/temp[:, None, None]
            logits -= logits.max(2, keepdims=True)
            p = np.exp(logits); p /= p.sum(2, keepdims=True)
            target += weights[:, i, None, None]*p
    view_exit = initial.max(2) >= v[:, 5, None]
    output = np.where(view_exit[:, :, None], initial, full)
    gate = base['back']['cascade']
    cheap = cache[gate['name'], 1]
    gate_exit = (cheap.max(1)[None] >= v[:, 4, None]) & np.isin(cheap.argmax(1), gate['classes'])[None]
    output = np.where(gate_exit[:, :, None], cheap[None], output)
    first = cache[base['pre_exit']['name'], 1]
    other = next(c['name'] for c in base['front']['components'] if c['name'] != base['pre_exit']['name'])
    second = cache[other, 1]
    pair = (first+second)*.5
    classes = pair.argmax(1)
    group = np.where(np.isin(classes, [0, 2, 4, 6]), 1, np.where(classes == 3, 2, 3))
    pre_exit = first.max(1)[None] >= v[:, 0, None]
    pair_exit = (pair.max(1)[None] >= v[:, group]) & (first.argmax(1) == second.argmax(1))[None]
    front_exit = pre_exit | pair_exit
    output = np.where(pair_exit[:, :, None], pair[None], output)
    output = np.where(pre_exit[:, :, None], first[None], output)
    front_cost = base['pre_exit']['macs']+(~pre_exit)*(base['front']['macs']-base['pre_exit']['macs'])
    back_cost = gate['macs']+(~gate_exit)*(base['back']['full_macs']-view_exit*base['back']['view_exit']['remaining_macs'])
    costs = (front_cost+(~front_exit)*back_cost).mean(1)
    predictions = output.argmax(2); hit = predictions == labels[None]
    tp = ((predictions == 6) & (labels[None] == 6)).sum(1)
    f1 = 2*tp/((predictions == 6).sum(1)+(labels == 6).sum())
    losses = -np.log(output[:, np.arange(len(labels)), labels].clip(1e-12)).mean(1)
    rows = [dict(correct=int(hit[i].sum()), accuracy=float(hit[i].mean()), shirt_f1=float(f1[i]),
                 validation_half_correct=[int(hit[i, :len(labels)//2].sum()), int(hit[i, len(labels)//2:].sum())],
                 loss=float(losses[i]), macs=float(costs[i]), parameters=base['parameters']) for i in range(len(v))]
    return (rows, output) if return_probabilities else rows


def rank(row, base):
    deficit = max(0, base['correct']-row['correct']) + sum(max(0, b-a) for a, b in
        zip(row['validation_half_correct'], base['validation_half_correct']))
    deficit += 6000*max(0, base['shirt_f1']-row['shirt_f1'])
    deficit += 100*max(0, row['macs']/base['macs']-.99)
    return (not guarded(row, base, .99), deficit, -row['correct'], row['macs'], row['loss'])


@torch.inference_mode()
def run(args):
    base = json.loads((OUT/'adaptive_balanced_recipe.json').read_text())
    # Fail rather than silently applying a decoder to a different front/view topology.
    assert base['pre_exit']['name'] == 'legacy_clean'
    assert [c['views'] for c in base['back']['components']] == [10, 30, 10]
    assert all(c['views'] == 1 and c['weight'] == .5 and c['temperature'] == 1 for c in base['front']['components'])
    train_x, train_y, vx, vy, _, _ = load_data(); del train_x, train_y
    labels = vy.cpu().numpy()
    hashes = {n: digest(n) for n in sorted(model_names(base))}
    code_hash = hashlib.sha256(b''.join(Path(n).read_bytes() for n in
        ('models.py', 'train.py', 'evaluation.py', 'hybrid_experiment.py'))).hexdigest()
    split_hash = hashlib.sha256(vx.cpu().numpy().tobytes()+labels.tobytes()).hexdigest()
    identity = dict(checkpoints=hashes, code=code_hash, split=split_hash, torch=str(torch.__version__),
                    device=torch.cuda.get_device_name() if vx.is_cuda else 'CPU', view_batch=4)
    cache_dir = OUT/'validation_cache'; cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache_path = cache_dir/f'latent_{key}.pt'
    models = {n: load_model(n) for n in hashes}
    started = time.perf_counter()
    if cache_path.exists():
        saved = torch.load(cache_path, weights_only=True)
        assert saved['identity'] == identity
        tensors = saved['probabilities']
        print('Reused hash-bound validation cache:', cache_path, flush=True)
    else:
        required = {(c['name'], views) for c in base['back']['components'] for views in (10, c['views'])}
        required |= {(c['name'], 1) for c in base['front']['components']} | {(base['back']['cascade']['name'], 1)}
        tensors = {}
        for name, views in sorted(required):
            tensors[name, views] = probabilities(models[name], vx, views, 4)
            print('Cached:', name, views, flush=True)
        torch.save(dict(identity=identity, probabilities=tensors), cache_path)
    assert all(p.shape == (len(labels), 10) and torch.isfinite(p).all() and
               torch.allclose(p.sum(1), torch.ones(len(labels)), atol=1e-5) for p in tensors.values())
    cache = {k: p.numpy() for k, p in tensors.items()}
    cache_seconds = time.perf_counter()-started
    baseline = score_batch([START], base, cache, labels)[0]
    actual_p, actual_cost = predict(base, vx, models)
    actual = dict(details(actual_p, vy), macs=actual_cost['macs_per_image'])
    assert actual['correct'] == base['correct'] and abs(actual['macs']-base['macs']) < 1
    print('Baseline cached/actual correct:', baseline['correct'], actual['correct'], flush=True)
    settings = dict(seed=SEED, rounds=args.rounds, population=args.population, identity=identity,
                    reference_sha256=hashlib.sha256((OUT/'adaptive_balanced_recipe.json').read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), protocol=PROTOCOL)
    state_path = cache_dir/'latent_search_state.json'
    rng = np.random.default_rng(SEED)
    rows = []; start_round = 0; center = START.copy(); spread = np.full(11, .18)
    elapsed = 0.
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text())
        assert state['settings'] == settings, 'Resume needs identical code, data, recipe and budget.'
        rows = state['rows']; start_round = state['round']; center = np.array(state['center'])
        spread = np.array(state['spread']); rng.bit_generator.state = state['rng']; elapsed = state['seconds']
        print('Resuming after candidates:', len(rows), flush=True)
    for round_id in range(start_round, args.rounds):
        tick = time.perf_counter(); q = args.population//4
        proposals = np.concatenate((rng.uniform(size=(q, 11)),
            np.clip(START+rng.normal(size=(q, 11))*.15, 0, 1),
            np.clip(center+rng.normal(size=(args.population-2*q, 11))*spread, 0, 1)))
        if round_id == 0: proposals[0] = START
        for offset in range(0, len(proposals), 32):
            chunk = proposals[offset:offset+32]
            for z, row in zip(chunk, score_batch(chunk, base, cache, labels)):
                rows.append(dict(row, z=z.tolist(), candidate_id=len(rows), generation=round_id))
        leaders = sorted(rows, key=lambda r: rank(r, base))[:max(8, args.population//8)]
        elite = np.array([r['z'] for r in leaders])
        center = elite.mean(0); spread = np.maximum(elite.std(0), .025)
        elapsed += time.perf_counter()-tick
        state = dict(settings=settings, rows=rows, round=round_id+1, center=center.tolist(),
                     spread=spread.tolist(), rng=rng.bit_generator.state, seconds=elapsed)
        temporary = state_path.with_suffix('.tmp'); temporary.write_text(json.dumps(state)); temporary.replace(state_path)
        print(f'Round {round_id+1}/{args.rounds}: {len(rows)} candidates; best correct={leaders[0]["correct"]}, '
              f'MACs={leaders[0]["macs"]:,.0f}; screening {elapsed:.2f}s', flush=True)
    pd.DataFrame(rows).to_csv(OUT/'latent_search_candidates.csv', index=False)
    eligible = [r for r in rows if guarded(r, base, .99)]
    # ponytail: five real finalists cap GPU work; expand only if cached-to-runtime drift warrants it.
    short = []
    orders = [sorted(eligible, key=lambda r: (-r['correct'], r['macs'], r['loss'])),
              sorted(eligible, key=lambda r: (r['macs'], -r['correct'], r['loss']))]
    seen = set()
    for i in range(5):
        for row in orders[i % 2]:
            signature = (row['correct'], round(row['macs']/2e6), *row['validation_half_correct'])
            if signature not in seen:
                short.append(row); seen.add(signature); break
    summaries = []
    for row in short:
        summaries.append(dict(decode(row['z'], base), **row, goal='balanced', label=f'latent:{row["candidate_id"]}'))
    summary = dict(complete=True, settings=settings, candidate_count=len(rows), eligible_count=len(eligible),
                   cache_seconds=cache_seconds, screening_seconds=elapsed, cached_baseline=baseline,
                   actual_baseline=actual, shortlist_ids=[r['candidate_id'] for r in short],
                   cache_file=str(cache_path), best_cached=sorted(rows, key=lambda r: rank(r, base))[:5])
    (OUT/'latent_search.json').write_text(json.dumps(summary, indent=2))
    select_candidates(summaries, dict(balanced=base), models, vx, vy, 'latent', .99, PROTOCOL)


def self_check():
    """Compare vectorized cache scoring with the real routed executor on view-invariant models."""
    class Toy(torch.nn.Module):
        def __init__(self, shift):
            super().__init__(); self.shift = shift
        def forward(self, x):
            # Max is preserved by the tested translations/scales of constant images.
            target = (x.amax((1, 2, 3))*9).round().long()
            logits = torch.zeros(len(x), 10)
            logits[torch.arange(len(x)), (target+self.shift) % 10] = 2+target.float()
            return logits
    base = json.loads((OUT/'adaptive_balanced_recipe.json').read_text())
    x = torch.linspace(.05, 1, 40)[:, None, None, None].expand(-1, 1, 28, 28)
    names = sorted(model_names(base)); models = {n: Toy(i % 2) for i, n in enumerate(names)}
    needed = {(c['name'], v) for c in base['back']['components'] for v in (10, c['views'])}
    needed |= {(c['name'], 1) for c in base['front']['components']} | {(base['back']['cascade']['name'], 1)}
    cache = {(n, v): probabilities(models[n], x, v, 4).numpy() for n, v in needed}
    labels = np.arange(len(x)) % 10
    for z in (START, np.zeros(11), np.ones(11)):
        rows, cached = score_batch([z], base, cache, labels, True)
        p, execution = predict(decode(z, base), x, models)
        assert np.allclose(p.numpy(), cached[0], atol=2e-6)
        assert abs(rows[0]['macs']-execution['macs_per_image']) < 1
        assert rows[0]['correct'] == int((p.argmax(1).numpy() == labels).sum())
        assert abs(rows[0]['shirt_f1']-details(p, torch.tensor(labels))['shirt_f1']) < 1e-12
    print('Vectorized probabilities, decoded routes, charged MACs and class metrics: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=16)
    parser.add_argument('--population', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1 or args.population < 32: parser.error('rounds >= 1 and population >= 32 required')
    self_check() if args.check else run(args)
