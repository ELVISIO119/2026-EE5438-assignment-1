"""Three-fold, same-student-seed development evaluation; never loads the test set."""
import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import FashionMNIST

from evaluation import probabilities
from train import DEVICE, OUT, SEED, Model, run, seed_all

CV = OUT/'cross_validation'
FAMILIES = {'adamw': 'configs/mixer.json', 'muon': 'configs/mixer_muon.json'}
PROTOCOL = ('Three stratified folds of the official 60,000 training images: 40,000 fitting and '
            '20,000 held out per fold. Split RNG and every fresh training RNG use student ID 58561440, '
            'never seed+fold. Compare the existing compact Mixer AdamW and Muon+AdamW configurations '
            'at their fixed 120 epochs. Report the final epoch, without early stopping or held-out '
            'checkpoint selection. Normalize using only each fitting fold. No warm start, teacher '
            'or inherited weights. Report both predeclared one-view and ten-view OOF predictions. '
            'No test access or deployment replacement. Architectures were previously developed on '
            'a subset of these images: this is development cross-validation, not nested unbiased '
            'assessment or independent-seed replication. Fold models are not an inference ensemble.')


def splits(labels):
    folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(labels)), labels))
    seen = np.zeros(len(labels), np.int8)
    for train_ids, valid_ids in folds:
        assert not np.intersect1d(train_ids, valid_ids).size
        assert len(train_ids)+len(valid_ids) == len(labels)
        seen[valid_ids] += 1
        assert np.ptp(np.bincount(labels[valid_ids], minlength=10)) == 0
    assert np.all(seen == 1)
    return folds


def metrics(p, y):
    pred = p.argmax(1)
    report = classification_report(y, pred, output_dict=True, zero_division=0)
    return dict(correct=int((pred == y).sum()), examples=len(y), accuracy=float((pred == y).mean()),
                macro_f1=report['macro avg']['f1-score'], shirt_f1=report['6']['f1-score'],
                loss=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()))


def execute():
    CV.mkdir(exist_ok=True, parents=True)
    configs = {family: json.loads(Path(path).read_text()) for family, path in FAMILIES.items()}
    assert all(c['epochs'] == 120 and not any(c.get(k) for k in ('init', 'teacher', 'averages')) for c in configs.values())
    dataset = FashionMNIST('data', train=True, download=True)
    labels = dataset.targets.numpy()
    folds = splits(labels)
    code = hashlib.sha256(b''.join(Path(n).read_bytes() for n in
        ('cross_validate.py', 'train.py', 'models.py', 'evaluation.py'))).hexdigest()
    dataset_hash = hashlib.sha256(dataset.data.numpy().tobytes()+labels.tobytes()).hexdigest()
    protocol = dict(seed=SEED, folds=3, configs=configs, code_sha256=code, dataset_sha256=dataset_hash,
                    protocol=PROTOCOL)
    protocol_path = CV/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Resume requires unchanged code/data/protocol.'
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2))
    fold_ids = np.full(len(labels), -1, np.int8)
    oof = {f'{family}_{views}': np.full((len(labels), 10), np.nan, np.float32)
           for family in configs for views in (1, 10)}
    rows = []; initial_hashes = set()
    for fold, (train_ids, valid_ids) in enumerate(folds):
        fold_ids[valid_ids] = fold
        x = dataset.data[train_ids].unsqueeze(1).float().div(255).to(DEVICE)
        vx = dataset.data[valid_ids].unsqueeze(1).float().div(255).to(DEVICE)
        y = dataset.targets[train_ids].to(DEVICE); vy = dataset.targets[valid_ids].to(DEVICE)
        mean, std = x.mean().item(), x.std().item()
        split_hash = hashlib.sha256(train_ids.tobytes()+valid_ids.tobytes()).hexdigest()
        for family, source in configs.items():
            cfg = dict(source, name=f'cv_{family}_fold{fold+1}',
                       hypothesis=PROTOCOL)
            seed_all()
            initial = Model(cfg, mean, std)
            init_hash = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in initial.parameters())).hexdigest()
            initial_hashes.add(init_hash); del initial
            assert len(initial_hashes) == 1, 'All matched folds/optimizers must have the same initialized weights.'
            record_path = CV/f'{cfg["name"]}.json'
            checkpoint_path = CV/f'{cfg["name"]}.pt'
            record = json.loads(record_path.read_text()) if record_path.exists() else {}
            if record.get('complete'):
                assert record['config'] == cfg and record['seed'] == SEED
                assert record['checkpoint_selection'] == 'fixed_last_epoch' and record['best_epoch'] == 120
                print('Reuse completed fold:', cfg['name'], flush=True)
            else:
                # ponytail: resume at completed folds; an interrupted fold restarts at the same seed.
                record = run(cfg, data=(x, y, vx, vy, mean, std), output_dir=CV, last_epoch=True)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            assert checkpoint['config'] == cfg
            assert abs(checkpoint['mean']-mean) < 1e-7 and abs(checkpoint['std']-std) < 1e-7
            model = Model(cfg, mean, std).to(DEVICE).eval(); model.load_state_dict(checkpoint['state_dict'])
            for views in (1, 10):
                p = probabilities(model, vx, views, 4).numpy()
                score = metrics(p, labels[valid_ids])
                if views == 1: assert abs(score['accuracy']-record['val_accuracy']) < 1e-6
                oof[f'{family}_{views}'][valid_ids] = p
                rows.append(dict(family=family, fold=fold+1, views=views, seed=SEED, **score,
                    parameters=record['parameters'], macs=record['macs']*views, seconds=record['seconds'],
                    train_examples=len(train_ids), mean=mean, std=std, split_sha256=split_hash,
                    initialization_sha256=init_hash, checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    checkpoint=str(checkpoint_path), checkpoint_selection='fixed_last_epoch', epoch=120))
                print('OOF fold:', family, fold+1, views, score, flush=True)
            del model, checkpoint
        del x, vx, y, vy
    assert np.all(fold_ids >= 0) and all(np.isfinite(p).all() for p in oof.values())
    np.savez_compressed(OUT/'cv_oof_predictions.npz', labels=labels, fold_ids=fold_ids, **oof)
    pd.DataFrame(rows).to_csv(OUT/'cv_folds.csv', index=False)
    summaries = []
    for family in configs:
        for views in (1, 10):
            selected = [r for r in rows if r['family'] == family and r['views'] == views]
            acc = [r['accuracy'] for r in selected]
            summaries.append(dict(family=family, views=views, **metrics(oof[f'{family}_{views}'], labels),
                fold_accuracies=acc, fold_mean=float(np.mean(acc)), fold_sample_std=float(np.std(acc, ddof=1)),
                parameters=selected[0]['parameters'], macs=selected[0]['macs']))
    result = dict(complete=True, seed=SEED, protocol=protocol, summaries=summaries, folds=rows,
        initial_weights_identical=len(initial_hashes) == 1,
        training_seconds=sum(r['seconds'] for r in rows if r['views'] == 1),
        completed_at=datetime.now(timezone.utc).isoformat(),
        comparison='Fold standard deviation describes variation across data partitions at one seed, not seed uncertainty. '
                   'OOF predictions come from different models; this score is not a deployable ensemble or official test score. '
                   'Training uses fewer images than the existing 54,000-image development runs, so scores are not directly comparable.')
    (OUT/'cv_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summaries, indent=2), flush=True)


def check_results():
    result = json.loads((OUT/'cv_summary.json').read_text())
    with np.load(OUT/'cv_oof_predictions.npz') as saved:
        assert result['complete'] and result['seed'] == SEED and result['initial_weights_identical']
        labels = saved['labels']; assignments = saved['fold_ids']
        assert len(labels) == 60000 and np.array_equal(np.bincount(labels), np.full(10, 6000))
        for fold, (_, ids) in enumerate(splits(labels)):
            assert np.all(assignments[ids] == fold)
        for summary in result['summaries']:
            p = saved[f'{summary["family"]}_{summary["views"]}']
            assert p.shape == (60000, 10) and np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5)
            actual = metrics(p, labels)
            assert actual['correct'] == summary['correct'] and abs(actual['macro_f1']-summary['macro_f1']) < 1e-12
            for row in result['folds']:
                if row['family'] == summary['family'] and row['views'] == summary['views']:
                    mask = assignments == row['fold']-1
                    assert metrics(p[mask], labels[mask])['correct'] == row['correct']
                    assert row['seed'] == SEED and row['epoch'] == 120 and row['checkpoint_selection'] == 'fixed_last_epoch'
    print('Recorded OOF coverage, student seed, fixed-epoch records and aggregate/class metrics: PASS')


def self_check():
    labels = np.tile(np.arange(10), 12)
    folds = splits(labels)
    assert all(np.array_equal(a, b) for pair, repeat in zip(folds, splits(labels)) for a, b in zip(pair, repeat))
    cfg = dict(name='smoke', model='baseline', epochs=3, lr=.001, decay=0., schedule=False,
               hypothesis='Temporary fixed-seed training-engine check.')
    seed_all(); x = torch.rand(30, 1, 28, 28, device=DEVICE); y = torch.arange(30, device=DEVICE) % 10
    data = (x[:20], y[:20], x[20:], y[20:], x[:20].mean().item(), x[:20].std().item())
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='ee5438_cv_check_') as directory:
        try:
            os.chdir(directory)
            for final in (False, True):
                record = run(cfg, data=data, output_dir=Path(directory)/str(final), last_epoch=final)
                expected = record['history'][-1] if final else max(record['history'], key=lambda r: (r['val_accuracy'], -r['val_loss']))
                assert record['best_epoch'] == expected['epoch'] and record['val_loss'] == expected['val_loss']
                ckpt = torch.load(Path(directory)/str(final)/'smoke.pt', map_location='cpu', weights_only=True)
                model = Model(cfg, ckpt['mean'], ckpt['std']).to(DEVICE); model.load_state_dict(ckpt['state_dict'])
                p = probabilities(model, data[2]).numpy()
                assert abs(metrics(p, data[3].cpu().numpy())['loss']-expected['val_loss']) < 1e-6
        finally:
            os.chdir(original)
    print('Disjoint stratified folds, exactly-once coverage, fixed seed and both checkpoint modes: PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    self_check() if args.check else check_results() if args.verify else execute()
