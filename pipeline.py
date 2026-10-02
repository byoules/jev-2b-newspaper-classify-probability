"""Evaluate the real Open-Jev decision checkpoint on labeled Kaggle headlines."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parent
CODE_COMMIT = '820e2a7cacf2ac906b1a5590ee9a1a8afc1d0312'


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'id', 'headline', 'true_label'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path.name}: required columns are id, headline, true_label')
        rows = [{k: (v or '') for k, v in row.items() if k is not None} for row in reader]
    if not rows:
        raise ValueError(f'{path.name}: the dataset is empty')
    return rows


def find_dataset(folder):
    matches = []
    for path in sorted(folder.rglob('*.csv')):
        if 'kaggle-cache' in path.relative_to(folder).parts:
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            names = csv.DictReader(stream).fieldnames or []
        if {'id', 'headline', 'true_label'} <= set(names):
            matches.append(path)
    if len(matches) > 1:
        raise ValueError('Multiple matching CSV files. Set local_csv in settings.json to choose one.')
    return matches[0] if matches else None


def obtain_dataset(settings):
    if settings['local_csv']:
        path = local_path(settings['local_csv'])
        read_csv(path)
        return path
    folder = local_path(settings['data_folder'])
    folder.mkdir(parents=True, exist_ok=True)
    path = find_dataset(folder)
    if path:
        print(f'Reusing dataset: {path}', flush=True)
        return path
    print(f"Downloading Kaggle dataset {settings['dataset']}...", flush=True)
    import kagglehub
    try:
        # The SDK uses configured credentials for private/consent-gated data.
        downloaded = Path(kagglehub.dataset_download(settings['dataset'], output_dir=str(folder)))
    except Exception as exc:
        # Do not print SDK exception bodies, which may include signed URLs.
        raise RuntimeError('Kaggle download failed (' + type(exc).__name__ + '). '
                           'Check connectivity and dataset access. If credentials are required, '
                           'configure ~/.kaggle/access_token or ~/.kaggle/kaggle.json, '
                           'or download the CSV into data/. No replacement data was created.') from None
    path = find_dataset(downloaded if downloaded.is_dir() else folder)
    if path is None:
        raise ValueError('Downloaded dataset has no CSV with id, headline, true_label.')
    return path


def audit_rows(rows, labels):
    ids = Counter(r['id'] for r in rows if r['id'])
    headlines = Counter(r['headline'].strip() for r in rows if r['headline'].strip())
    issues = []
    for number, row in enumerate(rows, 1):
        reasons = []
        if not row['id'].strip(): reasons.append('missing_id')
        if not row['headline'].strip(): reasons.append('missing_headline')
        if row['true_label'].strip().lower() not in labels: reasons.append('invalid_true_label')
        if row['id'] and ids[row['id']] > 1: reasons.append('duplicate_id')
        if row['headline'].strip() and headlines[row['headline'].strip()] > 1: reasons.append('duplicate_headline')
        if reasons:
            issues.append({'source_row': number, **row, 'issues': '; '.join(reasons)})
    return issues


def category_request(headline, settings):
    # Labels and identifiers are intentionally absent from this request.
    return {'state': headline, 'questions': {'topic': {
        'type': 'choice', 'instructions': settings['question'],
        'criteria': settings['categories']}}}


def validate_probabilities(answer, labels):
    probs = answer['probabilities']
    if set(probs) != set(labels):
        raise ValueError('Model returned a different category set')
    values = [float(probs[label]) for label in labels]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('Model returned invalid probabilities')
    if not math.isclose(sum(values), 1, abs_tol=1e-6):
        raise ValueError('Probabilities do not sum to one')
    prediction = max(labels, key=lambda key: probs[key])
    if answer['choice'] != prediction:
        raise ValueError('Selected category disagrees with the probabilities')
    return dict(zip(labels, values)), prediction


def choose_device(settings):
    import torch
    if settings['device'] == 'cpu':
        print('CPU selected explicitly. This will be substantially slower.', flush=True)
        return 'cpu', {'device': 'cpu'}
    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch cannot access CUDA. If using a container, enable its GPU access. '
                           'Existing model services were not changed. Run bash run.sh --check for diagnostics.')
    gpus = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        gpus.append({'index': index, 'name': torch.cuda.get_device_name(index),
                     'free_gib': free / 2**30, 'total_gib': total / 2**30})
    print('GPU memory: ' + json.dumps(gpus), flush=True)
    minimum = float(settings['minimum_free_gpu_gib'])
    if settings['device'] == 'auto':
        usable = [g for g in gpus if g['free_gib'] >= minimum]
        if not usable:
            raise RuntimeError(f'No GPU has {minimum:g} GiB free. Existing services remain running. '
                               'Use another available GPU or rerun when memory is free.')
        selected = max(usable, key=lambda gpu: gpu['free_gib'])
    else:
        device = torch.device(settings['device'])
        if device.type != 'cuda' or device.index is None:
            raise ValueError('device must be auto, cpu, or cuda:N')
        if device.index >= len(gpus): raise ValueError('Requested CUDA device is not visible')
        selected = gpus[device.index]
        if selected['free_gib'] < minimum:
            raise RuntimeError(f"Requested GPU has only {selected['free_gib']:.2f} GiB free; need {minimum:g}.")
    return f"cuda:{selected['index']}", {'visible_gpus': gpus, 'selected': selected}


def checkpoint(settings):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    print('Checking/downloading the pinned Open-Jev adapter and decision head...', flush=True)
    required = ('model.json', 'head.pt', 'temperature.json',
                'adapter/adapter_config.json', 'adapter/adapter_model.safetensors')
    try:
        snapshot = Path(snapshot_download(settings['model'], revision=settings['model_revision'],
                                          local_files_only=True))
    except LocalEntryNotFoundError:
        snapshot = None
    if snapshot is None or not all((snapshot / 'package' / 'checkpoint' / n).is_file() for n in required):
        snapshot = Path(snapshot_download(settings['model'], revision=settings['model_revision'],
                                          allow_patterns=['package/checkpoint/**']))
    path = snapshot / 'package' / 'checkpoint'
    config = json.loads((path / 'model.json').read_text())
    if (config['model_id'], config['revision']) != (settings['base_model'], settings['base_revision']):
        raise ValueError('Checkpoint metadata differs from the expected pinned Qwen base')
    for name in required:
        if not (path / name).is_file(): raise ValueError(f'Checkpoint is incomplete: {name}')
    return path, config


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def evaluate(results, labels):
    valid = [r for r in results if r['status'] == 'ok' and r['true_label'].strip().lower() in labels]
    matrix = {truth: {pred: 0 for pred in labels} for truth in labels}
    for row in valid:
        matrix[row['true_label'].strip().lower()][row['predicted_label']] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[t][label] for t in labels)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        per_class[label] = {'precision': precision, 'recall': recall,
                            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                            'support': support}
    n = len(valid)
    # Multiclass Brier: sum across all three classes, then mean across records.
    brier = sum(sum((r['prob_' + label] - int(label == r['true_label'].strip().lower()))**2
                    for label in labels) for r in valid) / n if n else None
    log_loss = -sum(math.log(max(r['prob_' + r['true_label'].strip().lower()], 1e-15))
                    for r in valid) / n if n else None
    return {'total_rows': len(results), 'successfully_scored': sum(r['status'] == 'ok' for r in results),
            'failed_rows': sum(r['status'] != 'ok' for r in results), 'evaluated_rows': n,
            'accuracy': sum(r['correct'] is True for r in valid) / n if n else None,
            'multiclass_brier': brier, 'log_loss': log_loss,
            'macro_f1': sum(v['f1'] for v in per_class.values()) / len(labels) if n else None,
            'per_category': per_class, 'confusion_matrix': matrix}


def report_text(metrics, issues, labels):
    def fmt(v): return 'N/A' if v is None else f'{v:.4f}'
    lines = ['# News headline evaluation', '',
             f"Rows: {metrics['total_rows']}; scored: {metrics['successfully_scored']}; failed: {metrics['failed_rows']}; evaluated: {metrics['evaluated_rows']}",
             f"Rows with data issues (not silently dropped): {len(issues)}", '',
             f"Accuracy: {fmt(metrics['accuracy'])}", f"Macro F1: {fmt(metrics['macro_f1'])}",
             f"Multiclass Brier (sum over classes): {fmt(metrics['multiclass_brier'])}",
             f"Log loss (natural logarithm, clipped at 1e-15): {fmt(metrics['log_loss'])}", '',
             '| Category | Precision | Recall | F1 | Support |', '|---|---:|---:|---:|---:|']
    for label in labels:
        m = metrics['per_category'][label]
        lines.append(f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    lines += ['', 'Confusion matrix: rows are true labels; columns are predictions.', '',
              '| True / predicted | ' + ' | '.join(labels) + ' |', '|---|' + '---:|' * len(labels)]
    for label in labels:
        lines.append('| ' + label + ' | ' + ' | '.join(str(metrics['confusion_matrix'][label][p]) for p in labels) + ' |')
    lines += ['', 'This is an exploratory evaluation on the supplied headlines. Small samples cannot establish general accuracy or probability calibration. Existing labels may also contain ambiguities or errors.',
              'Zero-denominator precision/recall/F1 are reported as 0. Duplicate rows remain included and are identified in data-issues.csv.',
              'Model inputs contain only headlines, the fixed question, and category descriptions. No true labels are passed to the model.', '']
    return '\n'.join(lines)


def diagnostics():
    import torch
    print(f'Python: {sys.version.split()[0]}; PyTorch: {torch.__version__}; built CUDA: {torch.version.cuda}; CUDA accessible: {torch.cuda.is_available()}')
    try:
        p = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=15)
        print(p.stdout or p.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print('nvidia-smi unavailable: ' + type(exc).__name__)
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f'cuda:{i}: {torch.cuda.get_device_name(i)}; {free / 2**30:.2f} / {total / 2**30:.2f} GiB free')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Show GPU diagnostics without downloading models/data')
    parser.add_argument('--download-only', action='store_true', help='Download data and model files without loading the model')
    parser.add_argument('--limit', type=int, help='Score only the first N rows')
    parser.add_argument('--device', help='Override auto with cuda:N or cpu')
    args = parser.parse_args()
    settings = json.loads((ROOT / 'settings.json').read_text())
    if args.device: settings['device'] = args.device
    if args.limit is not None: settings['row_limit'] = args.limit
    if settings['row_limit'] is not None and settings['row_limit'] < 1:
        parser.error('row limit must be positive')
    labels = list(settings['categories'])
    if set(labels) != {'sports', 'politics', 'finance'}:
        raise ValueError('This pipeline expects sports, politics, finance')
    os.environ['HF_HOME'] = str(local_path(settings['model_cache']))
    os.environ['KAGGLEHUB_CACHE'] = str(local_path(settings['data_folder']) / 'kaggle-cache')
    if args.check:
        diagnostics()
        return 0
    path = obtain_dataset(settings)
    all_rows = read_csv(path)
    issues = audit_rows(all_rows, labels)
    print(f'Loaded {len(all_rows)} headlines; {len(issues)} rows have data issues.', flush=True)
    if args.download_only:
        checkpoint(settings)
        from huggingface_hub import snapshot_download
        print('Downloading/reusing the pinned Qwen base weights...', flush=True)
        snapshot_download(settings['base_model'], revision=settings['base_revision'],
                          allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt', '*.jinja'])
        print('Downloads complete. Run bash run.sh to score.', flush=True)
        return 0
    out = local_path(settings['output_folder']) / (datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    out.mkdir(parents=True)
    packages = {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'kagglehub')}
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'starting',
                'settings': settings, 'dataset_file': str(path), 'dataset_sha256': sha256(path),
                'source_rows': len(all_rows), 'python': sys.version, 'packages': packages,
                'official_code_commit': CODE_COMMIT, 'pipeline_sha256': sha256(__file__)}
    meta_path = out / 'run-metadata.json'
    def save_metadata(): meta_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    save_metadata()
    write_csv(out / 'data-issues.csv', issues, ['source_row', 'id', 'headline', 'true_label', 'issues'])
    try:
        # These inherited settings would change the expected inference identity.
        for key in ('JEV_LOAD_4BIT', 'JEV_LOAD_8BIT', 'JEV_DEVICE_MAP', 'JEV_MAX_MEMORY', 'JEV_TORCH_DTYPE'):
            if os.environ.get(key): raise RuntimeError(f'Unset {key} for this exact checkpoint evaluation.')
        device, hardware = choose_device(settings)
        metadata.update(device=device, hardware=hardware)
        save_metadata()
        ckpt, config = checkpoint(settings)
        from jev.serving import load_predictor
        print(f'Loading real Open-Jev-2B on {device}. First use downloads Qwen base weights...', flush=True)
        start_load = time.perf_counter()
        predictor = load_predictor(checkpoint=str(ckpt), device=device,
                                   max_length=settings['max_length'], batch_size=settings['candidate_batch_size'],
                                   prefix_cache=False, backend='torch')
        metadata.update(load_seconds=time.perf_counter() - start_load, checkpoint_config=config,
                        inference_provenance=predictor.provenance, temperature=predictor.temperature)
        save_metadata()
        rows = all_rows[:settings['row_limit']] if settings['row_limit'] else all_rows
        fields = ['source_row', 'id', 'headline', 'true_label'] + ['prob_' + l for l in labels] + ['predicted_label', 'confidence', 'correct', 'inference_seconds', 'status', 'error']
        results = []
        with (out / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as stream, (out / 'raw-responses.jsonl').open('w', encoding='utf-8') as raw:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore')
            writer.writeheader()
            for number, row in enumerate(rows, 1):
                result = {field: '' for field in fields}
                result.update(row, source_row=number)
                began = time.perf_counter()
                try:
                    if not row['headline'].strip(): raise ValueError('Missing headline')
                    request = category_request(row['headline'], settings)
                    # Check the full candidate prompts before inference; no truncation.
                    from jev.api import compile_request, candidate_prompts
                    prompts = [p for r in compile_request(request['state'], request['questions']) for p in candidate_prompts(r)]
                    tok = predictor.scorer.model.tokenizer
                    lengths = [len(ids) for ids in tok(prompts, add_special_tokens=True, truncation=False)['input_ids']]
                    if max(lengths) > settings['max_length']:
                        raise ValueError('Headline plus prompt exceeds token limit; input was not truncated')
                    response = predictor.predict(request)
                    probs, predicted = validate_probabilities(response['answers']['topic'], labels)
                    result.update({'prob_' + label: probs[label] for label in labels})
                    truth = row['true_label'].strip().lower()
                    result.update(predicted_label=predicted, confidence=max(probs.values()),
                                  correct=(predicted == truth) if truth in labels else '', status='ok')
                    raw.write(json.dumps({'source_row': number, 'id': row['id'], 'response': response}, allow_nan=False) + '\n')
                except Exception as exc:
                    result.update(status='error', error=f'{type(exc).__name__}: {exc}')
                result['inference_seconds'] = time.perf_counter() - began
                writer.writerow(result)
                stream.flush(); raw.flush()
                results.append(result)
                print(f"[{number}/{len(rows)}] {result['status']}: {result['predicted_label'] or result['error']}", flush=True)
                # Do not repeatedly run a device after a CUDA fault or memory exhaustion.
                if result['status'] == 'error' and ('CUDA' in result['error'] or 'out of memory' in result['error'].lower()):
                    raise RuntimeError('CUDA scoring failed. Partial scores are saved; existing services were not changed.')
        metrics = evaluate(results, labels)
        (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
        (out / 'report.md').write_text(report_text(metrics, issues, labels))
        write_csv(out / 'incorrect-predictions.csv', [r for r in results if r['correct'] is False], fields)
        metadata.update(status='completed' if metrics['failed_rows'] == 0 else 'completed_with_errors',
                        completed_utc=datetime.now(timezone.utc).isoformat(), metrics=metrics)
        save_metadata()
        accuracy = metrics['accuracy']
        print(f"\nScored {metrics['successfully_scored']}/{len(rows)}. Accuracy: {accuracy:.1%}" if accuracy is not None else '\nNo valid labeled predictions to evaluate.')
        print(f'Results: {out}', flush=True)
        return 0 if metrics['failed_rows'] == 0 else 1
    except Exception as exc:
        metadata.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_metadata()
        (out / 'error.txt').write_text(traceback.format_exc())
        print(f'\nStopped: {exc}\nRun details: {out}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Stopped: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
