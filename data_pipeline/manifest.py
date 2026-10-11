"""Portable data locations and identities shared by selection and training."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SelectedRows:
    sources: list[Path]
    rows: list[dict]

    def __len__(self):
        return len(self.rows)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def row_keys(row):
    meta = row.get('metadata', {})
    task = meta.get('task', {})
    task_id = str(task.get('id') or '')
    trajectory_id = str(meta.get('trajectory_id') or '')
    seed = task.get('seed')
    group = json.dumps([task_id, seed], separators=(',', ':')) if task_id and seed is not None else ''
    if not task_id or not trajectory_id or not group or 'turn' not in meta:
        raise ValueError('Rows must record task.id, task.seed, trajectory_id and turn; re-export the data')
    return dict(task_id=task_id, trajectory_id=trajectory_id, split_group=group)


def load_manifest(path, data_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (not isinstance(manifest, dict) or type(manifest.get('version')) is not int
            or manifest.get('version') != 2):
        raise ValueError('Data manifest must have version: 2; regenerate the selection manifest')
    if manifest.keys() - {'version', 'base_root', 'train', 'validation', 'sources', 'selection',
                          'source_digests'}:
        raise ValueError('Unknown data manifest keys')
    base = manifest.get('base_root')
    if not isinstance(base, str) or Path(base).is_absolute():
        raise ValueError('base_root must be relative to the manifest; use --data-root for a mount path')
    root = Path(data_root).resolve() if data_root is not None else (path.parent / base).resolve()
    refs = manifest.get('sources')
    if not isinstance(refs, list) or not refs:
        raise ValueError('Manifest requires source files')
    sources = []
    for ref in refs:
        if not isinstance(ref, str) or Path(ref).is_absolute():
            raise ValueError('Source paths must be relative')
        resolved = (root / ref).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f'Data path must be inside base_root: {ref}')
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        sources.append(resolved)
    if len(set(sources)) != len(sources):
        raise ValueError('Duplicate source files')
    splits, seen = {}, set()
    for split in ('train', 'validation'):
        rows = manifest.get(split, [])
        if not isinstance(rows, list):
            raise ValueError(f'{split} must be a list of selected rows')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('Invalid selected row')
            for key, minimum in [('source', 0), ('source_line', 1), ('offset', 0),
                                 ('length', 1), ('estimated_tokens', 1)]:
                if type(row.get(key)) is not int or row[key] < minimum:
                    raise ValueError(f'Invalid selected row {key}')
            if row['source'] >= len(sources):
                raise ValueError('Selected row source is out of range')
            if any(not isinstance(row.get(key), str) or not row[key]
                   for key in ('task_id', 'trajectory_id', 'split_group')):
                raise ValueError('Selected row requires task and episode identities')
            identity = (row['source'], row['offset'])
            if identity in seen:
                raise ValueError('Duplicate selected row across train/validation')
            seen.add(identity)
        splits[split] = SelectedRows(sources, rows)
    if not splits['train']:
        raise ValueError('Manifest requires training rows')
    return root, splits


def check_split_groups(train, validation):
    train_groups = set(train.with_format(None)['split_group'])
    val_groups = set(validation.with_format(None)['split_group'])
    if '' in train_groups | val_groups:
        raise ValueError('Validation requires task and scene seed metadata in every row')
    overlap = train_groups & val_groups
    if overlap:
        raise ValueError(f'Train/validation share {len(overlap)} task/seed groups: {sorted(overlap)[:5]}')
