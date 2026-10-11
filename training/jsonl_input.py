"""Read selected decisions directly from their original SFT JSONL files."""

from contextlib import ExitStack
from functools import partial

from datasets import Dataset


def _read_batch(batch, paths):
    rows, roots = [], []
    with ExitStack() as stack:
        handles = {source: stack.enter_context(paths[source].open('rb'))
                   for source in set(batch['source'])}
        for source, offset, length in zip(batch['source'], batch['offset'], batch['length']):
            handle = handles[source]
            handle.seek(offset)
            raw = handle.read(length)
            if len(raw) != length:
                raise ValueError(f'Incomplete SFT row: {paths[source]} at byte {offset}')
            rows.append(raw.decode('utf-8'))
            roots.append(str(paths[source].parent))
    return dict(row_json=rows, row_index=batch['row_index'], dataset_root=roots)


def load_jsonl_input(selection, row_index_offset=0):
    """Build a small in-memory index; never copy or hash source sample bodies."""
    if not selection.rows:
        raise ValueError('SFT selection has no samples')
    index = Dataset.from_list([
        dict(row, row_index=i + row_index_offset) for i, row in enumerate(selection.rows)
    ])
    estimates = sorted(index['estimated_tokens'])
    return index.with_transform(partial(_read_batch, paths=selection.sources)), estimates
