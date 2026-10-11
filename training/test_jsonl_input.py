import json

import pytest
from torch.utils.data import DataLoader

from data_pipeline.manifest import load_manifest
from data_pipeline.select_sft_rows import main
from data_pipeline.test_sft_selection import row
from training.jsonl_input import load_jsonl_input


def test_selected_byte_offsets_workers_and_regeneration(tmp_path):
    source = tmp_path / 'sft_turns.jsonl'
    rows = [row(turn=i) for i in range(3)]
    rows[0]['prompt'][0]['content'] = '中文'
    rows[1]['metadata']['outcome']['official_success'] = False
    payloads = [json.dumps(r, ensure_ascii=False).encode() for r in rows]
    # Physical line numbers include blank lines; byte offsets include UTF-8 and CRLF.
    content = b'\n' + b'\r\n\n'.join(payloads)
    source.write_bytes(content)
    policy = tmp_path / 'policy.yaml'
    policy.write_text('official_success_only: true\n')
    manifest = tmp_path / 'train.json'
    args = ['--dataset', str(source), '--policy', str(policy), '--output', str(manifest)]
    main(args)
    selection = load_manifest(manifest)[1]['train']
    assert [r['source_line'] for r in selection.rows] == [2, 6]
    assert [r['offset'] for r in selection.rows] == [1, content.index(payloads[2])]
    dataset, estimates = load_jsonl_input(selection)
    assert estimates == [20, 20]
    assert dataset[[1, 0]]['row_index'] == [1, 0]
    batches = list(DataLoader(dataset, batch_size=1, num_workers=2))
    assert [json.loads(b['row_json'][0]) for b in batches] == [rows[0], rows[2]]
    assert source.read_bytes() == content
    assert not list(tmp_path.glob('*.arrow'))

    # A new selection intentionally replaces the old one, with no stale index cache.
    source.write_bytes(payloads[2] + b'\n' + payloads[0])
    main(args)
    fresh, _ = load_jsonl_input(load_manifest(manifest)[1]['train'])
    assert [json.loads(r['row_json']) for r in fresh] == [rows[2], rows[0]]
    source.write_bytes(b'')
    with pytest.raises(ValueError, match='Incomplete SFT row'):
        fresh[0]


@pytest.mark.parametrize('field,value', [('offset', -1), ('length', 0), ('source', 4),
                                        ('source_line', 0)])
def test_invalid_selection_positions(tmp_path, field, value):
    from data_pipeline.test_sft_selection import select

    manifest, _ = select(tmp_path, [row()], dict(official_success_only=True))
    data = json.loads(manifest.read_text())
    data['train'][0][field] = value
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_manifest(manifest)
