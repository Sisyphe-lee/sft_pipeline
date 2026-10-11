"""Portable data roots, split isolation and explicit recipe overrides."""

import json
from pathlib import Path
import shutil

import pytest
from PIL import Image
import torch

from data_pipeline.manifest import SelectedRows, check_split_groups, load_manifest
from data_pipeline.select_sft_rows import main as select_main
from data_pipeline.test_sft_selection import row
from training.history_truncation import prepare_length_policy
from training.jsonl_input import load_jsonl_input
from training.recipe import parse_recipe, record_run, validate_resume_run
from training.test_history_truncation import make_processor
from training.train_sft import build_parser
from training.vlm_collator import VlmCollator


def test_multiple_files_relocation_and_image_roots(tmp_path):
    root = tmp_path / 'original'
    for i, color in enumerate(['red', 'blue']):
        directory = root / str(i)
        directory.mkdir(parents=True)
        Image.new('RGB', (64, 64), color).save(directory / 'image.png')
        r = row(seed=i, episode=str(i))
        r['prompt'][0]['content'] = [dict(type='image'), dict(type='text', text='current')]
        r['images'] = ['image.png']
        (directory / 'train.jsonl').write_text(json.dumps(r) + '\n')
    manifest = tmp_path / 'data.json'
    policy = tmp_path / 'policy.yaml'
    policy.write_text('official_success_only: true\n')
    select_main(['--dataset', str(root / '0/train.jsonl'), str(root / '1/train.jsonl'),
                 '--policy', str(policy), '--output', str(manifest)])
    moved = tmp_path / 'moved'
    shutil.move(root, moved)
    _, paths = load_manifest(manifest, moved)
    dataset, _ = load_jsonl_input(paths['train'])
    assert [r['row_index'] for r in dataset] == [0, 1]
    assert [r['dataset_root'] for r in dataset] == [str(moved / str(i)) for i in range(2)]
    processor = make_processor()
    selected, cuts, _ = prepare_length_policy(dataset, processor, moved, 1000)
    collator = VlmCollator(processor, moved, 1000, history_cuts=cuts)
    a, b = [collator([record]) for record in selected]
    assert torch.equal(a['labels'], b['labels'])
    assert not torch.equal(a['pixel_values'], b['pixel_values'])
    reordered = dataset.select([1, 0])
    assert [r['row_index'] for r in reordered] == [1, 0]
    validation, _ = load_jsonl_input(SelectedRows(paths['train'].sources, paths['train'].rows[1:]),
                                     row_index_offset=len(dataset))
    assert validation[0]['row_index'] == 2
    with pytest.raises(ValueError, match='share 1'):
        check_split_groups(dataset, validation)
    check_split_groups(dataset.select([0]), validation)


@pytest.mark.parametrize('base, train', [('/absolute', ['a.jsonl']), ('.', ['../a.jsonl']),
                                        ('.', ['/absolute.jsonl']), ('.', ['a.jsonl', 'a.jsonl'])])
def test_manifest_rejects_nonportable_and_duplicate_paths(tmp_path, base, train):
    (tmp_path / 'a.jsonl').write_text('{}\n')
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(version=2, base_root=base, sources=train, train=[])))
    with pytest.raises(ValueError):
        load_manifest(path)


def arguments(tmp_path, config='learning_rate: 2.0e-5\ntruncate_history: false\n'):
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(config)
    return ['--recipe', str(recipe), '--model', 'explicit-model',
            '--data-manifest', str(tmp_path / 'manifest.json'), '--output', str(tmp_path / 'run')]


def test_recipe_applies_types_relative_paths_and_cli_overrides(tmp_path):
    argv = arguments(tmp_path, 'learning_rate: 2.0e-5\ntruncate_history: false\nlora: true\ndeepspeed: null\n')
    args = parse_recipe(build_parser(), argv + ['--learning-rate', '3e-5', '--truncate-history', '--no-lora'])
    assert args.learning_rate == 3e-5 and args.truncate_history and not args.lora
    assert args.deepspeed is None
    args = parse_recipe(build_parser(), arguments(tmp_path, 'deepspeed: ../zero.json\n'))
    assert args.deepspeed == str((tmp_path / '../zero.json').resolve())
    default = Path(__file__).resolve().parents[1] / 'configs/recipes/full_sft.yaml'
    argv = arguments(tmp_path)
    argv[1] = str(default)
    args = parse_recipe(build_parser(), argv)
    assert args.truncate_history and not args.validation and args.sampling_strategy == 'turn_uniform'
    assert Path(args.deepspeed).is_file()


def test_recipe_can_save_each_epoch(tmp_path):
    args = parse_recipe(build_parser(), arguments(
        tmp_path, 'num_train_epochs: 10.0\nmax_steps: -1\nsave_strategy: epoch\n'))
    assert args.num_train_epochs == 10.0
    assert args.max_steps == -1
    assert args.save_strategy == 'epoch'


@pytest.mark.parametrize('config', ['model: implicit-4B\n', 'dataset: data.jsonl\n',
                                    'validation: "false"\n', 'unknown: 1\n', 'sampling_strategy: mystery\n'])
def test_recipe_rejects_unknown_runtime_keys_and_wrong_types(tmp_path, config):
    with pytest.raises(SystemExit):
        parse_recipe(build_parser(), arguments(tmp_path, config))


def test_model_is_required_and_legacy_loss_escape_is_removed(tmp_path):
    argv = arguments(tmp_path)
    with pytest.raises(SystemExit):
        parse_recipe(build_parser(), argv[:2] + argv[4:])
    with pytest.raises(SystemExit):
        parse_recipe(build_parser(), argv + ['--all-tokens-loss'])


def test_resume_checks_resolved_data_and_recipe(tmp_path):
    args = parse_recipe(build_parser(), arguments(tmp_path))
    evidence = dict(manifest_sha256='selection-hash', loss='turn_mean')
    record_run(args, evidence)
    with pytest.raises(ValueError, match='already contains'):
        record_run(args, evidence)
    checkpoint = Path(args.output) / 'checkpoint-1'
    checkpoint.mkdir()
    (checkpoint / 'trainer_state.json').write_text('{}')
    args.resume_from_checkpoint = str(checkpoint)
    validate_resume_run(args, evidence)
    record_run(args, evidence)
    with pytest.raises(ValueError, match='differs'):
        validate_resume_run(args, dict(evidence, manifest_sha256='changed-selection'))
    with pytest.raises(ValueError, match='differs'):
        record_run(args, dict(evidence, manifest_sha256='changed-selection'))


def test_universal_resume_allows_only_world_size_change(tmp_path):
    args = parse_recipe(build_parser(), arguments(tmp_path))
    evidence = dict(manifest_sha256='selection-hash', loss='turn_mean', world_size=8)
    record_run(args, evidence)
    checkpoint = Path(args.output) / 'checkpoint-1'
    checkpoint.mkdir()
    (checkpoint / 'trainer_state.json').write_text('{}')
    args.resume_from_checkpoint = str(checkpoint)

    changed_world_size = dict(evidence, world_size=16)
    with pytest.raises(ValueError, match='differs'):
        validate_resume_run(args, changed_world_size)

    args.resume_with_universal_checkpoint = True
    validate_resume_run(args, changed_world_size)
    record_run(args, changed_world_size)
    with pytest.raises(ValueError, match='differs'):
        validate_resume_run(args, dict(changed_world_size, manifest_sha256='changed-selection'))
    with pytest.raises(ValueError, match='differs'):
        record_run(args, dict(changed_world_size, manifest_sha256='changed-selection'))
