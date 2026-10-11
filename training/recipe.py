"""Feed YAML values through the existing argparse validation and CLI overrides."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def parse_recipe(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--recipe', type=Path)
    recipe_path = bootstrap.parse_known_args(argv)[0].recipe
    if recipe_path is None or '--help' in argv or '-h' in argv:
        return parser.parse_args(argv)
    config = yaml.safe_load(recipe_path.read_text())
    if not isinstance(config, dict):
        parser.error('Recipe must be a YAML mapping')
    actions = {action.dest: action for action in parser._actions}
    runtime_keys = {
        'recipe', 'data_manifest', 'data_root', 'model', 'output',
        'resume_from_checkpoint', 'resume_with_universal_checkpoint', 'help',
    }
    tokens = []
    for key, value in config.items():
        if key not in actions or key in runtime_keys:
            parser.error(f'Unknown or runtime-only recipe key: {key}')
        action = actions[key]
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                parser.error(f'{key} must be a YAML boolean')
            tokens.append(action.option_strings[0 if value else 1])
        elif value is None:
            if key == 'deepspeed':
                tokens.extend([action.option_strings[0], 'none'])
            elif action.default is not None:
                parser.error(f'{key} cannot be null')
        else:
            if isinstance(value, (bool, list, dict)):
                parser.error(f'{key} must be a scalar argument value')
            if key in {'deepspeed', 'fla_config_dir'}:
                value = str((recipe_path.resolve().parent / str(value)).resolve())
            tokens.extend([action.option_strings[0], str(value)])
    return parser.parse_args(tokens + argv)


def build_run_record(args, evidence):
    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    values.pop('resume_from_checkpoint')
    values.pop('resume_with_universal_checkpoint', None)
    return dict(arguments=values, **evidence)


def _resume_record_matches(recorded, expected, allow_world_size_change, exact=False):
    """Compare resume evidence, optionally ignoring only the launcher world size."""
    if not isinstance(recorded, dict):
        return False
    keys = set(recorded) | set(expected) if exact else set(expected)
    for key in keys:
        if allow_world_size_change and key == 'world_size':
            continue
        if recorded.get(key) != expected.get(key):
            return False
    return True


def validate_resume_run(args, evidence):
    """Fail on static resume differences before expensive length preflight."""
    if not args.resume_from_checkpoint:
        return
    checkpoint = Path(args.resume_from_checkpoint)
    if checkpoint.resolve().parent != Path(args.output).resolve():
        raise ValueError('Resume checkpoint must belong to the output run directory')
    if not (checkpoint / 'trainer_state.json').is_file():
        raise ValueError('Resume requires a Trainer checkpoint containing trainer_state.json')
    path = Path(args.output) / 'resolved_config.json'
    if not path.is_file():
        raise ValueError('Resume data/config differs from resolved_config.json; use the original run configuration')
    recorded = json.loads(path.read_text())
    expected = build_run_record(args, evidence)
    if not _resume_record_matches(
        recorded,
        expected,
        args.resume_with_universal_checkpoint,
    ):
        raise ValueError('Resume data/config differs from resolved_config.json; use the original run configuration')


def record_run(args, evidence):
    """Record the resolved run once; resume only the same data and recipe."""
    if int(os.environ.get('RANK', '0')) != 0:
        return
    record = build_run_record(args, evidence)
    output = Path(args.output)
    path = output / 'resolved_config.json'
    if args.resume_from_checkpoint:
        if Path(args.resume_from_checkpoint).resolve().parent != output.resolve():
            raise ValueError('Resume checkpoint must belong to the output run directory')
        if not Path(args.resume_from_checkpoint, 'trainer_state.json').is_file():
            raise ValueError('Resume requires a Trainer checkpoint containing trainer_state.json')
        if not path.is_file() or not _resume_record_matches(
            json.loads(path.read_text()),
            record,
            args.resume_with_universal_checkpoint,
            exact=True,
        ):
            raise ValueError('Resume data/config differs from resolved_config.json; use the original run configuration')
    elif path.is_file():
        raise ValueError('Output already contains a run; use --resume-from-checkpoint or a new output')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty for a new training run')
    if not args.resume_from_checkpoint:
        output.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)
