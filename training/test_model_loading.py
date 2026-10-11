import json
from types import SimpleNamespace

import pytest

from training.model_loading import parameter_numel, validate_zero3_init


@pytest.mark.parametrize("config", [None, {}, {"zero_optimization": {"stage": 2}}])
def test_partitioned_init_rejects_non_zero3(config):
    with pytest.raises(ValueError, match="ZeRO stage 3"):
        validate_zero3_init(config)


def test_partitioned_init_accepts_file_and_resolved_config(tmp_path):
    config = {"zero_optimization": {"stage": 3}}
    path = tmp_path / "ds.json"
    path.write_text(json.dumps(config))
    validate_zero3_init(path)
    validate_zero3_init(str(path))
    validate_zero3_init(config)


def test_global_parameter_count_works_before_gather():
    assert parameter_numel(SimpleNamespace(ds_numel=1024, numel=lambda: 0)) == 1024
    assert parameter_numel(SimpleNamespace(numel=lambda: 1024)) == 1024
