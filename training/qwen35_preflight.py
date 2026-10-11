"""Fail-closed compatibility checks for Qwen3.5 multimodal training."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any

import transformers
import datasets
import trl
from packaging.version import InvalidVersion, Version
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_attention_backend(attn_implementation: str) -> None:
    """Reject the node image's incompatible FlashAttention monkeypatch."""

    if attn_implementation != "flash_attention_2":
        return
    import flash_attn

    patched_names = (
        "_old_flash_attn_func",
        "_old_flash_attn_varlen_func",
        "_old_flash_attn_varlen_qkvpacked_func",
    )
    if any(hasattr(flash_attn, name) for name in patched_names):
        raise RuntimeError(
            "FlashAttention preflight failed: zhenjin_utils replaced the native "
            "FlashAttention 2 API with an incompatible wrapper. Relaunch with "
            "ENABLE_MY_FLASH_ATTN_HOOK=0 before Python starts."
        )
    missing_fast_path = []
    for name in (
        "fla.modules",
        "fla.ops.gated_delta_rule",
        "causal_conv1d",
        "tilelang",
    ):
        try:
            available = importlib.util.find_spec(name) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            missing_fast_path.append(name)
    if missing_fast_path:
        raise RuntimeError(
            "Qwen3.5 hybrid attention fast path is missing: "
            + ", ".join(missing_fast_path)
            + ". Install the pinned training environment; the torch fallback "
            "is too slow for formal long-context SFT."
        )
    try:
        import tilelang  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "TileLang is present but its runtime dependencies are incomplete. "
            "Install the pinned apache-tvm-ffi, torch-c-dlpack-ext, and "
            "z3-solver packages before Qwen3.5 training."
        ) from error


def verify_training_stack() -> None:
    """Enforce the training entry point's minimum dependency versions."""

    minimums = {
        "transformers": (transformers.__version__, "5.14.1"),
        "trl": (trl.__version__, "1.9.1"),
        # 现役训练 overlay(qwendeps/qwen35-train)是 datasets 4.8.5,两轮 27B 训练都在它上面;
        # lcy-dev 把地板提到 5.0.1 是照他自己环境的 pin,合并后以实际验证过的版本为准。
        "datasets": (datasets.__version__, "4.8.5"),
    }
    incompatible = [
        f"{name}={installed} (<{required})"
        for name, (installed, required) in minimums.items()
        if Version(installed) < Version(required)
    ]
    if incompatible:
        raise RuntimeError(
            "Qwen3.5 training stack violates the minimum training versions: "
            + ", ".join(incompatible)
            + ". Refusing to reuse the node system environment."
        )


def load_verified_processor(model: str):
    """Load the checkpoint processor only after enforcing its declared contract."""

    model_path = Path(model)
    config = _read_json(model_path / "config.json")
    preprocessor = _read_json(model_path / "preprocessor_config.json")
    required_version = config.get("transformers_version")
    if required_version:
        try:
            installed = Version(transformers.__version__)
            required = Version(str(required_version))
        except InvalidVersion as error:
            raise RuntimeError(
                "Cannot compare the checkpoint and installed Transformers versions: "
                f"checkpoint={required_version!r}, installed={transformers.__version__!r}"
            ) from error
        if installed < required:
            raise RuntimeError(
                "Qwen3.5 multimodal preflight failed before dataset validation/training: "
                f"checkpoint requires transformers>={required}, but node has {installed}. "
                "Use a dedicated compatible training environment; do not let "
                "AutoProcessor fall back to a tokenizer."
            )

    try:
        resolved_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        AutoModelForImageTextToText._model_mapping[type(resolved_config)]
    except (KeyError, ValueError) as error:
        raise RuntimeError(
            "Qwen3.5 multimodal preflight failed: the installed Transformers "
            f"{transformers.__version__} cannot resolve model_type="
            f"{config.get('model_type')!r} through AutoModelForImageTextToText. "
            "A numerically newer stable release may still lack an architecture "
            "that existed only in the checkpoint's development build."
        ) from error

    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    expects_vision = bool(config.get("vision_config")) or bool(
        preprocessor.get("image_processor_type")
    )
    if expects_vision and (
        not hasattr(processor, "image_processor")
        or not hasattr(processor, "tokenizer")
    ):
        raise RuntimeError(
            "Qwen3.5 multimodal preflight failed: AutoProcessor resolved to "
            f"{type(processor).__name__}, expected "
            f"{preprocessor.get('processor_class', 'a multimodal processor')} with "
            "both image_processor and tokenizer. Refusing validation/training."
        )
    return processor
