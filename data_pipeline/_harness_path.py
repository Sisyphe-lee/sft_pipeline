"""把 harness pinned checkout 挂进 sys.path,给需要投影链的数据脚本用。

为什么不 vendor:EP-D 投影的入口 `prompt_epd` 本身很轻,但
`sft_contract_projection`(投影器要用)会一路拉到 `context_replay` →
`tool_contracts` → `trajectory` → `tools` → `perception*`,二十几个模块、
带 numpy 和感知几何。那是 harness 的核心,不是 SFT 侧该复制并各自演进的
东西——复制一份就等于开了第二条会漂的投影实现,而 train/serve 成对是这条
线的命根子。

所以规矩是:**投影必须在 pinned checkout 里跑**,由 `HARNESS_CHECKOUT`
显式指定。fail-closed:没设或指错就直接退出,不猜、不回落到"当前目录往上
爬两层"——猜错的后果是拿另一代 harness 的投影生成训练数据,而这种错不会
报错,只会在几天后的 eval 里表现为莫名其妙的 SR 塌方。

用法:
    export HARNESS_CHECKOUT=/personal/ppc_overfit_20260810/repo_eval_rt1
    python3 data_pipeline/project_sft_epd.py ...

各 lineage 的 checkout 见 docs/EVAL_VERSIONS.md 与根 README 的 pinned
commit 清单(RT1 / E3 / liftpot)。
"""

import os
import sys
from pathlib import Path

ENV = "HARNESS_CHECKOUT"


def bootstrap() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        sys.exit(
            f"{ENV} 未设置。投影链必须在 harness 的 pinned checkout 里跑,\n"
            "  例:export HARNESS_CHECKOUT=/personal/ppc_overfit_20260810/repo_eval_rt1\n"
            "  见 docs/EVAL_VERSIONS.md 与根 README 的 pinned commit 清单。")
    root = Path(raw).expanduser().resolve()
    if not (root / "physical_agent" / "prompt_epd.py").is_file():
        sys.exit(f"{ENV}={root} 里没有 physical_agent/prompt_epd.py,不是 harness checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
