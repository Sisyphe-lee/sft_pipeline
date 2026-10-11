"""LoRA 的纯逻辑部分,不 import torch / trl / datasets。

单独成一个模块是为了可测:``train_sft.py`` 顶层就要拉起 torch、trl、
datasets,在没装训练依赖的环境里连 import 都过不去,于是这些本来是纯字符串
处理的规则就没法写单测。规则出错的后果又很隐蔽(LoRA 悄悄训了 vision 塔,
不报错、只是多训一批参数),正是最需要测试盯住的那类。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

#: 判定"这个模块属于 vision 塔"的名字片段。与 ``--freeze-vision`` 用的是
#: 同一组,两处必须一致 —— 否则会出现"base 冻住了但 adapter 在训"的错配。
VISION_MODULE_TOKENS = ("vision", "visual", "aligner")


def lora_vision_exclusions(
    module_names: Iterable[str], targets: Sequence[str]
) -> list[str]:
    """挑出"名字命中 target、但属于 vision 塔"的模块全名,交给 peft 排除。

    为什么需要:``target_modules`` 是按后缀匹配的,而 ``q_proj`` 在语言侧和
    vision 塔里都叫 ``q_proj``。只给后缀的话 LoRA 会注进 vision 塔,于是 base
    权重虽然被 ``--freeze-vision`` 冻住了,adapter 却在训它 —— 这与冻结的意图
    正好相反,而且不会报错,只会安静地多训一部分参数。
    """

    target_set = set(targets)
    return sorted(
        {
            name
            for name in module_names
            if any(token in name.lower() for token in VISION_MODULE_TOKENS)
            and name.rsplit(".", 1)[-1] in target_set
        }
    )
