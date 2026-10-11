"""Own the W&B run lifetime; Trainer's native callback logs the metrics."""

from contextlib import contextmanager
import json
import os
from pathlib import Path


@contextmanager
def wandb_session(args, *, is_main_process):
    if args.report_to != "wandb" or not is_main_process:
        if is_main_process:
            print(f"[wandb] 不记录:report_to={args.report_to}", flush=True)
        yield
        return

    import wandb
    # 把实际生效的 mode / 地址 / 归属打进 master.log。第一次 smoke 一行 wandb 输出都没有,
    # 事后分不清是没 init 还是 init 成了 disabled;这一行以后就分得清。
    print("[wandb] init mode=%s base_url=%s entity=%s project=%s name=%s dir=%s" % (
        os.environ.get("WANDB_MODE") or "online", os.environ.get("WANDB_BASE_URL"),
        os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_PROJECT") or "physicalagent-sft",
        args.run_name or os.environ.get("WANDB_NAME"), os.environ.get("WANDB_DIR") or args.output),
        flush=True)

    output = Path(args.output)
    run = wandb.init(
        # 默认 online。training/README.md 一直写的就是 online,tools/aistudio_submit
        # 那批 job 也都是 online;这里原来写 offline,和两边都对不上(2026-09-16 改)。
        # 要离线就显式给 WANDB_MODE=offline,训完 `wandb sync <output>/wandb/offline-run-*`。
        mode=os.environ.get("WANDB_MODE") or "online",
        project=os.environ.get("WANDB_PROJECT") or "physicalagent-sft",
        name=args.run_name or os.environ.get("WANDB_NAME") or output.name,
        dir=os.environ.get("WANDB_DIR") or str(output),
        config={"sft": json.loads((output / "resolved_config.json").read_text())},
    )
    # WANDB_ENTITY / WANDB_BASE_URL / WANDB_API_KEY are read by the SDK.
    print(f"[wandb] run url={getattr(run, 'url', None)} id={run.id} mode={run.settings.mode}", flush=True)
    exit_code = 1
    try:
        yield
        exit_code = 0
    finally:
        # Also flush final evaluation before the optional os._exit workaround.
        run.finish(exit_code=exit_code)
