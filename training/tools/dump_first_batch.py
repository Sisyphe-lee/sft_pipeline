#!/usr/bin/env python3
"""合并门诊断:不改任何入口代码,monkeypatch FiniteSFTTrainer.compute_loss,把第一步喂给模型的
batch 各 key 与模型参数 checksum 存到 <dump>.pt,然后照常训练。两个入口各跑一次后用
--compare 逐 key 比较。

用法:
  python3 dump_first_batch.py --dump out.pt -- <train_sft.py 的全部参数>
  python3 dump_first_batch.py --compare a.pt b.pt
以 PYTHONPATH 里排在前面的 repo 决定 import 到哪个 training.train_sft。
"""
import sys


def _compare(a_path: str, b_path: str) -> int:
    import torch

    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    keys = sorted(set(a) | set(b))
    bad = 0
    for k in keys:
        if k not in a or k not in b:
            print(f"{k}: only in {'a' if k in a else 'b'}"); bad += 1; continue
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if x.shape != y.shape or x.dtype != y.dtype:
                print(f"{k}: shape/dtype differ {tuple(x.shape)}/{x.dtype} vs {tuple(y.shape)}/{y.dtype}"); bad += 1; continue
            if x.is_floating_point():
                d = (x.float() - y.float()).abs()
                eq = bool(torch.equal(x, y))
                print(f"{k}: shape={tuple(x.shape)} dtype={x.dtype} equal={eq} max_abs_diff={d.max().item():.3e} n_diff={(d > 0).sum().item()}")
            else:
                eq = bool(torch.equal(x, y))
                print(f"{k}: shape={tuple(x.shape)} dtype={x.dtype} equal={eq} n_diff={(x != y).sum().item()}")
            bad += (not eq)
        else:
            eq = x == y
            print(f"{k}: {x!r} vs {y!r} equal={eq}")
            bad += (not eq)
    print("BATCH_PARITY", "OK" if bad == 0 else f"MISMATCH keys={bad}")
    return 1 if bad else 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--compare":
        return _compare(sys.argv[2], sys.argv[3])
    assert sys.argv[1] == "--dump" and sys.argv[3] == "--", "用法见文件头"
    dump_path = sys.argv[2]
    rest = sys.argv[4:]
    import torch
    import training.train_sft as ts

    orig = ts.FiniteSFTTrainer.compute_loss
    state = {"done": False}

    def patched(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not state["done"]:
            state["done"] = True
            out = {}
            for k, v in inputs.items():
                out[f"input/{k}"] = v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
            out["num_items_in_batch"] = (num_items_in_batch.item() if isinstance(num_items_in_batch, torch.Tensor) else num_items_in_batch)
            out["model_accepts_loss_kwargs"] = bool(getattr(self, "model_accepts_loss_kwargs", None))
            out["model_training"] = bool(model.training)
            out["model_class"] = type(model).__name__
            out["forward_qualname"] = getattr(getattr(model, "forward", None), "__qualname__", "?")
            m = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                total = torch.zeros((), dtype=torch.float64, device="cuda")
                n_req = 0; n_all = 0
                for name, p in m.named_parameters():
                    total += p.detach().float().sum().to(torch.float64); n_all += 1; n_req += int(p.requires_grad)
                out["param_sum"] = total.item(); out["n_params"] = n_all; out["n_params_requires_grad"] = n_req
            cfg = getattr(m, "config", None)
            out["config_json_len"] = len(cfg.to_json_string()) if cfg is not None else -1
            out["gradient_checkpointing"] = bool(getattr(m, "is_gradient_checkpointing", False))
            torch.save(out, dump_path)
            print(f"[dump_first_batch] saved {dump_path}: keys={sorted(out)}", flush=True)
        return orig(self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

    ts.FiniteSFTTrainer.compute_loss = patched
    sys.argv = ["train_sft.py", *rest]
    ts.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
