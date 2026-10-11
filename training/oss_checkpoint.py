"""训练中每存一次 checkpoint 就直接传 OSS。

照 hatssl 的 `linetok/fsdp/__init__.py` 那套骨架:manager 可以是 None(没配就整个跳过)、
只 main process 传、传成功之后才动本地副本。三处按我们的栈改了:

1. **传的是目录不是单文件。** 那边是 FSDP 的 `torch.save(data, f)` 写一个 .pth,我们是 HF
   Trainer 存出来的 `checkpoint-<step>/` 目录,里面是 safetensors 分片加一堆 config,所以
   按文件遍历着传。
2. **跳过 ZeRO optimizer 分片。** 27B 一个 checkpoint 的 `global_step*/` 就有 300 GB,而且
   只有续训用得上;三个 epoch 全传是 900 GB。默认只传权重与 config,要续训另说。
3. **大文件走 resumable_upload。** wiki/guides/oss-operations.md 第 6 节记着 14 GB 以上
   用 `put_object_from_file` 可能 multipart abort 失败,而 27B 的分片是 47 GB。
   `oss2.resumable_upload` 自带分片与断点续传,part_size 给 100 MB(和 hatssl 的
   `create_oss_manager(chunk_size=100*1024*1024)` 一个量级)。

凭据从 ossutil 的 config 文件读,格式与 `~/.ossutilconfig` 相同,路径由 `oss_config` 指定。
**凭据只从文件读,不进日志、不进 run record、不落任何产物。**

没配 bucket、找不到 config、或者机器上没有 oss2,都只打一行说明然后什么都不做 —— 训练已经
在跑了,传不上去不该把 job 判成失败。
"""

from __future__ import annotations

import os
from pathlib import Path

# 这些文件只有续训用得上,权重与 config 之外的都不传。
SKIP_DIRS = ("global_step",)
SKIP_FILES = ("optimizer.pt", "scheduler.pt", "trainer_state.json.tmp")
SKIP_PREFIX = ("rng_state_",)

PART_SIZE = 100 * 1024 * 1024
MULTIPART_THRESHOLD = 200 * 1024 * 1024


def _read_ossutil_config(path):
    """从 ossutil config 取 endpoint / ak / sk。照 hatssl 的 lazy_oss 同一套解析。"""
    values = {}
    for line in Path(path).read_text().splitlines():
        for key in ("endpoint", "accessKeyID", "accessKeySecret"):
            if line.startswith(key + "="):
                values[key] = line.split("=", 1)[1].strip()
    missing = [k for k in ("endpoint", "accessKeyID", "accessKeySecret") if not values.get(k)]
    if missing:
        raise ValueError(f"ossutil config 缺这几项:{missing}")
    return values


def make_bucket(bucket_name, config_path):
    """返回 oss2.Bucket;缺任何前提就返回 None,并说明缺的是什么。"""
    if not bucket_name:
        return None
    if not config_path or not Path(config_path).is_file():
        print(f"[oss] 没有 config({config_path}),这次不传 OSS", flush=True)
        return None
    try:
        import oss2
    except ImportError:
        print("[oss] 这台机器没有 oss2,这次不传 OSS", flush=True)
        return None
    try:
        conf = _read_ossutil_config(config_path)
    except Exception as exc:                       # 不要把异常里的内容原样带出去
        print(f"[oss] config 读不出来({type(exc).__name__}),这次不传 OSS", flush=True)
        return None
    auth = oss2.Auth(conf["accessKeyID"], conf["accessKeySecret"])
    return oss2.Bucket(auth, conf["endpoint"], bucket_name)


def should_upload(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part.startswith(SKIP_DIRS) for part in rel.parts[:-1]):
        return False
    return rel.name not in SKIP_FILES and not rel.name.startswith(SKIP_PREFIX)


def upload_directory(bucket, local_dir, oss_prefix, *, delete_local=False):
    """把一个 checkpoint 目录传上去。返回 (传了几个, 跳过几个, 失败几个)。"""
    import oss2

    local_dir = Path(local_dir)
    sent = skipped = failed = 0
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        if not should_upload(path, local_dir):
            skipped += 1
            continue
        key = f"{oss_prefix.rstrip('/')}/{path.relative_to(local_dir).as_posix()}"
        try:
            oss2.resumable_upload(
                bucket, key, str(path),
                multipart_threshold=MULTIPART_THRESHOLD,
                part_size=PART_SIZE,
                num_threads=8,
            )
            sent += 1
        except Exception as exc:
            failed += 1
            print(f"[oss] 传失败 {key}: {type(exc).__name__}", flush=True)
    if failed == 0 and delete_local:
        # 与 hatssl 同一条规矩:确认全部传上去了才动本地副本。
        for path in sorted(local_dir.rglob("*")):
            if path.is_file() and should_upload(path, local_dir):
                path.unlink()
    return sent, skipped, failed


def make_callback(args):
    """按 recipe 里的 oss_bucket / oss_root 决定挂不挂这个 callback。没配就返回 None。"""
    bucket_name = getattr(args, "oss_bucket", None)
    if not bucket_name:
        return None
    config_path = getattr(args, "oss_config", None) or os.path.expanduser("~/.ossutilconfig")
    bucket = make_bucket(bucket_name, config_path)
    if bucket is None:
        return None
    root = getattr(args, "oss_root", None) or Path(args.output).name
    return OssCheckpointCallback(
        bucket, root, delete_local=bool(getattr(args, "oss_delete_local", False)))


class OssCheckpointCallback:
    """HF Trainer 的 callback:每存完一个 checkpoint 就把它传上去。

    只在 world process zero 上做。传失败只打日志不抛 —— 训练本身已经跑到这一步了,
    不该因为上传把 job 判成失败。
    """

    def __init__(self, bucket, oss_root, *, delete_local=False):
        self.bucket = bucket
        self.oss_root = oss_root.rstrip("/")
        self.delete_local = delete_local

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt.is_dir():
            print(f"[oss] step {state.global_step} 的 checkpoint 目录不在,跳过", flush=True)
            return control
        prefix = f"{self.oss_root}/{ckpt.name}"
        print(f"[oss] 传 {ckpt.name} → oss://{self.bucket.bucket_name}/{prefix}", flush=True)
        sent, skipped, failed = upload_directory(
            self.bucket, ckpt, prefix, delete_local=self.delete_local)
        print(f"[oss] {ckpt.name}: 传了 {sent} 个,跳过 {skipped} 个(optimizer 分片与 rng),"
              f"失败 {failed} 个", flush=True)
        return control

    # Trainer 的 callback 协议里其余钩子都不需要,给出空实现即可。
    def __getattr__(self, name):
        if name.startswith("on_"):
            return lambda *a, **k: k.get("control")
        raise AttributeError(name)
