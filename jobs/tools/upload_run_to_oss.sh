#!/usr/bin/env bash
# 把一次训练的产出传到 OSS,目录结构与 jiangyong/output/harness/ 下那几条线并列:
#
#   <前缀>/<run 名>/
#   ├── logs/                      master.log、worker-N.log、env_*.txt
#   ├── model-0000N-of-0000M.safetensors
#   ├── model.safetensors.index.json
#   ├── config.json / tokenizer.json / chat_template.jinja / …
#   └── trainer_state.json / all_results.json / …
#
# 只传输出目录的**顶层文件**。checkpoint-*/ 子目录不传 —— 每个里面有 300 GB 的 ZeRO
# optimizer 分片,对 serve 与 eval 没用,三个 epoch 传上去就是 900 GB。要续训才需要它们,
# 那种情况单独传。
#
# 用 ossutil64 而不是 oss_manager 的 Python SDK:wiki/guides/oss-operations.md 第 6 节记着
# 14 GB 以上的文件走 put_object_from_file 可能 multipart abort 失败,而 27B 的权重分片是
# 47 GB。ossutil64 自带分片上传与断点续传。
#
# 也不要用 --exclude 躲开 checkpoint-*:ossutil 的 --include/--exclude 不接受带 "/" 的
# 模式,给了会直接报 "does not support format containing dir info" 并整条退出 ——
# 2026-09-16 踩过,51 GB 一个字节没传却看不出错。这里改成 find 顶层逐个传。
#
# 用法:
#   bash jobs/tools/upload_run_to_oss.sh <输出目录> <日志目录> <run 名> [OSS 前缀]
#
# 默认前缀 oss://antsys-robbys3-prod/lingcong/output/harness
# 凭据从 OSS_CONFIG 指定的 ossutil config 读,默认 /root/.yubin-ossutil-configs/antsys-robbys3-prod-internal
#
# 缺 ossutil64 或缺 config 时**跳过并返回 0** —— 训练已经跑完了,上传不该把 job 判成失败。
set -uo pipefail

die()  { echo "ABORT: $*" >&2 ; exit 2 ; }
note() { echo "upload: $*" ; }

OUT=${1:?用法:upload_run_to_oss.sh <输出目录> <日志目录> <run 名> [OSS 前缀]}
LOGDIR=${2:?}
RUN=${3:?}
PREFIX=${4:-oss://antsys-robbys3-prod/lingcong/output/harness}
CONFIG=${OSS_CONFIG:-/root/.yubin-ossutil-configs/antsys-robbys3-prod-internal}
DEST=$PREFIX/$RUN

[ -d "$OUT" ] || die "输出目录不存在:$OUT"

command -v ossutil64 >/dev/null 2>&1 || {
  note "这台机器没有 ossutil64,跳过上传。产出留在 $OUT" ; exit 0 ; }
[ -f "$CONFIG" ] || {
  note "没有 OSS config($CONFIG),跳过上传。产出留在 $OUT"
  note "  凭据只能放容器本地,不要放 CPFS —— 那是全员可读的。"
  exit 0 ; }

note "目标 $DEST"

# --- 顶层文件逐个传 -----------------------------------------------------------
sent=0
while read -r f; do
  name=$(basename "$f")
  if ossutil64 cp "$f" "$DEST/$name" -f -u -j 16 --parallel 8 -c "$CONFIG" >/dev/null 2>&1; then
    sent=$((sent + 1))
    note "  ok  $name"
  else
    note "  FAIL $name"
  fi
done < <(find "$OUT" -maxdepth 1 -type f | sort)
note "顶层文件传了 $sent 个"

# --- 日志整个目录 -------------------------------------------------------------
if [ -d "$LOGDIR" ]; then
  ossutil64 cp -r "$LOGDIR/" "$DEST/logs/" -f -u -j 16 --parallel 8 -c "$CONFIG" >/dev/null 2>&1 \
    && note "logs 传完" || note "logs FAIL"
else
  note "日志目录不存在,跳过:$LOGDIR"
fi

# --- 传完对账:源顶层 vs OSS 上的对象,缺一个就报 -----------------------------
# wiki/guides/oss-operations.md 第 7 节:传完必做 diff。2026-09-16 那次就是只看进程还活着
# 就以为传上去了,实际一个字节没走。
TMP=$(mktemp -d) ; trap 'rm -rf "$TMP"' EXIT
find "$OUT" -maxdepth 1 -type f -printf '%f\n' | sort > "$TMP/src"
ossutil64 ls "$DEST/" -c "$CONFIG" 2>/dev/null \
  | grep -oE "$DEST/[^/]+$" | sed "s|$DEST/||" | sort > "$TMP/dst"
missing=$(comm -23 "$TMP/src" "$TMP/dst")
if [ -n "$missing" ]; then
  echo "ABORT: 这些顶层文件没传上去:" >&2
  echo "$missing" | sed 's/^/       /' >&2
  exit 2
fi
note "对账通过:源顶层 $(wc -l < "$TMP/src" | tr -d ' ') 个文件在 OSS 上都有"
note "路径 $DEST"
