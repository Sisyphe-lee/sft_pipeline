#!/usr/bin/env bash
# 合并门:同一 40 行(20260911_20646rows_h40e76994 的前 40 行)、同参数、deterministic,
# 老 CLI(main 的 train_sft.py,$P/repo_sft)与新 CLI(merge-lcy-dev 的 train_sft.py,
# $P/repo_sft_merge,recipe + manifest)各跑 3 步,逐步比 loss。s3 单卡跑,GPU 可与 eval 共用
#(9B 约 40 GB)。用法:GPU=7 bash training/tools/parity_new_cli.sh
set -uo pipefail
# 不用 -e:镜像在训练结束后的 Python 析构阶段会 SIGSEGV(rc 139),loss 已经打完;按 loss 行数判断成败。
P=${P:-/personal/ppc_overfit_20260810}
OLD=${OLD_REPO:-$P/repo_sft}          # main:老 CLI
NEW=${NEW_REPO:-$P/repo_sft_merge}    # merge-lcy-dev:新 CLI
BASE=${BASE:-$P/base_models/Qwen3.5-9B}
SRC=${SRC:-/personal/lcy_data/lcy_full_20260911/sft/lcy_full_20260911/train_le196608.jsonl}
ROWS=${ROWS:-40}
ONLY=${ONLY:-both}      # both | old | new
TAG=${TAG:-}            # 日志后缀,老 CLI 连跑两次比较用
RECIPE_NEW=${RECIPE_NEW:-configs/recipes/parity_9b_single_gpu.yaml}
GPU=${GPU:-auto}
if [ "$GPU" = auto ]; then  # 挑当前空闲显存最多的卡(s3 上 8 张卡都有 eval worker,谁在 unit 间隙谁空)
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
echo "parity on GPU $GPU (free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU))"
D=/personal/lcy_data/lcy_full_20260911/sft/parity${ROWS}   # 与源 jsonl 同深度,../../data 才能解析
mkdir -p $D $P/logs/merge
# 只取短行(< 120k 字符 ≈ 3.6 万 token):parity 只看两条入口是否同数值,不需要长行;
# 单卡不带 ZeRO,19 万 token 的行会 OOM(GPU 7 还跑着 eval worker)。
# 行的 JSON 都有十几万字符(带 provider_view 等),按字符长度筛不出来;用 measure_rendered_tokens
# 的报告(tokens.json,对应未过滤的 train.jsonl)挑 tokens_full <= MAX_TOK 的前 ROWS 行。
TOKENS=${TOKENS:-$(dirname $SRC)/tokens.json}
FULL=${FULL:-$(dirname $SRC)/train.jsonl}
MAX_TOK=${MAX_TOK:-12000}
python3 - "$TOKENS" "$FULL" "$D/rows.jsonl" "$ROWS" "$MAX_TOK" <<'PY'
import json, sys
rep, full, out, n, cap = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
want = [r["row_index"] for r in json.load(open(rep))["rows"] if r["tokens_full"] <= cap][:n]
keep = set(want); got = 0
with open(full, encoding="utf-8") as fh, open(out, "w", encoding="utf-8") as oh:
    for i, line in enumerate(fh):
        if i in keep:
            oh.write(line); got += 1
        if got == len(keep): break
print(f"parity rows: {got} (tokens_full <= {cap})")
PY
mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled 2>/dev/null || true
unset PYTHONPATH CUDA_HOME CUDA_PATH CUDA_BIN_PATH CUDACXX
export LD_LIBRARY_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export TORCH_CUDA_ARCH_LIST="10.0+PTX"
export TRITON_PTXAS_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvcc/bin/ptxas
mkdir -p /root/.cache/torch/kernels ; cp -n $P/torchkernels/* /root/.cache/torch/kernels/ 2>/dev/null || true
export ENABLE_MY_GLOBAL_HOOK=0 ENABLE_MY_FLASH_ATTN_HOOK=0 USE_TF=0
export HF_HOME=/root/hf TORCH_EXTENSIONS_DIR=/root/te TORCHINDUCTOR_CACHE_DIR=/root/ti TRITON_CACHE_DIR=/root/tc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled FLASH_ATTENTION_DETERMINISTIC=1
export CUDA_VISIBLE_DEVICES=$GPU
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ "${SKIP_OLD:-0}" != 1 ] && [ "$ONLY" != new ]; then
echo "=== old CLI ($OLD)"; rm -rf $P/checkpoints/_parity_old
(cd $OLD && PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$OLD /usr/bin/python3 -u training/train_sft.py --deterministic --model $BASE --dataset $D/rows.jsonl --output $P/checkpoints/_parity_old \
  --learning-rate 2e-5 --warmup-ratio 0.05 --max-grad-norm 1.0 --max-input-tokens 196608 --num-train-epochs 1 --max-steps 3 \
  --global-batch-size 1 --gradient-accumulation-steps 1 --dataloader-num-workers 0 --freeze-vision --gradient-checkpointing \
  --attn-implementation flash_attention_2 --report-to none --run-name parity_old --logging-steps 1 --save-steps 100000 --save-total-limit 1 \
  --hard-exit-after-train > $P/logs/merge/parity_old$TAG.log 2>&1); echo "old rc=$?"
grep -oE "\{'loss': '[0-9.]+'" $P/logs/merge/parity_old$TAG.log | tr -d "{'" > $P/logs/merge/parity_old$TAG.losses
fi
[ "$ONLY" = old ] && { echo "old only:"; cat $P/logs/merge/parity_old$TAG.losses; exit 0; }
echo "old loss lines: $(wc -l < $P/logs/merge/parity_old$TAG.losses)"

echo "=== new CLI ($NEW)"; rm -rf $P/checkpoints/_parity_new
(cd $NEW && PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$NEW /usr/bin/python3 data_pipeline/select_sft_rows.py --dataset $D/rows.jsonl --output $D/rows.manifest.json --policy configs/data_policies/successful.yaml > $P/logs/merge/parity_select.log 2>&1); echo "select rc=$? $(tail -1 $P/logs/merge/parity_select.log | cut -c1-100)"
(cd $NEW && PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$NEW /usr/bin/python3 -u training/train_sft.py --recipe $NEW/$RECIPE_NEW --model $BASE --data-manifest $D/rows.manifest.json --output $P/checkpoints/_parity_new --run-name parity_new > $P/logs/merge/parity_new$TAG.log 2>&1); echo "new rc=$?"
grep -oE "\{'loss': '[0-9.]+'" $P/logs/merge/parity_new$TAG.log | tr -d "{'" > $P/logs/merge/parity_new$TAG.losses
echo "new loss lines: $(wc -l < $P/logs/merge/parity_new$TAG.losses)"

echo "=== per-step loss (old | new):"; paste $P/logs/merge/parity_old${OLD_TAG:-}.losses $P/logs/merge/parity_new$TAG.losses
python3 - $P/logs/merge/parity_old${OLD_TAG:-}.losses $P/logs/merge/parity_new$TAG.losses <<'PY'
import sys
a=[float(l.split(':')[1]) for l in open(sys.argv[1]) if ':' in l]; b=[float(l.split(':')[1]) for l in open(sys.argv[2]) if ':' in l]
print("steps", len(a), len(b), "max_abs_diff", max((abs(x-y) for x,y in zip(a,b)), default=float('nan')))
PY
rm -rf $P/checkpoints/_parity_old $P/checkpoints/_parity_new
