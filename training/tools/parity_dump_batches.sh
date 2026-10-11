#!/usr/bin/env bash
# 合并门诊断:两个入口各跑 1 步(同一行),dump 第一步 batch 与参数 checksum,再逐 key 比较。
# 用法:ROWS_DIR=<parity1 目录> bash training/tools/parity_dump_batches.sh
P=/personal/ppc_overfit_20260810
OLD=$P/repo_sft; NEW=$P/repo_sft_merge; BASE=$P/base_models/Qwen3.5-9B
D=${ROWS_DIR:-/personal/lcy_data/lcy_full_20260911/sft/parity1}
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)}
export LD_LIBRARY_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export TORCH_CUDA_ARCH_LIST="10.0+PTX"
export TRITON_PTXAS_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvcc/bin/ptxas
export ENABLE_MY_GLOBAL_HOOK=0 ENABLE_MY_FLASH_ATTN_HOOK=0 USE_TF=0
export HF_HOME=/root/hf TORCH_EXTENSIONS_DIR=/root/te TORCHINDUCTOR_CACHE_DIR=/root/ti TRITON_CACHE_DIR=/root/tc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled FLASH_ATTENTION_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $P/logs/merge
echo "dump on GPU $GPU rows dir $D"
(cd $OLD && PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$OLD /usr/bin/python3 -u $NEW/training/tools/dump_first_batch.py --dump $P/logs/merge/batch_old.pt -- --deterministic --model $BASE --dataset $D/rows.jsonl --output $P/checkpoints/_parity_old \
  --learning-rate 2e-5 --warmup-ratio 0.05 --max-grad-norm 1.0 --max-input-tokens 196608 --num-train-epochs 1 --max-steps 1 \
  --global-batch-size 1 --gradient-accumulation-steps 1 --dataloader-num-workers 0 --freeze-vision --gradient-checkpointing \
  --attn-implementation flash_attention_2 --report-to none --run-name parity_old --logging-steps 1 --save-steps 100000 --save-total-limit 1 \
  --hard-exit-after-train > $P/logs/merge/dump_old.log 2>&1); echo "old rc=$?"
grep -E "dump_first_batch|'loss'" $P/logs/merge/dump_old.log | grep -oE "saved [^:]+|'loss': '[0-9.]+'" | head -3
sed "s/^max_steps: .*/max_steps: 1/" $NEW/configs/recipes/parity_9b_single_gpu.yaml > $P/logs/merge/parity_1step.yaml
(cd $NEW && PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$NEW /usr/bin/python3 -u $NEW/training/tools/dump_first_batch.py --dump $P/logs/merge/batch_new.pt -- --recipe $P/logs/merge/parity_1step.yaml --model $BASE --data-manifest $D/rows.manifest.json --output $P/checkpoints/_parity_new --run-name parity_new > $P/logs/merge/dump_new.log 2>&1); echo "new rc=$?"
grep -E "dump_first_batch|'loss'" $P/logs/merge/dump_new.log | grep -oE "saved [^:]+|'loss': '[0-9.]+'" | head -3
echo "=== compare:"
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$NEW /usr/bin/python3 $NEW/training/tools/dump_first_batch.py --compare $P/logs/merge/batch_old.pt $P/logs/merge/batch_new.pt
