#!/usr/bin/env bash
# 单卡对照 --chunked-ce 与 HF 原生 loss 是否一致(s3 dev container 上跑,一张 B300)。
# 训练环境变量照抄 archive/jobs/train_multi_c910_base27b_mn4c_noovl_robbys7.yml 的 command 段;
# 数据用 c910 训练集的前 N 行(路径相对 REG,用 symlink 指过去),模型用 9B 基座
# (s3 上没有 4B),不用 deepspeed。每步打印 chunked_ce_parity 事件,看 abs_diff。
set -euo pipefail
P=${P:-/personal/ppc_overfit_20260810}
REPO=${REPO:-$P/repo_sft}
BASE=${BASE:-$P/base_models/Qwen3.5-9B}
REG=${REG:-/personal/changkun_data/physicalagent_clean_by_teacher_910_12348rows_v10_training_clean_20260824}
ROWS=${ROWS:-3}
MAX_STEPS=${MAX_STEPS:-$ROWS}
EXTRA_ARGS=${EXTRA_ARGS:-}
GPU=${GPU:-0}
D=$P/data/parity_chunked_ce
OUT=$P/checkpoints/_parity_chunked_ce
LOG=$P/logs/parity_chunked_ce.log
mkdir -p $D
[ -e $D/robotwin ] || ln -s $REG/robotwin $D/robotwin
[ -e $D/robosuite ] || ln -s $REG/robosuite $D/robosuite
head -n $ROWS $REG/sft_turns_c910_train.jsonl > $D/rows.jsonl
rm -rf $OUT
mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled 2>/dev/null || true
unset PYTHONPATH CUDA_HOME CUDA_PATH CUDA_BIN_PATH CUDACXX
export LD_LIBRARY_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib
export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
export LD_PRELOAD=$P/qwendeps/qwen35-fla/nvidia/cuda_nvrtc/lib/libnvrtc.so.12
export TORCH_CUDA_ARCH_LIST="10.0+PTX"
export TRITON_PTXAS_PATH=$P/qwendeps/qwen35-fla/nvidia/cuda_nvcc/bin/ptxas
mkdir -p /root/.cache/torch/kernels ; cp -n $P/torchkernels/* /root/.cache/torch/kernels/ 2>/dev/null || true
export PYTHONPATH=$P/qwendeps/qwen35-fla:$P/qwendeps/qwen35-train:$REPO
export ENABLE_MY_GLOBAL_HOOK=0 ENABLE_MY_FLASH_ATTN_HOOK=0 USE_TF=0
export HF_HOME=/root/hf TORCH_EXTENSIONS_DIR=/root/te TORCHINDUCTOR_CACHE_DIR=/root/ti TRITON_CACHE_DIR=/root/tc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled
export FLASH_ATTENTION_DETERMINISTIC=1
export CUDA_VISIBLE_DEVICES=$GPU
export CHUNKED_CE_PARITY=1
cd $REPO
/usr/bin/python3 -u training/train_sft.py --deterministic --model $BASE --dataset $D/rows.jsonl --output $OUT \
  --learning-rate 2e-5 --warmup-ratio 0.05 --max-grad-norm 1.0 --max-input-tokens 196608 --num-train-epochs 1 \
  --max-steps $MAX_STEPS --global-batch-size 1 --gradient-accumulation-steps 1 --dataloader-num-workers 0 \
  --freeze-vision --gradient-checkpointing --attn-implementation flash_attention_2 --chunked-ce 8192 $EXTRA_ARGS \
  --report-to none --run-name parity_chunked_ce --logging-steps 1 --save-steps 100000 --save-total-limit 1 \
  --hard-exit-after-train > $LOG 2>&1
rc=$?
echo "PARITY_RC=$rc" >> $LOG
grep -E "chunked_ce_parity|Traceback|Error" $LOG | tail -8
rm -rf $OUT
exit $rc
