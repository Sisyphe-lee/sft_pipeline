#!/usr/bin/env bash
# Formal job: owned-device kernel checks, then the existing SFT entry point.
set -euo pipefail
for key in BASE DM DM_SHA DATA_ROOT OUT L RUN SFT_STACK_ROOT SFT_HF_OVERLAY CODE_COMMIT; do
  test -n "${!key:-}" || { echo "Missing $key" >&2; exit 2; }
done
export REPO=$PWD
test "$(git rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
export RECIPE=$REPO/configs/recipes/${RECIPE_FILE:-qwen38_flashnext_sft_experimental.yaml}
export EXPECTED_MODEL_TYPE=qwen4_exp
export PY=${PY:-/usr/bin/python3}
export PYTHONPATH=$SFT_HF_OVERLAY:$SFT_STACK_ROOT/qwen35-fla:$SFT_STACK_ROOT/qwen35-train:$REPO
export ENABLE_MY_GLOBAL_HOOK=0 ENABLE_MY_FLASH_ATTN_HOOK=0 USE_TF=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST='10.0+PTX'
# Retain the job image's NVRTC / ptxas paths for B300 (SM 10.3).
unset CUDA_HOME CUDA_PATH CUDA_BIN_PATH CUDACXX WANDB_DISABLED WANDB_SILENT
export WANDB_BASE_URL=http://33.180.4.104 WANDB_MODE=online WANDB_DIR=$OUT
export WANDB_ENTITY=${WANDB_ENTITY:-lingcong-hyb} WANDB_PROJECT=physicalagent-sft
mkdir -p "$L"
host=$(hostname)
trap 'rc=$?; echo "$rc" > "$L/exit-$host.txt"' EXIT
exec > >(tee -a "$L/startup-$host.log") 2>&1
# PYTHONPATH silently ignores missing directories, then imports the image's
# older packages. Validate the mount before any heavyweight Python import.
for file in "$SFT_HF_OVERLAY/transformers/__init__.py" \
            "$SFT_HF_OVERLAY/tokenizers/__init__.py" \
            "$SFT_STACK_ROOT/qwen35-fla/fla/__init__.py" \
            "$SFT_STACK_ROOT/qwen35-train/trl/__init__.py"; do
  test -f "$file" || { echo "Missing runtime dependency: $file" >&2; exit 2; }
done
bash jobs/tools/preflight_train.sh > "$L/preflight-$host.log" 2>&1
"$PY" - <<'PY'
import os
from pathlib import Path
import torch, transformers, tokenizers, trl
for package in (torch, transformers, tokenizers, trl):
    print(f"STACK {package.__name__}={package.__version__} file={package.__file__}", flush=True)
overlay = Path(os.environ['SFT_HF_OVERLAY']).resolve()
for package, version in ((transformers, '5.16.0'), (tokenizers, '0.23.1')):
    if package.__version__ != version or not Path(package.__file__).resolve().is_relative_to(overlay):
        raise RuntimeError(f"{package.__name__} must load version {version} from {overlay}")
if os.environ.get('SFT_CHECK_ENV_ONLY') != '1' and not torch.cuda.is_available():
    raise RuntimeError('No CUDA device available for the training job')
PY
if [ "${SFT_CHECK_ENV_ONLY:-0}" = 1 ]; then
  echo 'FLASHNEXT_ENV_CHECK_OK'
  exit 0
fi
# No separate smoke Job; fail before loading weights if this image's kernel is wrong.
CUDA_VISIBLE_DEVICES=0 "$PY" -m pytest -q -s training/backends/flashnext/test_attention.py \
  -k gpu_kernel > "$L/kernel-$host.log" 2>&1
export NODES=4
eval "$(bash jobs/tools/rendezvous.sh)"
"$PY" -m torch.distributed.run --nnodes=4 --node_rank="$NR" \
  --rdzv_backend=static --master_addr="$MA" --master_port="${MASTER_PORT:-29500}" \
  --max-restarts=0 --nproc_per_node=8 training/train_sft.py \
  --recipe "$RECIPE" --model "$BASE" --data-manifest "$DM" --data-root "$DATA_ROOT" \
  --output "$OUT" --run-name "$RUN" > "$L/train-node-$NR.log" 2>&1
