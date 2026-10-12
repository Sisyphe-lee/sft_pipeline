# 在 WebIDE 容器(8 卡 B300)里直接跑 FlashNext 训练或测试前 source 这个文件,之后用 /usr/bin/python3。
#
#   cd <本仓库> && source jobs/tools/ide_env.sh
#
# Job 的 pod 不用它:pod 继承镜像 ENV,由 jobs/tools/run_flashnext_sft.sh 设环境。
# SFT_STACK_ROOT 是 CPFS 上 pin 好的依赖(torch 用镜像自带的 2.8.0+cu128):
#   flashnext-hf-runtime/deps   transformers 5.16.0、tokenizers 0.23.1(qwen4_exp 架构需要)
#   qwendeps/qwen35-fla         FLA、tilelang,以及认 B300(sm_103)的 nvrtc 12.9 和 ptxas 12.9
#   qwendeps/qwen35-train       trl、datasets、deepspeed、accelerate
# 镜像自带的 nvrtc / ptxas 12.8 不认 sm_103a:2026-10-12 在 60003 上不设下面三行,triton 编译 kernel 时
# 报 "ptxas fatal: Value 'sm_103a' is not defined for option 'gpu-name'"。
SFT_STACK_ROOT=${SFT_STACK_ROOT:-/robby/share/cpfs/lingcong.hyb/ppc_overfit_20260810}
NV=$SFT_STACK_ROOT/qwendeps/qwen35-fla/nvidia
for f in "$SFT_STACK_ROOT/flashnext-hf-runtime/deps/transformers/__init__.py" \
         "$SFT_STACK_ROOT/qwendeps/qwen35-fla/fla/__init__.py" \
         "$SFT_STACK_ROOT/qwendeps/qwen35-train/trl/__init__.py" \
         "$NV/cuda_nvrtc/lib/libnvrtc.so.12" "$NV/cuda_nvcc/bin/ptxas"; do
  # PYTHONPATH 里不存在的目录会被安静地跳过,然后导入镜像里的旧包;所以先查文件在不在
  [ -f "$f" ] || { echo "ide_env: 缺少依赖 $f" >&2 ; return 2 ; }
done
unset PYTHONPATH CUDA_HOME CUDA_PATH CUDA_BIN_PATH CUDACXX
export PYTHONPATH=$SFT_STACK_ROOT/flashnext-hf-runtime/deps:$SFT_STACK_ROOT/qwendeps/qwen35-fla:$SFT_STACK_ROOT/qwendeps/qwen35-train:$PWD
export LD_LIBRARY_PATH=$NV/cuda_nvrtc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$NV/cuda_nvrtc/lib/libnvrtc.so.12
export TRITON_PTXAS_PATH=$NV/cuda_nvcc/bin/ptxas
export TORCH_CUDA_ARCH_LIST="10.0+PTX"
export ENABLE_MY_GLOBAL_HOOK=0 ENABLE_MY_FLASH_ATTN_HOOK=0 USE_TF=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
