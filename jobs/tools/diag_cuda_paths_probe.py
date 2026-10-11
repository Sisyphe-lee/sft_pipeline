#!/usr/bin/env python
"""Probe which CUDA code path breaks on this device.

Four paths, each isolated and reported separately:
  A. torch builtin kernels (SASS shipped in the wheel)
  B. TorchScript NNC fuser  (runtime NVRTC compile; independent of TORCH_CUDA_ARCH_LIST)
  C. jiterator elementwise  (runtime NVRTC compile, cached under ~/.cache/torch/kernels)
  D. cpp_extension load_inline (nvcc at runtime; obeys TORCH_CUDA_ARCH_LIST)
  E. cuRobo FK              (mixes B/D; this is what "expert preflight" runs)
Also prints which libcuda / libnvrtc the process actually mapped, so two hosts
with the same torch wheel can be compared.

Run the same file on the dev container and on a pod; diff the two logs.
"""
import glob
import os
import re
import sys
import traceback

import torch


def hdr(t):
    print(f"\n=== {t}", flush=True)


def section(name, fn):
    hdr(name)
    try:
        fn()
        print(f"[{name.split('.')[0]}] OK", flush=True)
    except Exception:  # noqa: BLE001 - diagnostic, report everything
        traceback.print_exc()
        print(f"[{name.split('.')[0]}] FAIL", flush=True)


hdr("device/driver/libs")
print("python", sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("capability", torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
print("arch_list", torch.cuda.get_arch_list())
try:
    print("driver_version(torch)", torch._C._cuda_getDriverVersion())
except Exception as e:  # noqa: BLE001
    print("driver_version n/a", e)
torch.ones(1, device="cuda")
torch.cuda.synchronize()
maps = open(f"/proc/{os.getpid()}/maps").read().split("\n")
for k in ("libcuda.so", "libnvrtc", "libcudart", "libnvidia-ptxjitcompiler"):
    print(k, sorted({ln.split()[-1] for ln in maps if k in ln})[:3])
print("compat dir:", os.path.exists("/usr/local/cuda/compat"), "compat.disabled:", os.path.exists("/usr/local/cuda/compat.disabled"))
for k in ("LD_LIBRARY_PATH", "TORCH_CUDA_ARCH_LIST", "TORCH_EXTENSIONS_DIR", "PYTORCH_JIT", "PYTORCH_NVFUSER_DISABLE", "CUDA_VISIBLE_DEVICES"):
    print(k, "=", os.environ.get(k))

x = torch.randn(4096, device="cuda")


def path_a():
    torch.cuda.synchronize()
    print("sum", (x * 2 + 1).sin().sum().item())


def path_b():
    @torch.jit.script
    def f(a, b):
        return (a * b + 1.0).sin() * a

    y = None
    for _ in range(4):
        y = f(x, x)
    torch.cuda.synchronize()
    g = str(f.graph_for(x, x))
    print("sum", y.sum().item(), "fused:", ("TensorExprGroup" in g) or ("FusionGroup" in g))


def path_c():
    z = torch.complex(x, x)
    torch.cuda.synchronize()
    print("sum", torch.exp(z).abs().sum().item())


def path_d():
    from torch.utils.cpp_extension import load_inline

    bd = os.environ.get("PROBE_BUILD_DIR") or None
    if bd:
        os.makedirs(bd, exist_ok=True)
    src = r"""
    #include <c10/cuda/CUDAException.h>
    __global__ void addk(const float* a, float* b, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) b[i] = a[i] + 1.0f; }
    torch::Tensor add1(torch::Tensor a) {
      auto b = torch::empty_like(a); int n = a.numel();
      addk<<<(n + 255) / 256, 256>>>(a.data_ptr<float>(), b.data_ptr<float>(), n);
      C10_CUDA_KERNEL_LAUNCH_CHECK(); return b; }
    """
    m = load_inline(name="probe_add1", cpp_sources="torch::Tensor add1(torch::Tensor a);",
                    cuda_sources=src, functions=["add1"], verbose=True, build_directory=bd)
    torch.cuda.synchronize()
    print("diff", m.add1(x).sum().item() - x.sum().item())
    for b in glob.glob(os.path.join(bd or "", "build.ninja")):
        print("gencode:", sorted(set(re.findall(r"arch=compute_\w+,code=\w+", open(b).read()))))


def path_e():
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.base import TensorDeviceType
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml

    cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    m = CudaRobotModel(CudaRobotModelConfig.from_data_dict(cfg, TensorDeviceType()))
    q = torch.zeros(1, m.get_dof(), device="cuda")
    st = m.get_state(q)
    torch.cuda.synchronize()
    print("ee", st.ee_position.flatten().tolist()[:3])


section("A. torch builtin kernel", path_a)
section("B. TorchScript NNC fuser (nvrtc)", path_b)
section("C. jiterator complex elementwise (nvrtc)", path_c)
section("D. cpp_extension load_inline (nvcc)", path_d)
section("E. cuRobo FK", path_e)
hdr("libs mapped at end")
maps = open(f"/proc/{os.getpid()}/maps").read().split("\n")
for k in ("libcuda.so", "libnvrtc", "libnvidia-ptxjitcompiler"):
    print(k, sorted({ln.split()[-1] for ln in maps if k in ln})[:3])
print("\n=== PROBE-END", flush=True)
