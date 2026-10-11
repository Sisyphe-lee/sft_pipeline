# AI Studio builds this file without a local Docker build context.
# Parent supplies Python 3.12, Torch 2.8/cu128, SAPIEN and /tmp/curobo.
FROM reg.docker.alibaba-inc.com/aii/aistudio:19900157-20260915210256

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PYTHONPATH="" LD_PRELOAD="" LD_LIBRARY_PATH="" TRITON_PTXAS_PATH="" \
    PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# virtualenv includes its own pip seed; python3-venv/ensurepip is not required.
RUN python3 -m pip install --no-cache-dir --target /opt/virtualenv virtualenv==21.14.6 && \
    PYTHONPATH=/opt/virtualenv python3 -m virtualenv --no-periodic-update --system-site-packages /opt/physicalagent/robot && \
    PYTHONPATH=/opt/virtualenv python3 -m virtualenv --no-periodic-update --system-site-packages /opt/physicalagent/sam3 && \
    PYTHONPATH=/opt/virtualenv python3 -m virtualenv --no-periodic-update /opt/physicalagent/vllm

RUN /opt/physicalagent/robot/bin/python -m pip install --no-cache-dir \
      embreex==4.4.0 manifold3d==3.5.2 numpy-quaternion==2024.0.13 \
      nvidia-cuda-nvrtc-cu12==12.9.86 pycollada==0.9.3 rtree==1.4.1 \
      svg.path==7.1 vhacdx==0.0.10 warp-lang==1.16.0 xatlas==0.0.11 yourdfpy==0.0.60 && \
    /opt/physicalagent/sam3/bin/python -m pip install --no-cache-dir \
      transformers==5.14.1 tokenizers==0.22.2 huggingface-hub==1.24.0 \
      safetensors==0.8.0 pillow==12.3.0 && \
    /opt/physicalagent/vllm/bin/python -m pip install --no-cache-dir \
      --only-binary=vllm,torch vllm==0.29.0

# Retain the parent's exact CuRobo snapshot; discard GPU-specific binaries.
RUN test -d /tmp/curobo/src/curobo && \
    cp -a /tmp/curobo /opt/physicalagent/curobo && \
    rm -rf /opt/physicalagent/curobo/.git && \
    find /opt/physicalagent/curobo/src/curobo/curobolib -maxdepth 1 -name '*.so' -delete && \
    sed -i 's/wp\.torch\.device_from_torch/wp.device_from_torch/g' \
      /opt/physicalagent/curobo/src/curobo/geom/sdf/world_mesh.py

# NVIDIA userspace driver libraries are supplied by the Job GPU runtime.
RUN mkdir -p /etc/vulkan/icd.d && \
    printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.0"}}' \
      > /etc/vulkan/icd.d/nvidia_icd.json
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

# Build nodes need no GPU. Real rendering/model execution happens in the Job.
RUN /opt/physicalagent/robot/bin/python -c 'import torch,sapien,warp,trimesh; assert torch.__version__.startswith("2.8.")' && \
    /opt/physicalagent/sam3/bin/python -c 'from transformers import Sam3Model,Sam3Processor' && \
    /opt/physicalagent/vllm/bin/python -c 'import importlib.metadata as m; assert m.version("vllm")=="0.29.0"' && \
    /opt/physicalagent/robot/bin/python -m pip freeze > /opt/physicalagent/robot.freeze.txt && \
    /opt/physicalagent/sam3/bin/python -m pip freeze > /opt/physicalagent/sam3.freeze.txt && \
    /opt/physicalagent/vllm/bin/python -m pip freeze > /opt/physicalagent/vllm.freeze.txt

ENV VLLM_PYTHON=/opt/physicalagent/vllm/bin/python \
    ROBOTWIN_PYTHON=/opt/physicalagent/robot/bin/python \
    ROBOTWIN_SAM3_PYTHON=/opt/physicalagent/sam3/bin/python \
    CUROBO_SOURCE=/opt/physicalagent/curobo
