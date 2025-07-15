#! /bin/bash

# saberi@GPU70A0:~/projects/mach/vortex$ nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on Tue_Aug_15_22:02:13_PDT_2023
# Cuda compilation tools, release 12.2, V12.2.140
# Build cuda_12.2.r12.2/compiler.33191640_0

# (vortex) saberi@GPU70A0:~/projects/mach/vortex$ nvidia-smi
# Fri May  9 08:54:39 2025       
# +---------------------------------------------------------------------------------------+
# | NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
# |-----------------------------------------+----------------------+----------------------+
# | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
# |                                         |                      |               MIG M. |
# |=========================================+======================+======================|
# |   0  NVIDIA H100 80GB HBM3          On  | 00000000:4E:00.0 Off |                    0 |
# | N/A   33C    P0              74W / 700W |      0MiB / 81559MiB |      0%      Default |
# |                                         |                      |             Disabled |
# +-----------------------------------------+----------------------+----------------------+

git clone git@github.com:Zymrael/vortex.git
cd vortex

conda create -n vortex python=3.11.5 -y
conda activate vortex

pip install torch torchvision torchaudio

## make sure you have cuDNN installed
# conda install -c conda-forge cudnn

export CUDNN_INCLUDE_DIR=$(dirname $(find $CONDA_PREFIX -name cudnn.h | grep 'nvidia/cudnn/include' | head -n 1))
export CPATH=$CUDNN_INCLUDE_DIR:$CPATH

pip install vtx
