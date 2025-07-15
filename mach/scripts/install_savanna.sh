#!/bin/bash
conda create -n mach python=3.12 -y
conda activate mach

pip install --upgrade pip setuptools wheel
conda install -c conda-forge ninja cmake -y

git clone https://github.com/Zymrael/savanna.git
cd savanna
rm savanna/data/*.so
make setup-env

conda install -c conda-forge cudatoolkit cudnn -y

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

CUDA_HEADERS_DIR=$(dirname $(find $CONDA_PREFIX -name cuda.h | grep 'nvidia/cuda_runtime/include' | head -n 1))
CUDNN_HEADERS_DIR=$(dirname $(find $CONDA_PREFIX -name cudnn.h | grep 'nvidia/cudnn/include' | head -n 1))

export CPATH=$CONDA_PREFIX/include:$CUDA_HEADERS_DIR:$CUDNN_HEADERS_DIR:$CPATH
export NVTE_CUDA_INCLUDE_PATH=$CUDA_HEADERS_DIR

pip install --no-cache-dir transformer_engine[pytorch]
make setup-env
pip install lm_dataformat ftfy

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

cat <<EOF > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#!/bin/bash
export CUDA_HOME=\$(dirname \$(dirname \$(which nvcc)))
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

# Get the directories for CUDA and cuDNN headers
CUDA_HEADERS_DIR=\$(dirname \$(find \$CONDA_PREFIX -name cuda.h | grep 'nvidia/cuda_runtime/include' | head -n 1))
CUDNN_HEADERS_DIR=\$(dirname \$(find \$CONDA_PREFIX -name cudnn.h | grep 'nvidia/cudnn/include' | head -n 1))

# Export the correct include paths
export CPATH=\$CONDA_PREFIX/include:\$CUDA_HEADERS_DIR:\$CUDNN_HEADERS_DIR:\$CPATH
export NVTE_CUDA_INCLUDE_PATH=\$CUDA_HEADERS_DIR
EOF

chmod +x $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

### OR

conda create -n mach2 --clone /home/mohsen/miniconda/envs/gb_evo2


