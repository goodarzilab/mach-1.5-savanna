#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=0-04:00:00
#SBATCH --job-name=v00-100m-r2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user="ali.saberi@arcinstitute.org"
#SBATCH --output=mach/slurm/sbatch-logs/v00-100m-r2_%j.out
#SBATCH --error=mach/slurm/sbatch-logs/v00-100m-r2_%j.err
#SBATCH --export=ALL

source /home/saberi/miniconda/etc/profile.d/conda.sh
conda activate mach2

CWD='/home/saberi/projects/mach/mach-2/savanna'
cd ${CWD}

port=29500
while lsof -i:$port &> /dev/null; do
    port=$((port+1))
done
export MASTER_PORT=$port

python launch.py \
    train.py \
    --wandb_project mach \
    --wandb_group v00-100m \
    --wandb_run_name ${SLURM_JOB_NAME} \
    --conf_dir mach/configs data/overfit.yml model/v00-100m-r2.yml
