#!/bin/bash

#SBATCH --partition=gpu_batch
#SBATCH --time=2-00:00:00
#SBATCH --job-name=v01-100m-euarchontoglires
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user="ali.saberi@arcinstitute.org"
#SBATCH --output=mach/slurm/sbatch-logs/v01-100m-euarchontoglires_%j.out
#SBATCH --error=mach/slurm/sbatch-logs/v01-100m-euarchontoglires_%j.err
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
    --wandb_group v01-100m \
    --wandb_run_name ${SLURM_JOB_NAME} \
    --conf_dir mach/configs data/euarchontoglires.yml model/v01-100m-r3-r2c.yml
