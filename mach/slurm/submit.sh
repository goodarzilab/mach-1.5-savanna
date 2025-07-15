#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=0-00:05:00
#SBATCH --job-name=v00-7m-r1-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user="ali.saberi@arcinstitute.org"
#SBATCH --output=sbatch-logs/v00-7m-r1-g2_%j.out
#SBATCH --error=sbatch-logs/v00-7m-r1-g2_%j.err
#SBATCH --export=ALL

source /home/saberi/miniconda/etc/profile.d/conda.sh
conda activate mach

CWD='/home/saberi/projects/mach/mach-2/savanna'
cd ${CWD}

python launch.py \
    train.py \
    --wandb_project mach \
    --wandb_group v00-7m \
    --wandb_run_name ${SLURM_JOB_NAME} \
    --conf_dir mach/configs data/overfit.yml model/${SLURM_JOB_NAME}.yml
