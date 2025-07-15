#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --time=0-00:30:00
#SBATCH --job-name=convert_v01_100m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user="ali.saberi@arcinstitute.org"
#SBATCH --output=mach/slurm/sbatch-logs/convert_v01_100m_%j.out
#SBATCH --error=mach/slurm/sbatch-logs/convert_v01_100m_%j.err
#SBATCH --export=ALL

# Activate conda environment
source /home/saberi/miniconda/etc/profile.d/conda.sh
conda activate mach2

# Define variables
TRAINING_CONFIG_PATH="/home/saberi/projects/mach/mach-2/savanna/mach/configs/model/v01-100m-r3-r2c.yml"
TRAINING_CHECKPOINT_PATH="/home/saberi/projects/mach/mach-2/savanna/mach/checkpoints/v01-100m-r2"
ITERATION=52224
INFERENCE_CHECKPOINT_PATH="/home/saberi/projects/mach/vortex/mach/checkpoints/v01-100m-r2/52224"

# Create output directory if it doesn't exist
mkdir -p "$INFERENCE_CHECKPOINT_PATH"

# Run the conversion script
python -m mach.scripts.convert_checkpoint_to_vortex \
    --config_path "$TRAINING_CONFIG_PATH" \
    --checkpoint_path "$TRAINING_CHECKPOINT_PATH" \
    --iteration "$ITERATION" \
    --new_checkpoint_path "$INFERENCE_CHECKPOINT_PATH" 