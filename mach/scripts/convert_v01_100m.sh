#!/bin/bash

# Define variables
TRAINING_CONFIG_PATH="/home/saberi/projects/mach/mach-2/savanna/mach/configs/model/v01-100m-r3-r2c.yml"
TRAINING_CHECKPOINT_PATH="/home/saberi/projects/mach/mach-2/savanna/mach/checkpoints/v01-100m-v01-100m-r3-r2c"
ITERATION=52224
INFERENCE_CHECKPOINT_PATH="/home/saberi/projects/mach/vortex/mach/checkpoints/v01-100m-r2/52224"

# Create output directory if it doesn't exist
mkdir -p "$NEW_CHECKPOINT_PATH"

# Run the conversion script
python -m mach.scripts.convert_checkpoint_to_vortex \
    --config_path "$TRAINING_CONFIG_PATH" \
    --checkpoint_path "$TRAINING_CHECKPOINT_PATH" \
    --iteration "$ITERATION" \
    --new_checkpoint_path "$INFERENCE_CHECKPOINT_PATH" 