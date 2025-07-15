#!/bin/bash

set -euo pipefail

NODES=(128)
SAVANNA_ROOT=$(realpath ..)

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
MODEL="40b"
DATA_CONFIG=$SAVANNA_ROOT/configs/context-scaling/data_configs/opengenome.yml
ARCHS=("sh1" "sh2" "transformer")
CONTEXT_LENS=("16K")

CONFIGS=()

for context_len in "${CONTEXT_LENS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    CONFIGS+=("${MODEL}-${arch}-${context_len}")
  done
done

CONFIG_IDX=$1
SELECTED_CONFIG=${CONFIGS[$CONFIG_IDX]}
#Print number of configs
echo "Number of configs: ${#CONFIGS[@]}"
echo $SELECTED_CONFIG

MODEL_CONFIG=$SAVANNA_ROOT/configs/context-scaling/model_configs/${MODEL}/generated/${SELECTED_CONFIG}.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0
ACCOUNT=dir_arc

BASE_NAME=${MODEL}-context-scaling
LAUNCHER=srun
JOBTIME="08:00:00"

SUFFIX=$(basename $MODEL_CONFIG | cut -d'.' -f1)

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --disable-checkpoint \
    --use-nvte-flash \
    --avoid_record_streams \
    --expandable_segments \
    --enable-heimdall \
    --use-wandb \
    --launcher $LAUNCHER \
    --job-time $JOBTIME \
    --partition $PARTITION \
    --account $ACCOUNT \
    --container $CONTAINER \
    --num-nodes $NUM_NODES \
    --num-gpus $NUM_GPUS \
    --data-config $DATA_CONFIG \
    --model-config $MODEL_CONFIG \
    --train-script $TRAIN_SCRIPT \
    --wandb-project $BASE_NAME \
    --wandb-group context-scaling \
    --wandb-run-name $RUN_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done