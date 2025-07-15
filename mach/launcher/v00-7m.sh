# 40b 8K early extension training script
#/lustre/fs01/portfolios/dir/users/jeromek/40b-train-n256-8K/202411291153/
 
NODES=(1)
SAVANNA_ROOT='/home/saberi/projects/mach/mach-2/savanna'
TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/mach/configs/data/overfit.yml
MODEL_CONFIG=$SAVANNA_ROOT/mach/configs/model/v00-7m.yml

# CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=2
PARTITION=gpu_batch
ACCOUNT=ali.saberi@arcinstitute.org

BASE_NAME=v00-7m
LAUNCHER=srun
JOBTIME="1-00:00:00"
SUFFIX="-r1-g2"

# CHECKPOINT_PATH="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-v2/40b_train_v2/202410271619"

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    JOB_NAME=$BASE_NAME-n$N$SUFFIX

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
    --enable-heimdall \
    --enable_async_save \
    --heimdall_log_straggler \
    --expandable_segments \
    --checkpoint_path $CHECKPOINT_PATH \
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
    --wandb-run-name $JOB_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done