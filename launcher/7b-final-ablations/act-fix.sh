# 7b final ablation 
# 7b-sh2-base 
# 7b-sh2-actfix
# 7b-sh2-short 
# 7b-sh2-med 
# 7b-sh1

NODES=(32)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-final-test/data_configs/shortphase_v3_nvidia.yml

# 7b-sh2-base 7b-sh2-actfix
# 7b-sh2-short 7b-sh2-med 7b-sh1
# 7b-transformerpp 7b-sh2-ls
#7b-sh2-actfix_longer_warmup.yml
#7b-transformerpp_longer_warmup.yml
#7b-sh2-ls_longer_warmup.yml
#7b-sh2-short_longer_warmup.yml
#7b-sh2-med_longer_warmup.yml
#7b-sh1_longer_warmup.yml

MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-final-test/model_configs/act-fix/v2/7b-sh2-base_longer_warmup.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=7b-final-ablation
LAUNCHER=srun
JOBTIME="48:00:00"

#Get basename with no suffix of MODEL_CONFIG
SUFFIX=$(basename $MODEL_CONFIG | cut -d'.' -f1)

for N in "${NODES[@]}"; do
    NUM_NODES=$N 
    RUN_NAME=n$N-$SUFFIX
    JOB_NAME=$RUN_NAME

    CMD="python generate_distributed_launcher.py \
    $JOB_NAME \
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
    --wandb-group act-fix \
    --wandb-run-name $RUN_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done