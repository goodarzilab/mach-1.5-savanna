# 7b final ablation 

#  
# 7b-sh2-actfix_group-ablation.yml
# 7b-sh2-actfix_group-ablation_short_hyena_mlp.yml
# 7b-sh2-short_hyena_mlp_group-ablation.yml
# 7b-sh2-med_group-ablation.yml
# 7b-sh2-ls_group-ablation.yml

CONFIGS=(7b-sh2-actfix_group-ablation.yml 7b-sh2-actfix_group-ablation_short_hyena_mlp.yml 7b-sh2-short_group-ablation.yml 7b-sh2-med_group-ablation.yml 7b-sh2-ls_group-ablation.yml)
NODES=(32)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-final-test/data_configs/shortphase_v3_nvidia.yml

CONFIG_IDX=3
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-final-test/model_configs/group/${CONFIGS[$CONFIG_IDX]}
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
    --wandb-group group-ablation \
    --wandb-run-name $RUN_NAME"    
    
    echo $CMD
    eval $CMD
    echo -e "\n"

done