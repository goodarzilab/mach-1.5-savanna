# 7b 1M ft
# gtdb: /lustre/fs01/portfolios/dir/users/jeromek/n32-7b-1M-gtdb-ft/202412311405
# ecoli: /lustre/fs01/portfolios/dir/users/jeromek/n32-7b-1M-ecoli-ft/202412311830
# klebsiella: /lustre/fs01/portfolios/dir/users/jeromek/n32-7b-1M-klebsiella-ft/202412311835
# animalia: /lustre/fs01/portfolios/dir/users/jeromek/n32-7b-1M-animal-ft/202501080143
# yeast / fungi - this is already in progress, originally housed in 7b-context-ext folder

NODES=(32)
SAVANNA_ROOT=$(realpath ..)
SCRIPT_DIR=$SAVANNA_ROOT/launcher/7b-context-ext

DATASETS=(gtdb ecoli klebsiella animal)

DS_IDX=3
DATASET=${DATASETS[$DS_IDX]}
DATA_CONFIG=$SAVANNA_ROOT/configs/7b-ft/data_configs/${DATASET}_ft_nvidia.yml

TRAIN_SCRIPT=$SAVANNA_ROOT/train.py
MODEL_CONFIG=$SAVANNA_ROOT/configs/7b-ft/model_configs/7b-1M-${DATASET}-ft.yml
CONTAINER=/lustre/fs01/portfolios/dir/projects/dir_arc/heimdall/scalable_container_images/clara-discovery+savanna+arc-evo2_efa-nv-internal+pt24.09-py3_ncclv2.23.4-2024-10-26.sqsh

NUM_GPUS=8
PARTITION=pool0_datahall_a
ACCOUNT=dir_arc

BASE_NAME=7b-ft
LAUNCHER=srun
JOBTIME="48:00:00"

#Get basename with no suffix of MODEL_CONFIG
SUFFIX=$(basename $MODEL_CONFIG | cut -d'.' -f1)

N=${NODES[0]}
NUM_NODES=$N 
RUN_NAME=n$N-$SUFFIX
JOB_NAME=$RUN_NAME

CMD="python generate_distributed_launcher.py \
$JOB_NAME \
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
--wandb-group finetune \
--wandb-run-name $RUN_NAME"    

echo $CMD
eval $CMD
echo -e "\n"