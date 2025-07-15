#/bin/bash

set -euo pipefail

ITERATION=12500
BASE_BUCKET=s3://nv-arc-dna-us-east-1
REMOTE_DIR=chkpt/7b-context-extension-n32-hybrid-log_evo1-1M/7b-hybrid-log_evo1-1M/global_step${ITERATION}/
BUCKET=$BASE_BUCKET/$REMOTE_DIR

FLAGS="--recursive --summarize"
LS_CMD="/home/jeromek/aws-cli/bin/aws s3 ls $FLAGS $BUCKET | tail -n 2"


LOCAL_DIR=/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/7b-context-extension-n32-hybrid-log_evo1-1M/7b-hybrid-log_evo1-1M/202412211305/global_step${ITERATION}/
LOCAL_SIZE=$(du -sb $LOCAL_DIR | awk '{print $1}')
echo $LOCAL_SIZE

CMD="/home/jeromek/aws-cli/bin/aws s3 sync "$LOCAL_DIR" "$BUCKET" --exact-timestamps"

echo $CMD
$CMD
# done

# echo $LS_CMD
# eval $LS_CMD