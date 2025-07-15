#! /bin/bash

set -euo pipefail
CHECKPOINT_DIR="/lustre/fs01/portfolios/dir/projects/dir_arc/evo/checkpoints/40b-train-n256-extension/128K/interleaved/zero3/MP64DP16/padded/global_step0"
#TARGET_MODEL_CONFIG="/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/extension/1M/40b_1M.yml"
TARGET_MODEL_CONFIG=/lustre/fs01/portfolios/dir/users/jeromek/savanna-40b-ctx-fixed/configs/40b/model_configs/tests/128K/40b_128K_mp64dp16.yml

CMD="python extension-checks/check_lr_scheduler.py --checkpoint_dir $CHECKPOINT_DIR --target_config_path $TARGET_MODEL_CONFIG"

echo $CMD
eval $CMD