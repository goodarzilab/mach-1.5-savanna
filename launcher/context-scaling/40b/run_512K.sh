#!/bin/bash
set -euo pipefail

CONFIGS=("12" "13" "14")
SCRIPT_PATH='/lustre/fs01/portfolios/dir/users/jeromek/savanna-7b-ablations/launcher/context-scaling/40b/_run.sh'
for config in "${CONFIGS[@]}"; do
  CMD="${SCRIPT_PATH} ${config}"
  echo $CMD
  $CMD
  # Wait for user to press enter
  read -s -n 1 -p "Press enter to continue..."
done
