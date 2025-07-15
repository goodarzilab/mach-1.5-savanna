import glob
import os
import re

import torch

SOURCE_TAG="global_step100"
SOURCE_DIR = "converted_checkpoints/"
TARGET_TAG="global_step0"
TARGET_DIR = "extended_checkpoints/512K/merged"
SOURCE_CHECKPOINT_DIR = os.path.join(SOURCE_DIR, SOURCE_TAG)
TARGET_CHECKPOINT_DIR = os.path.join(TARGET_DIR, TARGET_TAG)
FILTER_PATTERN = "filter.h"
HYENA_MEDIUM_FILTER_LEN = 128
# Specific to evo2 40b training on 2048 GPUs
MP_SIZE = 8
DP_SIZE = 256
NUM_GROUPS = 512

SOURCE_SEQ_LEN = 8192
TARGET_SEQ_LEN = 512 * 2 ** 10

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def get_checkpoint_files(checkpoint_dir, glob_pattern):
    # XXX: need to test that this simple glob rule works for multi-node setup too
    ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, glob_pattern)), key=natural_keys)

    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"can't find {glob_pattern} files in directory '{checkpoint_dir}'")

    return ckpt_files

def get_filter_stats(filter):
    return filter.mean(), filter.min(), filter.max()


source_model_shard_paths = get_checkpoint_files(SOURCE_CHECKPOINT_DIR, "mp*.pt")
target_model_shard_paths = get_checkpoint_files(TARGET_CHECKPOINT_DIR, "mp*.pt")

assert len(source_model_shard_paths) == len(target_model_shard_paths), f"Expected {len(source_model_shard_paths)} source model shard paths, found {len(target_model_shard_paths)} target model shard paths"
assert len(source_model_shard_paths) == 8, f"Expected {len(source_model_shard_paths)} source model shard paths, found {len(target_model_shard_paths)} target model shard paths"

def get_filter_stats(filter):
    return filter.mean(), filter.min(), filter.max()
for source_model_shard_path, target_model_shard_path in zip(source_model_shard_paths, target_model_shard_paths):
    print(f"Checking {os.path.basename(source_model_shard_path)} and {os.path.basename(target_model_shard_path)}")
    assert os.path.basename(source_model_shard_path) == os.path.basename(target_model_shard_path), f"Expected source and target model shard paths to be the same, found {os.path.basename(source_model_shard_path)} and {os.path.basename(target_model_shard_path)}"

    source_model_dict = torch.load(source_model_shard_path)['module']
    target_model_dict = torch.load(target_model_shard_path)['module']

    for name, source_param in source_model_dict.items():
        if FILTER_PATTERN in name:
            target_param = target_model_dict[name]

            print(f"{name}: {source_param.shape} -> {target_param.shape}")

            within_filter_mean, within_filter_min, within_filter_max = get_filter_stats(source_param[:,:HYENA_MEDIUM_FILTER_LEN])
            outside_filter_mean, outside_filter_min, outside_filter_max = get_filter_stats(source_param[:,HYENA_MEDIUM_FILTER_LEN:])
            print(f"{name}: {source_param.shape} {within_filter_mean=}:{outside_filter_mean=} | {within_filter_min=}:{outside_filter_min=} | {within_filter_max=}:{outside_filter_max=}")

            target_within_filter_mean, target_within_filter_min, target_within_filter_max = get_filter_stats(target_param[:,:HYENA_MEDIUM_FILTER_LEN])
            target_outside_filter_mean, target_outside_filter_min, target_outside_filter_max = get_filter_stats(target_param[:,HYENA_MEDIUM_FILTER_LEN:SOURCE_SEQ_LEN])
            print(f"{name}: {target_param.shape} {target_within_filter_mean=}:{target_outside_filter_mean=} | {target_within_filter_min=}:{target_outside_filter_min=} | {target_within_filter_max=}:{target_outside_filter_max=}")
  

# for source_model_shard_path in source_model_shard_paths:
#     print(f"Checking {os.path.basename(source_model_shard_path)}")

#     source_model_dict = torch.load(source_model_shard_path)['module']
#     for name, source_param in source_model_dict.items():

#         if FILTER_PATTERN in name:
#             breakpoint()
#             within_filter_mean, within_filter_min, within_filter_max = get_filter_stats(source_param[:,:HYENA_MEDIUM_FILTER_LEN])
#             outside_filter_mean, outside_filter_min, outside_filter_max = get_filter_stats(source_param[:,HYENA_MEDIUM_FILTER_LEN:])
#             print(f"{name}: {source_param.shape} {within_filter_mean=}:{outside_filter_mean=} | {within_filter_min=}:{outside_filter_min=} | {within_filter_max=}:{outside_filter_max=}")
