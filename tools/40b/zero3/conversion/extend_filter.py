import argparse
import re
from pathlib import Path

import torch
from common import (
    DEFAULT_CHECKPOINT_NAME,
    EXPLICIT_FILTER_PATTERNS,
    copy_checkpoint_to_new_dir,
    copy_checkpoints_to_new_dir,
    load_state_dict,
    parse_to_list,
)

NUM_GROUPS = 256
SEQ_LEN = 8192
TARGET_SEQ_LEN = 32768

def fix_model_state(model_state_dict, seq_len, expected_shape, target_shape, output_path):
    """
    Extend the filters in the model state in-place
    """

    assert "module" in model_state_dict
    model_dict = model_state_dict["module"]

    for k in model_dict.keys():
        for pat in args.filter_pattern:
            if pat in k:
                print(f"   -> Fixing {k}, reshaping from {expected_shape} to {target_shape}...")
                w = model_dict[k]
                assert w.shape == expected_shape
                new_w = torch.zeros(target_shape, dtype=w.dtype, device=w.device)
                new_w[:, :seq_len] = w
                assert new_w.shape == target_shape
                assert new_w[:, :seq_len].equal(w)
                model_dict[k] = new_w

    print(f" -> Saving to {output_path}")
    torch.save(model_state_dict, output_path)

def fix_optimizer_states(args, output_dir, seq_len, expected_shape, target_shape):
    """
    Extend the filters in the optimizer states in-place
    """
    for file in output_dir.glob("zero/**/*.pt"):
        match = re.search(args.filter_pattern, str(file))
        if match:
            print(
                f" -> Fixing {file}, reshaping from {expected_shape} to {target_shape}..."
            )

            state_dict = torch.load(file)
            assert 'param' in state_dict

            w = state_dict['param']
            assert w.shape == expected_shape

            new_w = torch.zeros(target_shape, dtype=w.dtype)
            new_w[:, :SEQ_LEN] = w
            assert new_w.shape == target_shape
            assert new_w[:, :seq_len].equal(w)
            
            state_dict['param'] = new_w
            print(f" -> Saving to {file}")
            torch.save(state_dict, str(file))

def extend_filters(args, model_state_dict, output_dir, output_path):
    
    num_groups = args.num_groups
    seq_len = args.seq_len
    target_seq_len = args.target_seq_len
    filter_pattern = args.filter_pattern

    expected_shape = torch.Size([num_groups, seq_len])
    target_shape = torch.Size([num_groups, target_seq_len])
    print(f"Extending filters from {seq_len} to {target_seq_len}...")
    print(f" -> Expected shape: {expected_shape}")
    print(f" -> Target shape: {target_shape}")
    print(f" -> Filter pattern: {filter_pattern}")

    # First fix model state weights
    print(" Extending model state filters...")
    fix_model_state(
        model_state_dict,
        seq_len=seq_len,
        expected_shape=expected_shape,
        target_shape=target_shape,
        output_path=output_path,
    )

    # Fix optimizer states
    if args.optimizer_states:
        print(" Extending filters in optimizer states...")
        fix_optimizer_states(
            args,
            output_dir=output_dir,
            seq_len=seq_len,
            expected_shape=expected_shape,
            target_shape=target_shape,
        )

def get_args():
    parser = argparse.ArgumentParser("Fix explicit filter lengths in a checkpoint", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing the checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_groups", type=int, required=True, help="Number of groups in the filter")
    parser.add_argument("--seq_len", type=int, required=True, help="Sequence length of the source filter")
    parser.add_argument("--target_seq_len", type=int, required=True, help="Target sequence length")
    parser.add_argument("--filter_pattern", type=parse_to_list, default=EXPLICIT_FILTER_PATTERNS, help="Pattern to match for filter weights")
    parser.add_argument("--checkpoint_name", type=str, default=DEFAULT_CHECKPOINT_NAME, help="Name of the checkpoint file")
    parser.add_argument("--optimizer_states", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    return args


def setup_dirs(args):
    source_checkpoint = Path(args.source_dir) / args.checkpoint_name
    assert source_checkpoint.exists(), f"Checkpoint {source_checkpoint} does not exist"
    
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.checkpoint_name

    if output_dir.exists():
        if output_path.exists() and not args.overwrite:
            raise ValueError(
                f"{output_path} already exists, specify --overwrite to overwrite."
            )

        if args.optimizer_states:
            optimizer_states_dir = output_dir / "zero"
            if optimizer_states_dir.exists() and not args.overwrite:
                raise ValueError(
                    f"WARNING: Directory {optimizer_states_dir} already exists, specify --overwrite to overwrite"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.optimizer_states:
        copy_checkpoints_to_new_dir(args.source_dir, output_dir)
    else:
        copy_checkpoint_to_new_dir(source_checkpoint, output_path)
    
    return output_dir, output_path

def main(args):
    output_dir, output_path = setup_dirs(args)
    model_state_dict = load_state_dict(args.source_dir, args.checkpoint_name)
    extend_filters(args, model_state_dict=model_state_dict, output_dir=output_dir, output_path=output_path)
    
if __name__ == "__main__":
    args = get_args()
    main(args)