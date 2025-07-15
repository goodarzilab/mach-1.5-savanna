import os
import glob
import json
from pathlib import Path
import numpy as np
import argparse

def main(
    tokenized_data_dir: str = '/large_storage/goodarzilab/saberi/mach/mach-2/data/tokenized',
    wandb_project: str = 'mach',
    data_config_dir: str = 'mach/configs/data',
    config_name: str = 'data_config.yml'
):
    # Get all .bin files and remove the .bin extension
    files = glob.glob(f"{tokenized_data_dir}/*.bin")
    files = [f[:-4] for f in files]  # remove .bin extension

    # Get unique prefixes by removing train/test/validation suffixes
    prefixes = set()
    for f in files:
        # Remove _text suffix and train/test/validation part
        prefix = f.split('.seqs.processed.')[0]
        prefixes.add(prefix)

    # Check which prefixes have all three splits
    valid_prefixes = set()
    for prefix in prefixes:
        train_exists = any(f"{prefix}.seqs.processed.train_text_CharLevelTokenizer_document" in f for f in files)
        test_exists = any(f"{prefix}.seqs.processed.test_text_CharLevelTokenizer_document" in f for f in files)
        valid_exists = any(f"{prefix}.seqs.processed.validation_text_CharLevelTokenizer_document" in f for f in files)
        
        if train_exists and test_exists and valid_exists:
            valid_prefixes.add(prefix)

    # Filter files to only include those with valid prefixes
    # Get file sizes and sort train files by size
    train_files_with_size = [(f, Path(f + '.bin').stat().st_size) 
                            for f in files if "train_text" in f and 
                            any(p in f for p in valid_prefixes)]
    train_files_with_size.sort(key=lambda x: x[1], reverse=True)

    # Take top 255 prefixes
    top_train_files = train_files_with_size[:255]
    train_files = [f[0] for f in top_train_files]

    # Extract prefixes from top train files
    top_prefixes = set()
    for f in train_files:
        prefix = f.split('.seqs.processed.')[0]
        top_prefixes.add(prefix)

    # Filter test and valid files to only include top prefixes
    test_files = sorted([f for f in files if "test_text" in f and
                        any(p in f for p in top_prefixes)])
    valid_files = sorted([f for f in files if "validation_text" in f and
                         any(p in f for p in top_prefixes)])

    # Get weights and counts for each split
    train_weights, train_lines, train_chars = get_normalized_weights(train_files)
    test_weights, test_lines, test_chars = get_normalized_weights(test_files)
    valid_weights, valid_lines, valid_chars = get_normalized_weights(valid_files)

    # Create the configuration dictionary
    config = {
        "train-data-paths": train_files,
        "valid-data-paths": valid_files,
        "test-data-paths": test_files,
        "train-data-weights": train_weights,
        "test-data-weights": test_weights,
        "valid-data-weights": valid_weights,
        "checkpoint_validation_with_forward_pass": False,
        "enforce_sample_length": False
    }

    # Create the statistics dictionary
    stats = {
        "train": {
            "num_files": len(train_files),
            "num_sequences": train_lines,
            "num_chars": train_chars
        },
        "validation": {
            "num_files": len(valid_files),
            "num_sequences": valid_lines,
            "num_chars": valid_chars
        },
        "test": {
            "num_files": len(test_files),
            "num_sequences": test_lines,
            "num_chars": test_chars
        }
    }

    # Write config to file
    os.makedirs(data_config_dir, exist_ok=True)
    output_path = os.path.join(data_config_dir, config_name)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Write stats to file
    stats_path = output_path + '.stats'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print statistics
    def format_number(n):
        return f"{n:,}"

    print(f"Created config with:")
    print(f"Training split:")
    print(f"- {len(train_files)} files")
    print(f"- {format_number(train_lines)} sequences")
    print(f"- {format_number(train_chars)} characters")
    print(f"\nValidation split:")
    print(f"- {len(valid_files)} files")
    print(f"- {format_number(valid_lines)} sequences")
    print(f"- {format_number(valid_chars)} characters")
    print(f"\nTest split:")
    print(f"- {len(test_files)} files")
    print(f"- {format_number(test_lines)} sequences")
    print(f"- {format_number(test_chars)} characters")
    print(f"\nConfig saved to: {output_path}")
    print(f"Statistics saved to: {stats_path}")

def count_lines_and_chars_in_binary(file_path):
    """Count number of lines and characters in a binary file.
    
    Returns:
        tuple: (num_lines, num_chars)
    """
    num_lines = 0
    num_chars = 0
    with open(file_path, 'rb') as f:
        for line in f:
            num_lines += 1
            num_chars += len(line.strip())  # strip to remove newline characters
    return num_lines, num_chars

def format_float(n):
    """Format float to 6 decimal places without scientific notation."""
    return '{:.6f}'.format(n)

def get_normalized_weights(file_list):
    """Calculate normalized weights for each file based on line counts.
    
    Returns:
        tuple: (normalized_weights, total_lines, total_chars)
    """
    counts = [count_lines_and_chars_in_binary(f + '.bin') for f in file_list]
    line_counts = [c[0] for c in counts]
    char_counts = [c[1] for c in counts]
    
    total_lines = sum(line_counts)
    total_chars = sum(char_counts)
    
    # Calculate weights with minimum threshold and format as decimal
    weights = [max(float(format_float(count / total_chars)), 0.000001) for count in char_counts]
    weight_sum = sum(weights)
    normalized_weights = [format_float(w / weight_sum) for w in weights]
    
    return normalized_weights, total_lines, total_chars

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create data configuration for training')
    parser.add_argument('--tokenized-data-dir', type=str, 
                       default='/large_storage/goodarzilab/saberi/mach/mach-2/data/tokenized',
                       help='Directory containing tokenized .bin files')
    parser.add_argument('--data-config-dir', type=str, 
                       default='mach/configs/data',
                       help='Directory to save the config file')
    parser.add_argument('--config-name', type=str, 
                       default='data_config.yml',
                       help='Name of the config file')
    
    args = parser.parse_args()
    main(
        tokenized_data_dir=args.tokenized_data_dir,
        wandb_project=args.wandb_project,
        data_config_dir=args.data_config_dir,
        config_name=args.config_name
    )
