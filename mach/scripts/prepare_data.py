#!/usr/bin/env python3
import argparse
import os
import subprocess

def process_file(input_jsonl_file: str, output_dir: str, workers: int = 16, log_interval: int = 1000,
                tokenizer_type: str = "CharLevelTokenizer", dataset_impl: str = "mmap",
                append_eod: bool = True, meta_attrs: str = None):
    
    """Process a single JSONL.ZST file."""
    print(f"Processing file: {input_jsonl_file}")
    
    basename = os.path.basename(input_jsonl_file).replace('.jsonl.zst', '')
    
    # Define output paths
    output_bin = os.path.join(output_dir, f"{basename}_text_{tokenizer_type}_document.bin")
    output_idx = os.path.join(output_dir, f"{basename}_text_{tokenizer_type}_document.idx")
    
    # Check if output files already exist
    if os.path.isfile(output_bin) and os.path.isfile(output_idx):
        print(f"Output files already exist for {input_jsonl_file}, skipping...")
        return
    
    # Generate the output prefix
    output_prefix = os.path.join(output_dir, basename)
    
    # Construct and run the preprocessing command
    cmd = [
        "python", "tools/preprocess_data.py",
        "--input", input_jsonl_file,
        "--tokenizer-type", tokenizer_type,
        "--output-prefix", output_prefix,
        "--workers", str(workers),
        "--log-interval", str(log_interval),
        "--dataset-impl", dataset_impl,
    ]
    
    if append_eod:
        cmd.append("--append-eod")
    
    if meta_attrs:
        cmd.extend(["--meta-attrs", meta_attrs])
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing file {input_jsonl_file}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Process JSONL.ZST files for tokenization')
    parser.add_argument('--input_file_list', required=True,
                      help='Path to file containing list of JSONL.ZST files to process')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for tokenized files (default: ../tokenized relative to input files)')
    parser.add_argument('--tokenizer_type', type=str, default="CharLevelTokenizer",
                      help='Type of tokenizer to use (default: CharLevelTokenizer)')
    parser.add_argument('--dataset_impl', type=str, default="mmap",
                      help='Dataset implementation type (default: mmap)')
    parser.add_argument('--workers', type=int, default=16,
                      help='Number of workers for processing (default: 16)')
    parser.add_argument('--log_interval', type=int, default=1000,
                      help='Log interval for processing (default: 1000)')
    parser.add_argument('--no_append_eod', action='store_true',
                      help='Disable appending end-of-document token')
    parser.add_argument('--meta_attrs', type=str, default=None,
                      help='Comma-separated list of meta attributes (default: transcript_id,gene_id)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file_list):
        raise FileNotFoundError(f"Input file list not found: {args.input_file_list}")
    
    # Process each file in the list
    with open(args.input_file_list, 'r') as f:
        for line in f:
            input_file = line.strip()
            if input_file:
                
                output_dir = args.output_dir if args.output_dir is not None else os.path.join(os.path.dirname(input_file), '..', 'tokenized')
                os.makedirs(output_dir, exist_ok=True)
                
                process_file(
                    input_file,
                    output_dir=output_dir,
                    workers=args.workers,
                    log_interval=args.log_interval,
                    tokenizer_type=args.tokenizer_type,
                    dataset_impl=args.dataset_impl,
                    append_eod=not args.no_append_eod,
                    meta_attrs=args.meta_attrs
                )

if __name__ == '__main__':
    main() 