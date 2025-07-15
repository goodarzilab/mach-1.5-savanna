import argparse

import torch
from partition_lib import get_all_model_files, load_model_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    args = parser.parse_args()

    model_files = get_all_model_files(args.checkpoint_dir)

    for model_file in model_files:
        failed_count = 0

        print(f"Checking {model_file}", flush=True)
        model_state = torch.load(model_file, map_location="cpu")
        for name, shape in model_state['param_shapes'][0].items():
            if "w1" in name or "w2" in name:
               # print(f"{model_file} {name} {shape}", flush=True)
                if shape[0] % 16 != 0 and shape[1] % 8 != 0:
                    print(f"{model_file} failed {name} {shape}", flush=True)
                    failed_count += 1
            elif "w3" in name:
                #print(f"{model_file} {name} {shape}", flush=True)
                if shape[1] % 16 != 0 and shape[0] % 8 != 0:
                    print(f"{model_file} failed {name} {shape}", flush=True)
                    failed_count += 1
            # else:
            #     print(f"{model_file} passed {name} {shape}", flush=True)
        
        print(f"Total failed: {failed_count}", flush=True)
        assert failed_count == 0, f"Total failed {model_file}: {failed_count}"
