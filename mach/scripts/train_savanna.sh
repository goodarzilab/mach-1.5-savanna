### Overfit testing

python mach/scripts/prepare_data.py \
    --input_file_list /large_storage/goodarzilab/saberi/mach/mach-2/refseq/overfit/seqs_processed_sampled_list.txt \
    --output_dir /large_storage/goodarzilab/saberi/mach/mach-2/refseq/overfit

python launch.py \
    train.py \
    --wandb_project mach \
    --wandb_group v00-7m \
    --wandb_run_name v00-7m-r1 \
    --conf_dir mach/configs data/overfit.yml model/v00-7m.yml

python mach/scripts/slurm_submit.py \
    --job_name v00-6.5m-r1 \
    --wandb_group v00-6.5m \
    --model_config model/v00-6.5m.yml \
    --data_config data/overfit.yml \
    --time 0-01:00:00 \
    --gpus 1 \
    --cpus_per_task 8 \
    --no_submit
    
python mach/scripts/slurm_submit.py \
    --job_name v00-12.5m-r1 \
    --wandb_group v00-12.5m \
    --model_config model/v00-12.5m.yml \
    --data_config data/overfit.yml \
    --time 0-01:00:00 \
    --gpus 1 \
    --cpus_per_task 8

python mach/scripts/slurm_submit.py \
    --job_name v00-12.5m-r1 \
    --wandb_group v00-12.5m \
    --model_config model/v00-12.5m-r1-c1.yml \
    --data_config data/overfit.yml \
    --time 0-02:00:00 \
    --gpus 1 \
    --cpus_per_task 8

python mach/scripts/slurm_submit.py \
    --job_name v00-50m-r0 \
    --wandb_group v00-50m \
    --model_config model/v00-50m.yml \
    --data_config data/overfit.yml \
    --time 0-00:05:00 \
    --gpus 2 \
    --cpus_per_task 8

python mach/scripts/slurm_submit.py \
    --job_name v00-100m-r1 \
    --wandb_group v00-100m \
    --model_config model/v00-100m-r1.yml \
    --data_config data/overfit.yml \
    --time 0-04:00:00 \
    --gpus 2 \
    --cpus_per_task 8

python mach/scripts/slurm_submit.py \
    --job_name v00-100m-r2 \
    --wandb_group v00-100m \
    --model_config model/v00-100m-r2.yml \
    --data_config data/overfit.yml \
    --time 0-04:00:00 \
    --gpus 2 \
    --cpus_per_task 8

### Euarchontoglires
# find /large_storage/goodarzilab/saberi/mach/mach-2/refseq/processed  -maxdepth 1 -name "*.jsonl.zst" -type f | sort > mach/configs/data/euarchontoglires.data_list.txt
python mach/scripts/prepare_data.py \
    --input_file_list mach/configs/data/euarchontoglires.data_list.txt \
    --output_dir /large_storage/goodarzilab/saberi/mach/mach-2/refseq/tokenized

python mach/scripts/create_data_config.py \
    --tokenized-data-dir /large_storage/goodarzilab/saberi/mach/mach-2/refseq/tokenized \
    --config-name euarchontoglires.yml

#   "train-total-chars": 40767590224,
#   "valid-total-chars": 2455386666,

## training: number of iteretions per epoch: (train-total-chars) / (train-micro-batch-size-per-gpu * max-seq-length)
## 40767590224 / (14 * 65536) = 43,520
## 10 epochs: 435,200
## 5% warmup: 21,760 (0.5 epochs)
## eval-interval every 20% epoch: 8,704
## save checkpoint every 40% epoch: 17,773

## validation: number of iteretions per epoch: (valid-total-chars) / (valid-micro-batch-size-per-gpu * max-seq-length)
## 2455386666 / (14 * 65536) = 2,676
## 20% for eval-iter: 535

python mach/scripts/slurm_submit.py \
    --job_name v01-100m-euarchontoglires \
    --wandb_group v01-100m \
    --model_config model/v01-100m-r2.yml \
    --data_config data/euarchontoglires.yml \
    --time 1-00:00:00 \
    --gpus 4 \
    --cpus_per_task 16
    
python mach/scripts/slurm_submit.py \
    --job_name v01-100m-euarchontoglires \
    --wandb_group v01-100m \
    --model_config model/v01-100m-r2-c1.yml \
    --data_config data/euarchontoglires.yml \
    --partition gpu_batch \
    --time 4-00:00:00 \
    --gpus 4 \
    --cpus_per_task 16

python mach/scripts/slurm_submit.py \
    --job_name v01-100m-euarchontoglires \
    --wandb_group v01-100m \
    --model_config model/v01-100m-r3-r2c.yml \
    --data_config data/euarchontoglires.yml \
    --partition gpu_batch \
    --time 2-00:00:00 \
    --gpus 4 \
    --cpus_per_task 16

python mach/scripts/slurm_submit.py \
    --job_name v01-100m-euarchontoglires \
    --wandb_group v01-100m \
    --model_config model/v01-100m-r3-r2c.yml \
    --data_config data/euarchontoglires.yml \
    --partition gpu_batch \
    --time 2-00:00:00 \
    --gpus 2 \
    --cpus_per_task 16