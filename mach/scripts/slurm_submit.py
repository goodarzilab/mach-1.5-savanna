#!/usr/bin/env python3
import argparse
import os
import subprocess
from typing import Dict, Any

def create_slurm_script(
    job_name: str,
    cwd: str = '/home/saberi/projects/mach/mach-2/savanna',
    env: str = 'mach2',
    partition: str = "gpu",
    time: str = "0-00:05:00",
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int = 8,
    mem_per_cpu: str = "4G",
    gpus: int = 2,
    mail_type: str = "ALL",
    mail_user: str = "ali.saberi@arcinstitute.org",
    wandb_project: str = "mach",
    wandb_group: str = "v00",
    conf_dir: str = "mach/configs",
    data_config: str = "data/overfit.yml",
    model_config: str = None,  # Will default to model/${SLURM_JOB_NAME}.yml
    log_dir: str = "sbatch-logs",
) -> str:
    """Generate a SLURM submission script with an automatic MASTER_PORT assignment."""
    
    if model_config is None:
        model_config = f"model/{job_name}.yml"
    
    # Insert the port check snippet directly before launching python.
    port_check_snippet = """port=29500
while lsof -i:$port &> /dev/null; do
    port=$((port+1))
done
export MASTER_PORT=$port
"""
    
    script_content = f"""#!/bin/bash

#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --gres=gpu:{gpus}
#SBATCH --mail-type={mail_type}
#SBATCH --mail-user="{mail_user}"
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err
#SBATCH --export=ALL

source /home/saberi/miniconda/etc/profile.d/conda.sh
conda activate {env}

CWD='{cwd}'
cd ${{CWD}}

{port_check_snippet}
python launch.py \\
    train.py \\
    --wandb_project {wandb_project} \\
    --wandb_group {wandb_group} \\
    --wandb_run_name ${{SLURM_JOB_NAME}} \\
    --conf_dir {conf_dir} {data_config} {model_config}
"""
    return script_content

def main():
    parser = argparse.ArgumentParser(description="Generate SLURM submission script")
    
    # Environment parameters
    parser.add_argument("--cwd", default="/home/saberi/projects/mach/mach-2/savanna", 
                        help="Working directory for the job")
    parser.add_argument("--env", default="mach2", 
                        help="Conda environment to activate")
    parser.add_argument("--log_dir", default="mach/slurm/sbatch-logs",
                        help="Directory for SLURM output and error logs")
    
    # SLURM parameters
    parser.add_argument("--job_name", required=True, help="Name of the job")
    parser.add_argument("--partition", default="gpu", help="SLURM partition")
    parser.add_argument("--time", default="0-00:05:00", help="Time limit (days-HH:MM:SS)")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of tasks")
    parser.add_argument("--cpus_per_task", type=int, default=8, help="CPUs per task")
    parser.add_argument("--mem_per_cpu", default="4G", help="Memory per CPU")
    parser.add_argument("--gpus", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--mail_type", default="ALL", help="Mail notification type")
    parser.add_argument("--mail_user", default="ali.saberi@arcinstitute.org", help="Email address")
    
    # Wandb and config parameters
    parser.add_argument("--wandb_project", default="mach", help="Weights & Biases project name")
    parser.add_argument("--wandb_group", default="v00", help="Weights & Biases group name")
    parser.add_argument("--conf_dir", default="mach/configs", help="Configuration directory")
    parser.add_argument("--data_config", help="Data configuration file")
    parser.add_argument("--model_config", default=None, 
                        help="Model configuration file (defaults to model/${{SLURM_JOB_NAME}}.yml)")
    
    # Submission control
    parser.add_argument("--no_submit", action="store_true", 
                        help="Only create the script without submitting to SLURM")
    
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Generate the script content
    script_content = create_slurm_script(
        job_name=args.job_name,
        cwd=args.cwd,
        env=args.env,
        partition=args.partition,
        time=args.time,
        nodes=args.nodes,
        ntasks=args.ntasks,
        cpus_per_task=args.cpus_per_task,
        mem_per_cpu=args.mem_per_cpu,
        gpus=args.gpus,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        conf_dir=args.conf_dir,
        data_config=args.data_config,
        model_config=args.model_config,
        log_dir=args.log_dir,
    )
    
    # Write the script to a file
    mach_dir = os.path.dirname(args.conf_dir)
    slurm_dir = os.path.join(mach_dir, "slurm")
    os.makedirs(slurm_dir, exist_ok=True)
    output_file = os.path.join(slurm_dir, f"submit_{args.job_name}.sh")
    with open(output_file, "w") as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(output_file, 0o755)
    
    print(f"Created SLURM submission script: {output_file}")
    
    # Submit the job if --no_submit is not specified
    if not args.no_submit:
        try:
            result = subprocess.run(["sbatch", output_file], 
                                    check=True, 
                                    capture_output=True, 
                                    text=True)
            print(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e.stderr}")
    else:
        print(f"To submit manually, run: sbatch {output_file}")

if __name__ == "__main__":
    main()
