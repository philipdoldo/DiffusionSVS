#!/bin/bash

#SBATCH --job-name=bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/data_r60_1/adv_robust_project/DiffusionSVS/logs/inference_%A.log
#SBATCH -e /mnt/data_r60_1/adv_robust_project/DiffusionSVS/logs/inference_%A.err
#SBATCH --time=2-00:00:00

###source /home/pdoldo/fs/bin/activate
source $(conda info --base)/etc/profile.d/conda.sh
conda activate drose


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export TORCH_MAX_MEMORY_FRACTION=0.98

#srun python /mnt/data_r60_1/adv_robust_project/DiffusionSVS/inference.py


set -euo pipefail

EXP_DIR="/mnt/data_r60_1/adv_robust_project/DiffusionSVS/experiments/_train-dit-flow-patchify=2-15m-popcs-sanity-check/07-14-2026-01h07m15s"
CONFIG="${EXP_DIR}/config.toml"
CKPT_DIR="${EXP_DIR}/checkpoints"

mkdir -p logs

num_iters=(100 10 1)

for ckpt in $(find "$CKPT_DIR" -name 'checkpoint_step*.pt' | sort -V); do
    for num_iter in "${num_iters[@]}"; do
        echo "Running $(basename "$ckpt") with ${num_iter} sampling iterations"
        srun python /mnt/data_r60_1/adv_robust_project/DiffusionSVS/inference.py "$ckpt" --config "$CONFIG" --num_iter "$num_iter"
    done
done