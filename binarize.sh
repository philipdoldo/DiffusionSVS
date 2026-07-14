#!/bin/bash

#SBATCH --job-name=bin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH -o /mnt/data_r60_1/adv_robust_project/DiffusionSVS/logs/binarize_%A.log
#SBATCH -e /mnt/data_r60_1/adv_robust_project/DiffusionSVS/logs/binarize_%A.err
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

srun python /mnt/data_r60_1/adv_robust_project/DiffusionSVS/data/binarize.py --config "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/configs/_binarize-popcs-gtsinger-chinese.toml"