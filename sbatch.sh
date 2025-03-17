#!/bin/bash 
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J HAT_SRx8_ERN5
#SBATCH --mail-user=gaoyang29@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -t 47:59:59
#SBATCH -A m4876

module load pytorch

cd /pscratch/sd/y/yanggao/HAT
source myenv/bin/activate
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/y/yanggao/HAT

# Use all 4 GPUs for faster training
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx8_ERN5.yml --launcher pytorch
