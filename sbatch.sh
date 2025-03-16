#!/bin/bash 
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J HAT_SRx8_ERN5
#SBATCH --mail-user=gaoyang29@berkeley.edu
#SBATCH --mail-type=all
#SBATCH -t 23:59:59
#SBATCH -A m4876

module load pytorch

cd /pscratch/sd/y/yanggao/HAT
source myenv/bin/activate
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/y/yanggao/HAT
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx8_ERN5.yml --launcher pytorch
