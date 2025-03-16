#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/y/yanggao/HAT
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx8_ERN5.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=4321 hat/train.py -opt options/train/train_HAT_SRx2_from_scratch.yml --launcher pytorch