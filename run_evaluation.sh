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

# Set the correct PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:/pscratch/sd/y/yanggao/HAT

# Print the PYTHONPATH for debugging
echo "PYTHONPATH: $PYTHONPATH"

# Print loaded packages for debugging
echo "Checking if hat.data module can be imported:"
python -c "import hat.data; print(hat.data); print('hat.data module imported successfully')"

# Step A: Run the HAT model evaluation
echo "Running HAT model evaluation..."
CUDA_VISIBLE_DEVICES=0 python hat/data/evaluate_hat.py --opt options/test/HAT_ERN5_SRx8.yml --scale 8 --batch_size 32 --num_gpu 1 --save_results

