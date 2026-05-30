#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="simclr_train"

# Load modules and activate environment
module load python/miniconda3-py3.12-usr
source activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to job submission directory
export WORK_DIR=${SLURM_SUBMIT_DIR}
export IMAGENET_PATH=${WORK_DIR}/unlabelled_dataset
export EXMAN_PATH=${WORK_DIR}/logs/simclr_run

# Create log directory
mkdir -p ${EXMAN_PATH}

# Run training from simclr-pytorch directory
cd ${WORK_DIR}/simclr-pytorch

python train.py \
    --config configs/train_unlabelled_dataset.yaml \
    --name train_run

conda deactivate