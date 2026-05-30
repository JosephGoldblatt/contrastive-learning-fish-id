#!/bin/bash
# Local training script for SimCLR (non-HPC version)

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to script location
export WORK_DIR=$(dirname "$(realpath "$0")")
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