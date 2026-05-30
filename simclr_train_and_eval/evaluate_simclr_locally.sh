#!/bin/bash
# Local script for evaluation SimCLR trained Resnet

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to script location
export WORK_DIR=$(dirname "$(realpath "$0")")

# Run evaluation from simclr-pytorch directory
cd ${WORK_DIR}/simclr-pytorch

echo "=========================================="
echo "Batch Linear Evaluation Starting"
echo "Date: $(date)"
echo "=========================================="

# Loop over seeds 1 through 10
for SEED in {1..10}; do
    RUN_NAME="test_all_${SEED}"
    echo "------------------------------------------"
    echo "Running with seed: $SEED"
    echo "Run Name: $RUN_NAME"
    echo "Date: $(date)"
    echo "------------------------------------------"

    python batch_evaluator.py \
        --pretrain_dir $WORK_DIR/logs/simclr_run/exman-train.py/runs/000001 \
        --dataset_path $WORK_DIR/labelled_dataset \
        --output_dir $WORK_DIR/simclr_evaluation \
        --finetune_mode all \
        --epochs 60 \
        --seed $SEED \
        --run_name $RUN_NAME \
        --use_imagenet_head
done

conda deactivate