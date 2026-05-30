#!/bin/bash
# Local evaluation of baseline ResNet model

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to script location
export WORK_DIR=$(dirname "$(realpath "$0")")

echo "=========================================="
echo "Batch ResNet Baseline Evaluation Starting"
echo "Date: $(date)"
echo "=========================================="

# Loop over seeds 1 through 10
for SEED in {1..10}; do
    RUN_NAME="seed_${SEED}"
    
    echo "------------------------------------------"
    echo "Running with seed: $SEED"
    echo "Run Name: $RUN_NAME"
    echo "Date: $(date)"
    echo "------------------------------------------"
    
    python resnet_evaluator.py \
        --data-dir $WORK_DIR/labelled_dataset \
        --output-dir $WORK_DIR/baseline_resnet_linear_evaluation \
        --run-name $RUN_NAME \
        --seed $SEED
done

conda deactivate