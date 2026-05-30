#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="baseline_resnet_eval"

# Load modules and activate environment
module load python/miniconda3-py3.12-usr
source activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to job submission directory
export WORK_DIR=${SLURM_SUBMIT_DIR}

echo "=========================================="
echo "Batch ResNet Baseline Evaluation Starting"
echo "Job ID: $SLURM_JOB_ID"
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