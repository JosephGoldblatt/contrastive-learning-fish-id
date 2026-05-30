#!/bin/bash
#SBATCH --account=l40sfree
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --job-name="simclr_eval"

# Load modules and activate environment
module load python/miniconda3-py3.12-usr
source activate pytorch_env

# Set Python environment variables
export PYTHONNOUSERSITE=0
export PYTHONPATH=""

# Set paths relative to job submission directory
export WORK_DIR=${SLURM_SUBMIT_DIR}

# Run evaluation from simclr-pytorch directory
cd ${WORK_DIR}/simclr-pytorch

echo "=========================================="
echo "Batch Linear Evaluation Starting"
echo "Job ID: $SLURM_JOB_ID"
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
