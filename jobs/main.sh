#!/bin/bash
#SBATCH --job-name=Running_main
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# -----------------------------
# Environment setup
# -----------------------------
module purge
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/first_env/bin/activate

# -----------------------------
# Environment variables
# -----------------------------
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

# -----------------------------
# Run training
# -----------------------------
echo "Running LoRA fine-tuning on $(hostname)"
python -u main.py