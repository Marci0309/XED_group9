#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
module purge
module load Python/3.10.4-GCCcore-11.3.0

source .venv/LLM/bin/activate

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
python pipeline/eval_model.py