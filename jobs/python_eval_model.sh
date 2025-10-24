#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=10000
module purge
module load Python/3.11.5-GCCcore-13.2.0

source .venv/LLM/bin/activate

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
python pipeline/eval_model.py