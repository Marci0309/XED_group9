#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=10000
module purge
module load Python/3.10.4-GCCcore-11.3.0

source $HOME/venvs/LLM/bin/activate

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
python pipeline/train_lora_llm.py