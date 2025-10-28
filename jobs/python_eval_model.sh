#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=digitallab
#SBATCH --gpus-per-node=h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=10000
module purge
module load Python/3.11.5-GCCcore-13.2.0

source $HOME/venvs/first_env/bin/activate

module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
python src/eval_model.py