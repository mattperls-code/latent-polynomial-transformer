#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --output=my_gpu_job.out
#SBATCH --error=my_gpu_job.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00

module load cuda/11.8

source ./venv/bin/activate

python ./model.py