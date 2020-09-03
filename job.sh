#!/bin/bash
#SBATCH --job-name=noise_gpu_matrix
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:tesla:1
#SBATCH --partition=gpu
#SBATCH --mem=50G
#SBATCH --time=0-05:00:00
#SBATCH --account=G_DSG

module load miniconda3

source "$CONDA_PREFIX/etc/profile.d/conda.sh"

conda activate qiskit-0.20.0-cuda-10.2

python3 test_gpu.py

conda deactivate
