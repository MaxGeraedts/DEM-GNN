#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --output=test_python-%j.log
#SBATCH --partition=be.gpustudent.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=2gb
#SBATCH --time=00:05:00
#SBATCH --mail-user=m.geraedts1@student.tue.nl
#SBATCH --mail-type=END

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

cd $HOME/DEM-GNN

python test.py