#!/bin/bash
#SBATCH --job-name=BCC_Training
#SBATCH --output=./Jobs/BCC-%j.log
#SBATCH --partition=be.gpuresearch.q,tue.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=32G
#SBATCH --time=72:00:00
#SBATCH --mail-user=m.geraedts1@student.tue.nl
#SBATCH --mail-type=END

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source $HOME/Environments/grad/bin/activate

cd $HOME/DEM-GNN
#python Training.py
python TrainingHetero.py

deactivate
