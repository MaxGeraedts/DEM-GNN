#!/bin/bash
#SBATCH --job-name=Emb16-ML-Training
#SBATCH --output=./Jobs/Emb16-Training-%j.log
#SBATCH --partition=be.gpuresearch.q,tue.gpu.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16gb
#SBATCH --time=12:00:00
#SBATCH --mail-user=m.geraedts1@student.tue.nl
#SBATCH --mail-type=END

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source $HOME/Environments/grad/bin/activate

cd $HOME/DEM-GNN
python Training.py

deactivate
