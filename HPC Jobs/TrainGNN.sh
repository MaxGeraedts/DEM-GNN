#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --output=test_python-%j.log
#SBATCH --partition=tue.default.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb
#SBATCH --time=00:05:00

module purge
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch/2.1.2-foss-2023a

cd $HOME/DEM-GNN

python test.py