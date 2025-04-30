#!/bin/bash
#SBATCH --job-name=test_python
#SBATCH --output=test_python-%j.log
#SBATCH --partition=be.research.q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=05:00:00
#SBATCH --mail-user=m.geraedts1@student.tue.nl
#SBATCH --mail-type=END

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source $HOME/Environments/grad/bin/activate

procdir=$HOME/DEM-GNN/Data/processed
rawdir=$HOME/DEM-GNN/Data/raw
mkdir $HOME/DEM-GNN/Data
mkdir $rawdir
mkdir $procdir

cd $HOME/DEM-GNN
python Training.py

deactivate

name=N400_Mono
cd $HOME/DEM-GNN/Data
zip -r ${name}_raw.zip $HOME/Data/$name $rawdir/${name}_Data.npy $rawdir/${name}_Topology.npy $rawdir/${name}_BC.npy 
zip ${name}_processed.zip $procdir/${name}_train.pt $procdir/${name}_test.pt $procdir/${name}_validate.pt $procdir/${name}_scale_pos.pt $procdir/${name}_scale_x.pt