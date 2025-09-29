#!/bin/bash
#SBATCH --job-name=segformer_train        # Kurzname des Jobs
#SBATCH --output=logs/R-%j.out
#SBATCH --partition=p2
#SBATCH --qos=gpuultimate
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                # Anzahl Knoten
#SBATCH --ntasks=1               # Gesamtzahl der Tasks über alle Knoten hinweg
#SBATCH --cpus-per-task=4        # CPU Kerne pro Task (>1 für multi-threaded Tasks)
#SBATCH --mem=32G                # RAM pro CPU Kern #20G #32G #64G

BASE_DIR=/nfs/scratch/staff/schmittth/code-nexus/transformers
DATA_DIR=${1:-datasets/holz00}

module purge
module load python/anaconda3
eval "$(conda shell.bash hook)"

conda activate conda-transformers

srun python $BASE_DIR/python_scripts/segformer_train.py --data_dir $BASE_DIR/$DATA_DIR
