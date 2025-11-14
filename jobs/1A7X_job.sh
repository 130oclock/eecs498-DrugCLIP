#!/bin/bash

#SBATCH --job-name=1A7X-job
#SBATCH --account=eecs498f25s006_class
#SBATCH --partition=spgpu

#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --gpus=1

#SBATCH --output=/home/%u/%x-%j.log

#SBATCH --mail-user=aidand@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Python and any other desired modules
module load python/3.9.12
module load cuda/11.8.0

# Specify the script you want to run
cd /scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/group2_DrugCLIP/eecs498-DrugCLIP
source ./.venv/bin/activate

./retrieval.sh 1A7X