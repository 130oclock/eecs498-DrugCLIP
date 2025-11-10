#!/bin/bash

#SBATCH --account=eecs498f25s006_class
#SBATCH --partition=spgpu

#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --mem=32g

#SBATCH --job-name=lit-pcba-job
#SBATCH --output=/home/%u/%x-%j.log

#SBATCH --mail-user=aidand@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load Python and any other desired modules
module load python/3.9.12
module load cuda/11.8.0

# Specify the script you want to run
cd /scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/group2_DrugCLIP/eecs498-DrugCLIP
source ./.venv/bin/activate
python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test --results-path ./test --num-workers 8 --ddp-backend=c10d --batch-size 8 --task drugclip --loss in_batch_softmax --arch drugclip --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 --path checkpoint_best.pt --log-interval 100 --log-format simple --max-pocket-atoms 511 --test-task PCBA

