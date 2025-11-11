#!/bin/bash

#SBATCH --job-name=dude-job
#SBATCH --account=eecs498f25s006_class
#SBATCH --partition=spgpu

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
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

results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"

TASK="DUDE" # DUDE or PCBA

CUDA_VISIBLE_DEVICES="0" python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \

