#!/bin/bash

#SBATCH --job-name=4EIY-job
#SBATCH --account=eecs498f25s006_class
#SBATCH --partition=gpu

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

target="4EIY"

results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"
MOL_PATH="./data/custom/$target/mols.lmdb" # path to the molecule file
POCKET_PATH="./data/custom/$target/pocket.lmdb" # path to the pocket file
EMB_DIR="./data/emb" # path to the cached mol embedding file

CUDA_VISIBLE_DEVICES="0" python ./unimol/retrieval.py --user-dir ./unimol $data_path "./data" --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task drugclip --loss in_batch_softmax --arch drugclip  \
       --max-pocket-atoms 256 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --mol-path $MOL_PATH \
       --pocket-path $POCKET_PATH \
       --emb-dir $EMB_DIR \