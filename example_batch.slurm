#!/bin/bash

# Lines beginning with #SBATCH specify your computing resources and other logistics about how to run your job.
# Access computing resources allocated to the MLRE course account, the section may be 006 or 007 depending on the student.
#SBATCH --account=eecs498f25s007_class
# Specify the maximum runtime (in Hours:Minutes:Seconds). If your job hits that runtime, it will be terminated.
#SBATCH --time=5:00
# Specify whether to use a GPU (partition=gpu) or CPU (partition=standard). If you use the GPU partition, only request one GPU (gpus=1). Note that if you use the standard partition, you may have to remove the gpu configuration.
#SBATCH --partition=gpu
#SBATCH --gpus=1
# There are also more specific settings for configuring your CPU/GPU. You can reference the documentation for more information.
# Specify the amount of memory you need. If your job exceeds this memory limit, it will be terminated.
#SBATCH --mem=1g
# Name this job and the output file
#SBATCH --job-name=example_job
#SBATCH --output=example_output.out
# Receive an email when your job starts and ends
#SBATCH --mail-user=UNIQNAME@umich.edu
#SBATCH --mail-type=BEGIN,END

# Load Python and any other desired modules
module load python/3.9.12
module load cuda/11.8.0

# Specify the script you want to run
cd /scratch/eecs498f25s006_class_root/eecs498f25s006_class/shared_data/group2_DrugCLIP/eecs498-DrugCLIP
source ./.venv/bin/activate
python ./unimol/test.py --user-dir ./unimol $data_path "./data" --valid-subset test --results-path ./test --num-workers 8 --ddp-backend=c10d --batch-size 8 --task drugclip --loss in_batch_softmax --arch drugclip --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --seed 1 --path checkpoint_best.pt --log-interval 100 --log-format simple --max-pocket-atoms 511 --test-task PCBA

