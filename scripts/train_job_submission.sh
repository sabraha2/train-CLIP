#!/bin/bash

# Replace 'your_email@domain.com' with your email address
#$ -M your_email@domain.com
#$ -m abe
#$ -q gpu-long
#$ -l gpu_card=1
# Exclude specific nodes if necessary
#$ -l h=!qa-a10-*
#$ -e errors/
#$ -N job_name

# Load necessary modules
module load python
module load intel
module load cuda
module load cudnn

# Activate the virtual environment
source /path/to/your/venv/bin/activate

# Run the training script
# Replace '/path/to/your/dataset' with the path to your dataset
# Replace '/path/to/your/output_directory' with the path where you want to save outputs
python train_finetune.py --folder /path/to/your/dataset --batch_size 32 --num_workers 32 --default_root_dir /path/to/your/output_directory
