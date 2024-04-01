#!/bin/bash

#$ -M sabraha2@nd.edu
#$ -m abe
#$ -q gpu-long
#$ -l gpu_card=1
#$ -l h=!qa-a10-*
#$ -e errors/
#$ -N HOM-clip-fine-tune-hom


module load python
module load intel
module load cuda
module load cudnn


fsync $SGE_STDOUT_PATH &

source /afs/crc.nd.edu/user/s/sabraha2/Projects/SPIE_2024/venv/bin/activate

python train_finetune.py --folder /project01/cvrl/sabraha2/DSIAC_CLIP_DATA/ --batch_size 32 --num_workers 32 --default_root_dir /project01/cvrl/sabraha2/SPIE_2024/fine_tune_hom/
