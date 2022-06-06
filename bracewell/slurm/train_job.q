#!/bin/bash

#SBATCH --time=160:00:00 # walltime
#SBATCH --nodes=1 # Number of computer nodes
#SBATCH --ntasks=2 # number of process per node
#SBATCH --mem=16G # memory per node
#SBATCH --gres=gpu:1 # number of gpus


# Load libraries to run the code
SRC_DIR=/scratch2/fon022/CFPP/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh

python train.py outputs.output_dir=${OUT_DIR} user_config=${CONFIG}

# example:
# sbatch --export=OUT_DIR=<path to output directory>,CONFIG="<path_to_overriding_config_file>" /scratch2/fon022/CFPP/bracewell/slurm/train_job.q