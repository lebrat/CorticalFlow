#!/bin/bash

#SBATCH --time=2:00:00 # walltime
#SBATCH --nodes=1 # Number of computer nodes
#SBATCH --ntasks-per-node=1 # number of process per node
#SBATCH --cpus-per-task=1 # number of threads per process
#SBATCH --mem-per-cpu=32G # memory per node
#SBATCH --gres=gpu:1 # number of gpus


# Load libraries to run the code
SRC_DIR=/flush5/fon022/NeuroI_DL_Surf/
cd ${SRC_DIR}
source /apps/miniconda3/4.3.13/etc/profile.d/conda.sh
conda deactivate
source ${SRC_DIR}/bracewell/setup.sh

python predict.py outputs.output_dir=${OUT_DIR} user_config=${CONFIG}

# example:
# sbatch --export=OUT_DIR=<path to output directory>,CONFIG="<path_to_overriding_config_file>" /flush5/fon022/CorticalFlow/bracewell/slurm/predict_job.q