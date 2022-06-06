#!/bin/bash

#SBATCH --time=00:30:00 # walltime
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

# Set the number of subjects (chunk size) that each SLURM task should do
SUBJECTS_PER_TASK=100

# Compute the starting and ending values for this task based
# on the SLURM task number and the chunk size.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $SUBJECTS_PER_TASK))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $SUBJECTS_PER_TASK ))

# Evaluate
python eval.py in_file=${IN_FILE} start_idx=${START_NUM} end_idx=${END_NUM} out_dir=${OUT_DIR}

# command example:
# (using SUBJECTS_PER_TASK=100 and file with 1148 rows)
# sbatch --array=[1-12] --export=IN_FILE=<>,OUT_DIR=<> /flush5/fon022/NeuroI_DL_Surf/bracewell/slurm/eval_job.q 

