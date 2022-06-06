# root directory
CFPP_ROOT_DIR=/scratch2/fon022/CFPP/
# load bracewell modules
module load miniconda3/4.9.2
source /apps/miniconda3/4.9.2/etc/profile.d/conda.sh
conda deactivate
# load virtual enviroment
conda activate ${CFPP_ROOT_DIR}/CONDA_ENV/
export CONDA_PKGS_DIRS=${CFPP_ROOT_DIR}/CONDA_PKGS/
