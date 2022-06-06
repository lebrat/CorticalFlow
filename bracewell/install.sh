# set root directory
CFPP_ROOT_DIR=/scratch2/fon022/CFPP/
cd CFPP_ROOT_DIR

# load modules
module load miniconda3/4.9.2
conda deactivate

# create conda enviroment
mkdir ./CONDA_PKGS/
export CONDA_PKGS_DIRS=${CFPP_ROOT_DIR}/CONDA_PKGS/
conda create python=3.8 --prefix ./CONDA_ENV/
conda activate ./CONDA_ENV/

# Pykeops and geomloss
module load gcc/9.3.0
module load cmake/3.20.2
module load cuda/11.1.1 
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
pip install pykeops
mkdir ${CFPP_ROOT_DIR}/PYKEOPS/
pip install git+https://github.com/jeanfeydy/geomloss.git@master#egg=geomloss[full]

# Install igl
conda install -c conda-forge igl

# Install Hydra, trimesh and nibabel
pip install hydra-core --upgrade
pip install trimesh nibabel vtk
pip install networkx

# install ants
pip install antspyx

# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d


