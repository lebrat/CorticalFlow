# CorticalFlow - Deep Learning Model For Cortical Surface Reconstruction From Magnetic Resonance Image

<a href="http://www.youtube.com/watch?feature=player_embedded&v=zQoMHwTHK2k" target="_blank">
 <img src="http://img.youtube.com/vi/zQoMHwTHK2k/mqdefault.jpg" alt="Watch the video" width="560" height="315" border="10" />
</a>

This repository contains the official implementation of CorticalFlow (CF), introduced in ***"CorticalFlow: A Diffeomorphic Mesh Transformer Network for Cortical Surface Reconstruction"***, and its improved version CorticalFlow++ (CFPP), introduced in ***"CorticalFlow++: Boosting Cortical Surface Reconstruction Accuracy, Regularity, and Interoperability"***. 

If you find our code or papers useful, please cite

```
@article{lebrat2021corticalflow,
  title     = {CorticalFlow: A Diffeomorphic Mesh Transformer Network for Cortical Surface Reconstruction},
  author    = {Lebrat, Leo and Santa Cruz, Rodrigo and de Gournay, Frederic and Fu, Darren and Bourgeat, Pierrick and Fripp, Jurgen and Fookes, Clinton and Salvado, Olivier},
  journal   = {Advances in Neural Information Processing Systems},
  volume    = {34},
  year      = {2021}
}

@article{santacruz2022cfpp,
  title     = {CorticalFlow++: Boosting Cortical Surface Reconstruction Accuracy, Regularity, and Interoperability},
  author    = {Santa Cruz, Rodrigo and Lebrat, Leo and Fu, Darren and Bourgeat, Pierrick and Fripp, Jurgen and Fookes, Clinton and Salvado, Olivier},
  journal   = {Arxiv},
  year      = {2022}
}
```

For further information, please check our [project page](https://lebrat.github.io/CorticalFlow/) or email rodrigo.santacruz@csiro.au and leo.lebrat@csiro.au.
See detailed usage instructions below:


## Installation
This software was developed using a High Computing Platform with SUSE Linux Enterprise 12, Nvidia P100 GPUs, Python 3.8, and CUDA 11.1.1. To manage python dependencies, we use Miniconda3 v4.9.2. To set our environment up, you can use the requirements file (requirements.txt) or perform the sequence of steps below, 

1. Create a python environment using conda:
```
# set root directory path
CFPP_ROOT_DIR=<ROOT_DIR>
cd CFPP_ROOT_DIR
# create conda enviroment
mkdir ./CONDA_PKGS/
export CONDA_PKGS_DIRS=${CFPP_ROOT_DIR}/CONDA_PKGS/
conda create python=3.8 --prefix ./CONDA_ENV/
conda activate ./CONDA_ENV/
```

2. Install [PyTorch 1.9.0](https://pytorch.org/) and [PyTorch3D v0.5.0](https://github.com/facebookresearch/pytorch3d):
```
conda install pytorch=1.9.0 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install -c pytorch3d pytorch3d
```

3. Install other python packages:
```
pip install hydra-core --upgrade
pip install trimesh nibabel tensorboard pandas
```

## Data Preparation
To train and evaluate our models, we use the cortical surface reconstruction benchmark introduced by the [DeepCSR](https://bitbucket.csiro.au/projects/CRCPMAX/repos/deepcsr/browse). This dataset is based on MR images extracted from the [Alzheimer’s Disease Neuroimaging Initiative (ADNI) study](http://adni.loni.usc.edu/) and their pseudo-ground-truth surfaces generated using [FreeSurfer V6](https://surfer.nmr.mgh.harvard.edu/) cross-sectional pipeline. We follow the pre-processing steps described in DeepCSR's repository. Note that it is not necessary to compute implicit surfaces with DeepCSR pre-processing code; thence, those steps can be skipped to reduce the data's processing time.

## Usage
This repository comprises python scripts parametrized by [Hydra configuration files](https://hydra.cc/docs/intro/). The main scripts are in the root directory, while the default configuration files for these scripts are in the ./configs/ folder.  **Our default configuration files are densely commented, explaining in detail each one of the script's parameters.** These parameters can be overwritten from the command line or by providing an external configuration file to the use_config parameter.

### Training Models
CFPP is trained in two steps: first, we train CFPP models for the white surfaces, and then we train CFPP models for the pial surfaces.
To train a CFPP model for a white surface run, 
```
python train.py  outputs.output_dir=<OUT_DIR> user_config=<CONFIG>
``` 
Once the white surface model is trained, we can train the CFPP model for the same hemisphere pial surface by running, 
```
python train_pial.py  outputs.output_dir=<OUT_DIR> user_config=<CONFIG>
``` 
where <OUT_DIR> is a given output directory for log metrics, snapshot weights, and store intermediate predictions. <CONFIG> is a .yaml file specifying the training hyper-parameters (e.g., number deformation blocks, surface, hemisphere). This file should be used to override the default train configuration in configs/train.yaml and configs/train_pial.yaml, respectively. See examples of configuration files to train CFPP for each cortical surface in the resources/config_samples/ folder.

To train CF models, one can only use the train.py script for all surfaces and set the integration_method to 'Euler' and templates to those in resources/neurips_templates.

### Generating Surfaces
To generate cortical surfaces with CF or CFPP, run
```
python predict.py outputs.output_dir=${OUT_DIR} user_config=${CONFIG}
```
with <OUT_DIR> the output directory where the generated surfaces will be saved. <CONFIG> is a .yaml specifying the path to CFPP's weights for the reconstruction of the white and pial surfaces between other options. As before, the default parameters are presented in configs/predict.yaml. 

We also provide **pre-trained models** and their prediction configuration files for CorticalFlow and CorticalFlow++. These files are in resources/trained_models folder.

### End-To-End Generation Script
Note that our prediction script expects a pre-processed MRI as input. To facilitate the use of CorticalFlow++ with raw MRI, we provide the recon_all.sh script that, given an input MRI, performs all pre-processing steps in addition to the cortical surface reconstruction. This script has the following extra requirements,

1. **NiftyReg v1.5.58** medical image registration toolbox. Please follow the instructions in [http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg_install).
    
2. **FreeSurfer V6** (for data preprocessing), please follow instructions in [https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads).](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads).

and can be executed as,
```
bash recon_all.sh <INPUT_MRI_ID> <INPUT_MRI_PATH> <OUTPUT_DIR>
```
where <INPUT_MRI_ID>, <INPUT_MRI_PATH>, and <OUTPUT_DIR> are the subject id, the path to the input MRI, and the path to the directory where the generated cortical surfaces will be saved, respectively. This script can be easily adapted for CorticalFlow version by changing the surface reconstruction steps.

### Evaluating Surfaces
To evaluate the reconstructed surfaces, we use the following metrics: chamfer distance, Hausdorff distance (90 perc. and maximum variants), chamfer normals, and percentage of self-intersecting faces. These metrics can be computed by running the following script,
```
python eval.py in_file=<IN_FILE> out_dir=<OUT_DIR>
```
with <IN_FILE> a CSV input file whose each row has the columns subject_id, surface_id, gt_mesh_path, and pred_mesh_path determining a surface to be evaluated. <OUT_DIR> is the output directory where a CSV file with the computed evaluation metrics will be saved. An example of the input CSV file is provided in resources/config_samples/sample_eval_list.csv, and the evaluation parameters can be configured in configs/eval.yaml. Note that for computing the percentage of self-intersecting faces metric, it is necessary to install the [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) python package.

## Acknowledgment
This research was supported by [Maxwell plus](https://maxwellplus.com/).
