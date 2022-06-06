#!/bin/bash

# inputs
INPUT_MRI_ID=${1}
INPUT_MRI_PATH=${2}
OUTPUT_DIR=${3}

echo "CORTICALFLOW++ RECON ALL:"
echo "MRI_ID=${1}"
echo "MRI_PATH=${2}"
echo "OUT_DIR=${3}"

# internal paths
TEMPLATE_PATH='/scratch2/fon022/CFPP/resources/MNI152_T1_1mm.nii.gz'
CFPP_PREDICT_LH='/scratch2/fon022/CFPP/resources/trained_models/CFPP_predict_lh.yaml'
CFPP_PREDICT_RH='/scratch2/fon022/CFPP/resources/trained_models/CFPP_predict_rh.yaml'

# create subdirectories
mkdir -p ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/
mkdir -p ${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/
mkdir -p ${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/
cp ${INPUT_MRI_PATH} ${OUTPUT_DIR}/${INPUT_MRI_ID}/

# fresurfer pre-processing
recon-all -autorecon1 -subjid ${INPUT_MRI_ID} -i ${INPUT_MRI_PATH} -sd ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/
mri_convert ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/${INPUT_MRI_ID}/mri/orig.mgz ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/${INPUT_MRI_ID}/mri/orig.nii.gz

# affine registration
reg_aladin -ref ${TEMPLATE_PATH} -flo ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/${INPUT_MRI_ID}/mri/orig.nii.gz -aff ${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/reg_affine.txt
reg_resample -ref ${TEMPLATE_PATH} -flo ${OUTPUT_DIR}/${INPUT_MRI_ID}/FS6/${INPUT_MRI_ID}/mri/orig.nii.gz -trans ${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/reg_affine.txt -res ${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/orig_affine.nii.gz -inter 3

# running prediction with CorticalFlow++ pretrained models
python predict.py user_config=${CFPP_PREDICT_LH} outputs.output_dir=${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/ inputs.data_type='list' inputs.path=${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/orig_affine.nii.gz inputs.split_name=${INPUT_MRI_ID} outputs.out_deform='[2]'
python predict.py user_config=${CFPP_PREDICT_RH} outputs.output_dir=${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/ inputs.data_type='list' inputs.path=${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/orig_affine.nii.gz inputs.split_name=${INPUT_MRI_ID} outputs.out_deform='[2]'

# mapping surfaces back to original space
ls -1 ${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/${INPUT_MRI_ID}/*.{pial,white} |  while read line; do echo -e "${OUTPUT_DIR}/${INPUT_MRI_ID}/niftyreg/reg_affine.txt\\t${line}"; done > ${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/${INPUT_MRI_ID}/affine_surf_file.txt
python scripts/apply_affine.py ${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/${INPUT_MRI_ID}/affine_surf_file.txt
cp ${OUTPUT_DIR}/${INPUT_MRI_ID}/CFPP/${INPUT_MRI_ID}/*_native.{pial,white} ${OUTPUT_DIR}/${INPUT_MRI_ID}/
