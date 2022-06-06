from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
import torch
import torch.nn.functional as tnnf
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from src.utils import TicToc, DatasetWrapper, export_mesh
from src.data import NormalizeMRIVoxels, InvertAffine, collate_CSRData_fn, csr_dataset_factory
from src.models import CorticalFlow, load_checkpoint
import pandas as pd
import numpy as np
import nibabel as nib
import trimesh


# A logger for this file
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name='predict')
def predict_app(cfg):
    
    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Predicting surfaces with Cortical Flow\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # timer
    timer = TicToc(); timer_dict = {}; timer.tic('Total')

    # read MRI
    timer.tic('ReadData')
    field_transforms = {'mri': Compose([NormalizeMRIVoxels('mean_std'), InvertAffine('mri_affine')])}         
    
    test_dataset = None
    if cfg.inputs.data_type == 'formatted':
        logger.info('loading from formatted dataset...')
        test_dataset = csr_dataset_factory('formatted', cfg.inputs.hemisphere, field_transforms, 
            dataset_path=cfg.inputs.path, split_file=cfg.inputs.split_file, split_name=cfg.inputs.split_name, surface_name=None)

    elif cfg.inputs.data_type == 'file':
        logger.info('loading from file...')
        test_dataset = csr_dataset_factory('file', cfg.inputs.hemisphere, field_transforms, input_file=cfg.inputs.path)

    elif cfg.inputs.data_type == 'list':
        logger.info('loading from list...')
        test_dataset = csr_dataset_factory('list', cfg.inputs.hemisphere, field_transforms, subjects=[cfg.inputs.split_name], mris=[cfg.inputs.path], surfaces=None)

    else:
        raise ValueError("Data format is not supported")
    test_dataloader = DataLoader(test_dataset, batch_size=6, collate_fn=collate_CSRData_fn, shuffle=False)   
    timer_dict['ReadData'] = timer.toc('ReadData')
    logger.info("{} subjects loaded for test in {:.4f} secs".format(len(test_dataset), timer_dict['ReadData']))
    
    # load template mesh
    template_mesh = trimesh.load(cfg.inputs.template)
    logger.info("Template mesh {} read from {}".format(template_mesh, cfg.inputs.template))

    # setup white model
    white_model = None
    if cfg.white_model is not None:
        timer.tic('WhiteModelSetup')
        white_model = CorticalFlow(cfg.white_model.share_flows, cfg.white_model.nb_features, cfg.white_model.integration_method, cfg.white_model.integration_steps).to(cfg.inputs.device)
        white_model.eval()    
        timer_dict['WhiteModelSetup'] = timer.toc('WhiteModelSetup')
        logger.info("{:.4f} secs for white model setup:\n{}".format(timer_dict['WhiteModelSetup'], white_model))        
        model_num_params = sum(p.numel() for p in white_model.parameters())    
        logger.info('White model Total number of parameters: {}'.format(model_num_params))
    
        # load white model weights    
        timer.tic('WhiteModelLoadWeights')
        best_df, best_ite, best_val_loss = load_checkpoint(cfg.white_model.model_checkpoint, model=white_model)
        assert len(white_model.deform_blocks) == best_df + 1, "White Model seem not have trained all deformation blocks" 
        white_model_use_deforms = list(range(len(white_model.deform_blocks)))
        timer_dict['WhiteModelLoadWeights'] = timer.toc('WhiteModelLoadWeights')
        logger.info("White Model weights at deformation train {} iteration {} and validation metric {:.4f} loaded from {} in {:.4f} secs".format(
            best_df, best_ite, best_val_loss, cfg.white_model.model_checkpoint, timer_dict['WhiteModelLoadWeights']))
    else:
        logger.info("White model is not specified.")

    # setup pial model 
    pial_model = None
    if cfg.pial_model is not None:
        timer.tic('PialModelSetup')
        pial_model = CorticalFlow(cfg.pial_model.share_flows, cfg.pial_model.nb_features, cfg.pial_model.integration_method, cfg.pial_model.integration_steps).to(cfg.inputs.device)
        pial_model.eval()    
        timer_dict['PialModelSetup'] = timer.toc('PialModelSetup')
        logger.info("{:.4f} secs for white model setup:\n{}".format(timer_dict['PialModelSetup'], pial_model))        
        model_num_params = sum(p.numel() for p in pial_model.parameters())    
        logger.info('Pial model Total number of parameters: {}'.format(model_num_params))
    
        # load pial model weights    
        timer.tic('PialModelLoadWeights')
        best_df, best_ite, best_val_loss = load_checkpoint(cfg.pial_model.model_checkpoint, model=pial_model)
        assert len(pial_model.deform_blocks) == best_df + 1, "Pial Model seem not have trained all deformation blocks" 
        pial_model_use_deforms = list(range(len(pial_model.deform_blocks)))
        timer_dict['PialModelLoadWeights'] = timer.toc('PialModelLoadWeights')
        logger.info("Pial Model weights at deformation train {} iteration {} and validation metric {:.4f} loaded from {} in {:.4f} secs".format(
            best_df, best_ite, best_val_loss, cfg.pial_model.model_checkpoint, timer_dict['PialModelLoadWeights']))
    else:
        logger.info("Pial model is not specified.")

    assert white_model is not None or pial_model is not None, "At least on of white or pial needs to be configured"

    # network forward  and save results   
    logger.info("predicting...")
    os.makedirs(cfg.outputs.output_dir, exist_ok=True)
    count = 0
    with torch.no_grad():                
        for ite, data in enumerate(test_dataloader):
                                    
            # read batch data               
            mri_vox = data.get('mri_vox').to(cfg.inputs.device)            
            mri_affine = data.get('mri_affine').to(cfg.inputs.device)             
            subject_ids = data.get('subject')              

            # reinstate template for safety
            template_verts = torch.from_numpy(template_mesh.vertices).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).float().to(cfg.inputs.device) 
            template_faces  = torch.from_numpy(template_mesh.faces).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).int().to(cfg.inputs.device) 

            # network prediction
            if white_model is not None:
                _, _, _, pred_verts_white = white_model(mri_vox, mri_affine, template_verts, white_model_use_deforms); 
            
            if pial_model is not None:
                _, _, _, pred_verts_pial = pial_model(mri_vox, mri_affine, 
                    pred_verts_white[-1] if white_model is not None else template_verts, pial_model_use_deforms)            

            # save outputs                    
            for surface, pred_verts in [('white', pred_verts_white if white_model is not None else None), ('pial', pred_verts_pial if pial_model is not None else None)]:
                if pred_verts is not None:
                    pred_verts = [verts_batches.cpu().numpy() for verts_batches in pred_verts]
                    for d in range(len(pred_verts)):
                        if d in cfg.outputs.out_deform:
                            for b in range(pred_verts[d].shape[0]):
                                subject_dir = os.path.join(cfg.outputs.output_dir, subject_ids[b])
                                os.makedirs(subject_dir, exist_ok=True)
                                mesh_output_filename_noext = os.path.join(subject_dir, "{}_{}_{}_Df{}".format(subject_ids[b], cfg.inputs.hemisphere, surface, d))                                
                                export_mesh(pred_verts[d][b], template_mesh.faces, mesh_output_filename_noext, cfg.outputs.out_format, surface)                                

            # log progress
            count += len(subject_ids)
            if ite % 10 == 0:
                logger.info("{}/{} subjects processed.".format(count, len(test_dataset)))

    logger.info("DONE !!!".format(count, len(test_dataset)))            
            

if __name__ == "__main__":
    predict_app()
