from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
import torch, torchvision, trimesh
import numpy as np
import nibabel as nib
from src.data import csr_dataset_factory, NormalizeMRIVoxels, InvertAffine, collate_CSRData_fn
from src.models import CorticalFlow, DiffeoMeshDeformer, save_checkpoint, load_checkpoint
from src.utils import TicToc, cycle
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes


# A logger for this file
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name='train')
def train_app(cfg):
    
    # override configuration with a user defined config file
    if cfg.user_config is not None:
        user_config = OmegaConf.load(cfg.user_config)
        cfg = OmegaConf.merge(cfg, user_config)
    logger.info('Training Cortical Flow\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # setting up dataset and data loader
    transforms = {
        'mri': torchvision.transforms.Compose([NormalizeMRIVoxels('mean_std'), InvertAffine('mri_affine')]),        
    } 
    train_dataset = csr_dataset_factory('formatted', cfg.dataset.surface_name.split('_')[0], transforms, dataset_path=cfg.dataset.path, split_file=cfg.dataset.split_file, split_name=cfg.dataset.train_split_name, surface_name=cfg.dataset.surface_name)    
    train_dataloader = cycle(torch.utils.data.DataLoader(train_dataset, batch_size=cfg.trainer.img_batch_size, collate_fn=collate_CSRData_fn, shuffle=True,  pin_memory=True, num_workers=2))
    logger.info("{} subjects loaded for training".format(len(train_dataset)))    
    val_dataset = csr_dataset_factory('formatted', cfg.dataset.surface_name.split('_')[0], transforms, dataset_path=cfg.dataset.path, split_file=cfg.dataset.split_file, split_name=cfg.dataset.val_split_name, surface_name=cfg.dataset.surface_name)    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.trainer.img_batch_size, collate_fn=collate_CSRData_fn, shuffle=False,  pin_memory=True, num_workers=2) 
    logger.info("{} subjects loaded for validation".format(len(val_dataset)))    

    # setting up model
    num_of_deformations = len(cfg.model.number_of_iterations)
    assert num_of_deformations == len(cfg.model.nb_features)
    assert num_of_deformations == len(cfg.model.templates)
    model = CorticalFlow(cfg.model.share_flows, cfg.model.nb_features, cfg.model.integration_method, cfg.model.integration_steps).to(cfg.trainer.device)
    logger.info("Model setup:\n{}".format(model))

    # setup tensorboard logs
    tb_logs_dir_path = os.path.join(cfg.outputs.output_dir, 'tb_logs')
    tb_logger = SummaryWriter(tb_logs_dir_path)
    logger.info("Tensorboard logs saved to {}".format(tb_logs_dir_path))    

    # resume train from training checkpoint
    if cfg.trainer.resume:
        logger.info("Resuming training from output directory {}".format(cfg.trainer.resume))        
        last_train_deform_idx, _, _ = load_checkpoint(cfg.trainer.resume)
        
    # deformation training loop
    for deform_train_idx  in range(last_train_deform_idx if cfg.trainer.resume else 0, num_of_deformations):        
        logger.info("Training deformation {}".format(deform_train_idx))

        # load template mesh
        template_mesh = trimesh.load(cfg.model.templates[deform_train_idx])        
        logger.info("Train deform {}: Template mesh {} read from {}".format(deform_train_idx, template_mesh, cfg.model.templates[deform_train_idx]))
        template_shift = template_mesh.vertices.mean(axis=0, keepdims=True)         
        template_scale = np.max(np.linalg.norm(template_mesh.vertices - template_shift, axis=1))        
        template_shift = torch.from_numpy(template_shift).float().to(cfg.trainer.device).view(1, 1, 3)            
        logger.info("Train deform {}: Shift={} and Scale={} for template mesh.".format(deform_train_idx, template_shift, template_scale))
        
        # load model weights for resuming or previous best model for training from scratch
        if cfg.trainer.resume and deform_train_idx == last_train_deform_idx:                        
            _, _, _ = load_checkpoint(cfg.trainer.resume, model=model)
            logger.info("loading model from {} due to resume training".format(cfg.trainer.resume))
        else:
            # load best model from previous        
            if deform_train_idx > 0:
                best_ckp_file = os.path.join(cfg.outputs.output_dir, 'best_model_DT{}.pth'.format(deform_train_idx-1))
                df_check, _, best_ch_prev = load_checkpoint(best_ckp_file, model=model)
                assert df_check == (deform_train_idx-1)
                logger.info("Best wights for deformation {} loaded from {}".format(df_check, best_ckp_file))

        # setting up optimizer (train one deformation per time)
        model_num_params, model_num_lr_params = 0, 0
        for param in model.parameters(): param.requires_grad= False; model_num_params += param.numel();        
        for param in model.deform_blocks[deform_train_idx].parameters(): param.requires_grad=True; model_num_lr_params += param.numel();
        logger.info('Train deform {}: number of learnable parameters: {}/{}'.format(deform_train_idx, model_num_lr_params, model_num_params))
        optimizer = getattr(torch.optim, cfg.optimizer.name)([param for param in model.parameters() if param.requires_grad], lr=cfg.optimizer.lr[deform_train_idx])
        logger.info("Train deform {}: Optimizer setup:\n{}".format(deform_train_idx, optimizer))
        
        # load optimizer state and constants if resuming or training from scratch
        if cfg.trainer.resume and deform_train_idx == last_train_deform_idx: 
            _, last_train_ite, best_val_loss = load_checkpoint(cfg.trainer.resume, optimizer=optimizer)
            logger.info("loading optimizer from {} due to resume training".format(cfg.trainer.resume))
        else:
            last_train_ite, best_val_loss = 1, np.finfo(np.float32).max

        # train and validation loop
        use_deforms, timer, train_loss_acc, train_ite_acc = list(range(deform_train_idx+1)), TicToc(), defaultdict(float), 0        
        logger.info('Starting train deformation {} from {} iterations with best validation loss of {}'.format(deform_train_idx, last_train_ite, best_val_loss))
                        
        ####################### deformation train/val loop ##########################        
        timer.tic('train_step')
        for ite in range(last_train_ite, cfg.model.number_of_iterations[deform_train_idx]+1):                                    

            #################### Train step ####################
            data = next(train_dataloader); model.train(); optimizer.zero_grad();        

            # read batch data                        
            mri_vox = data.get('mri_vox').to(cfg.trainer.device)            
            mri_affine = data.get('mri_affine').to(cfg.trainer.device) 
            gt_meshes = data.get('py3d_meshes').to(cfg.trainer.device)          

            # reinstate template for safety with memory pointers
            template_verts = torch.from_numpy(template_mesh.vertices).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).float().to(cfg.trainer.device) 
            template_faces  = torch.from_numpy(template_mesh.faces).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).int().to(cfg.trainer.device) 

            # network prediction  
            pred_flow_down, pred_flow_fields, pred_flow_fields_int, pred_verts = model(mri_vox, mri_affine, template_verts, use_deforms)
            
            # center/scaling gt/pred meshes and sampling point clouds
            gt_meshes = gt_meshes.offset_verts(-template_shift.view(1,3).expand_as(gt_meshes.verts_packed())).scale_verts(1.0/template_scale)            
            gt_pcl = sample_points_from_meshes(gt_meshes, num_samples=cfg.trainer.points_per_image)                
            pred_meshes = Meshes(verts=[vs for vs in pred_verts[-1]], faces=[fs for fs in template_faces])
            pred_meshes = pred_meshes.offset_verts(-template_shift.view(1,3).expand_as(pred_meshes.verts_packed())).scale_verts(1.0/template_scale)            
            pred_pcl = sample_points_from_meshes(pred_meshes, num_samples=cfg.trainer.points_per_image)               

            # loss, gradient, and back-propagation            
            train_ch_points, _ = chamfer_distance(pred_pcl,  gt_pcl)
            train_edge_loss = mesh_edge_loss(pred_meshes)                                        
            train_loss = cfg.objective.chamffer_weight[deform_train_idx] * train_ch_points  \
                        + cfg.objective.edge_loss_weight[deform_train_idx] * train_edge_loss                        
                        
            train_loss.backward()            
            optimizer.step()                      

            # accumulate loss for logging            
            loss_name_tensor_ite = [('chamffer_points', train_ch_points),('edge_loss', train_edge_loss), ('loss', train_loss)]
            for loss_name, loss_tensor in loss_name_tensor_ite:
                train_loss_acc[loss_name] += loss_tensor.item()            
                del loss_tensor
            train_ite_acc += 1

            # log train
            if ite % cfg.trainer.train_log_interval == 0: 
                avg_train_ite_time = timer.toc('train_step') / float(train_ite_acc)              
                for loss_name in train_loss_acc:                    
                    train_loss_acc[loss_name] = train_loss_acc[loss_name] / float(train_ite_acc) 
                    tb_logger.add_scalar('train/{}'.format(loss_name), train_loss_acc[loss_name], sum(cfg.model.number_of_iterations[:deform_train_idx]) + ite)
                logger.info("Training: deform={}, Ite={}, Losses={}, AvgIteTime={:.2f} secs".format(deform_train_idx, ite, train_loss_acc, avg_train_ite_time))                
                train_loss_acc, train_ite_acc  = defaultdict(float), 0
                timer.tic('train_step')
            #################### Train step ####################                  

            #################### Val loop ####################
            if ite % cfg.trainer.evaluate_interval == 0:
                with torch.no_grad():   
                    model.eval()
                    val_loss_acc = defaultdict(float); timer.tic('eval_step'); logger.info("Evaluating...");
                    for data in val_dataloader:                        

                        # read batch data               
                        mri_vox = data.get('mri_vox').to(cfg.trainer.device)            
                        mri_affine = data.get('mri_affine').to(cfg.trainer.device) 
                        gt_meshes = data.get('py3d_meshes').to(cfg.trainer.device)
                        subject_ids = data.get('subject')              

                        # reinstate template for safety
                        template_verts = torch.from_numpy(template_mesh.vertices).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).float().to(cfg.trainer.device) 
                        template_faces  = torch.from_numpy(template_mesh.faces).unsqueeze(0).repeat(mri_vox.shape[0], 1, 1).int().to(cfg.trainer.device) 

                        # network prediction         
                        pred_flow_down, pred_flow_fields, pred_flow_fields_int, pred_verts = model(mri_vox, mri_affine, template_verts, use_deforms)
                                                
                        # center/scaling gt/pred meshes and sampling point clouds
                        gt_meshes = gt_meshes.offset_verts(-template_shift.view(1,3).expand_as(gt_meshes.verts_packed())).scale_verts(1.0/template_scale)            
                        gt_pcl = sample_points_from_meshes(gt_meshes, num_samples=cfg.trainer.points_per_image)                
                        pred_meshes = Meshes(verts=[vs for vs in pred_verts[-1]], faces=[fs for fs in template_faces])
                        pred_meshes = pred_meshes.offset_verts(-template_shift.view(1,3).expand_as(pred_meshes.verts_packed())).scale_verts(1.0/template_scale)            
                        pred_pcl = sample_points_from_meshes(pred_meshes, num_samples=cfg.trainer.points_per_image)                        

                        # loss, gradient, and back-propagation            
                        val_ch_points, _ = chamfer_distance(pred_pcl,  gt_pcl)                        
                        val_edge_loss = mesh_edge_loss(pred_meshes)                                        
                        val_loss = cfg.objective.chamffer_weight[deform_train_idx] * val_ch_points  \
                                    + cfg.objective.edge_loss_weight[deform_train_idx] * val_edge_loss
                                                                                                                   
                        # accumulate validation loss for logging                        
                        loss_name_tensor_ite = [('chamffer_points', val_ch_points), ('edge_loss', val_edge_loss), ('loss', val_loss)]
                        for loss_name, loss_tensor in loss_name_tensor_ite:
                            val_loss_acc[loss_name] += loss_tensor.item()            
                            del loss_tensor                           
                        
                        if cfg.trainer.debug: break;

                    # average and log metrics                                            
                    for loss_name in val_loss_acc:                        
                        val_loss_acc[loss_name] = val_loss_acc[loss_name] / float(len(val_dataset))                                        
                        tb_logger.add_scalar('val/{}'.format(loss_name), val_loss_acc[loss_name], sum(cfg.model.number_of_iterations[:deform_train_idx]) + ite)
                    val_elapsed_time = timer.toc('eval_step')
                    logger.info("Evaluation: deform={}, Ite={}, Loss={}, EvalTime={:.2f} secs".format(deform_train_idx, ite, val_loss_acc, val_elapsed_time))                    

                    # if found the best val loss so far ->  checkpoint best
                    if val_loss_acc['chamffer_points'] <= best_val_loss:
                        best_val_loss = val_loss_acc['chamffer_points']
                        ckp_file = os.path.join(cfg.outputs.output_dir, 'best_model_DT{}.pth'.format(deform_train_idx))
                        save_checkpoint(deform_train_idx, ite, model, optimizer, None, best_val_loss, ckp_file)
                        logger.info("Best model found with chamffer_points={:.6f} !!! checkpoint to {}".format(best_val_loss, ckp_file))

                    # snapshot last batch  
                    flow_fields = [tnnf.upsample(flow_field, scale_factor=down, mode='trilinear') if down != 1 else flow_field for down, flow_field in zip(pred_flow_down, pred_flow_fields)]
                    flow_fields_int = [tnnf.upsample(flow_field_int, scale_factor=down, mode='trilinear') if down != 1 else flow_field_int for down, flow_field_int in zip(pred_flow_down, pred_flow_fields_int)]
                    flow_fields, flow_fields_int =  [flow_field.permute(0, 2, 3, 4, 1).cpu().numpy() for flow_field in flow_fields], [flow_field_int.permute(0, 2, 3, 4, 1).cpu().numpy() for flow_field_int in flow_fields_int]
                    mri_vox, mri_affine = mri_vox.cpu().numpy(), mri_affine.cpu().numpy()
                    gt_verts = [((verts * template_scale) + template_shift.squeeze(0)).cpu().numpy() for verts in gt_meshes.verts_list()]
                    gt_faces = [f.cpu().numpy() for f in gt_meshes.faces_list()]
                    pred_verts = [[verts.cpu().numpy() for verts in batch_verts] for batch_verts in pred_verts]                                            
                    vis_folder_path = os.path.join(cfg.outputs.output_dir, 'visualize', 'DT_{}'.format(deform_train_idx),'vis_ite{:06d}'.format(ite))
                    os.makedirs(vis_folder_path, exist_ok=True)
                    for i in range(mri_vox.shape[0]):
                        nib_affine = np.linalg.inv(mri_affine[i])
                        nib.save(nib.Nifti1Image(mri_vox[i], nib_affine), os.path.join(vis_folder_path, 'mri_{}.nii.gz'.format(subject_ids[i])))
                        for j in range(len(flow_fields)):
                            nib.save(nib.Nifti1Image(flow_fields[j][i], nib_affine), os.path.join(vis_folder_path, 'flow_D{}_{}.nii.gz'.format(j, subject_ids[i])))
                            nib.save(nib.Nifti1Image(flow_fields_int[j][i], nib_affine), os.path.join(vis_folder_path, 'flow_int_D{}_{}.nii.gz'.format(j, subject_ids[i])))
                            trimesh.Trimesh(pred_verts[j][i], template_mesh.faces, process=False).export(os.path.join(vis_folder_path, 'pred_mesh_D{}_{}.stl'.format(j, subject_ids[i])))                                                                            
                        trimesh.Trimesh(gt_verts[i], gt_faces[i], process=False).export(os.path.join(vis_folder_path, 'gt_mesh_{}.stl'.format(subject_ids[i])))                        
                    logger.info('visualization of predictions saved into {}'.format(vis_folder_path))
            #################### Val loop ####################

            ################## REGULAR CHECKPOINT STEP ########################
            if ite % cfg.trainer.checkpoint_interval == 0:
                checkpoints_dir_path = os.path.join(cfg.outputs.output_dir, 'checkpoints')
                os.makedirs(checkpoints_dir_path, exist_ok=True)
                ckp_file = os.path.join(checkpoints_dir_path, 'model_DT{}_ite{:06d}.pth'.format(deform_train_idx, ite))
                save_checkpoint(deform_train_idx, ite, model, optimizer, None, best_val_loss, ckp_file)
                logger.info("checkpoint saved to {}".format(ckp_file))
            ################## REGULAR CHECKPOINT STEP ########################

    logger.info("Training finished.")

if __name__ == "__main__":
    train_app()
