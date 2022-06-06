from omegaconf import DictConfig, OmegaConf
import hydra, logging, os
import trimesh as tri
from pytorch3d.structures import Meshes
import torch
from pytorch3d.ops import sample_points_from_meshes
from src.metrics import compute_metrics, SI
from src.utils import import_mesh
import pandas as pd

# A logger for this file
logger = logging.getLogger(__name__)

def chunker(seq, size):
    for pos in range(0, len(seq), size):
        yield seq.iloc[pos:pos + size] 


@hydra.main(config_path="configs", config_name='eval')
def eval_app(cfg):

    logger.info('Evaluation script\nConfig:\n{}'.format(OmegaConf.to_yaml(cfg)))

    # read input csv file and processing interval
    logger.info("Reading input file from {}".format(cfg.in_file))
    in_file_df = pd.read_csv(cfg.in_file, usecols=['subject_id', 'surface_id', 'gt_mesh_path', 'pred_mesh_path'])
    if cfg.end_idx < 0: cfg.end_idx = len(in_file_df)
    in_file_df = in_file_df[cfg.start_idx:cfg.end_idx]
    logger.info("performing evaluation on {} rows starting at ({}, {}) and finishing at ({}, {})".format(in_file_df.shape[0], cfg.start_idx, in_file_df.iloc[0].subject_id, cfg.end_idx, in_file_df.iloc[-1].subject_id))
    
    with torch.no_grad(): 
        # iterate over chunks according to the batch size
        acc_results_df = pd.DataFrame()
        for b_num, batch_df in enumerate(chunker(in_file_df, cfg.batch_size)):
            batch_df = batch_df.copy()
            # load meshes as [gt1,pred1,gt2,pred2,..,gtn,predn]
            vertices, faces = [], []
            for row, (gt_mesh_path, pred_mesh_path) in batch_df[['gt_mesh_path', 'pred_mesh_path']].iterrows():
                logger.info("Loading predicted and ground-truth meshes: {} x {}".format(pred_mesh_path, gt_mesh_path))
                for mesh_path in [gt_mesh_path, pred_mesh_path]:                            
                    mesh_verts, mesh_faces = import_mesh(mesh_path)
                    vertices.append(torch.from_numpy(mesh_verts).float().to(torch.device("cuda"))); faces.append(torch.from_numpy(mesh_faces).long().to(torch.device("cuda")));
            meshes = Meshes(verts=vertices, faces=faces)

            # sample point clouds
            logger.info("sampling point clouds")
            pcl_points, pcl_normals = sample_points_from_meshes(meshes, num_samples=cfg.num_sampled_points, return_normals=True)
            gt_points, gt_normals = pcl_points[0::2], pcl_normals[0::2]
            pred_points, pred_normals = pcl_points[1::2], pcl_normals[1::2]

            # compute metrics
            logger.info("Computing metrics")
            metrics = compute_metrics(gt_points, gt_normals, pred_points, pred_normals)               
            metrics['SI'] = SI(meshes[1::2])
            logger.info("Metrics:\n{}".format('\n'.join(["{}: {}".format(k, metrics[k]) for k in metrics.keys()])))

            # accumulate results            
            for key in metrics.keys(): batch_df[key] = metrics[key]                        
            acc_results_df = pd.concat([acc_results_df, batch_df], ignore_index=True)

            logger.info("{}/{} rows has been processed".format((b_num + 1)*cfg.batch_size, in_file_df.shape[0]))        


    # save results to disc
    if not os.path.exists(cfg.out_dir): os.makedirs(cfg.out_dir, exist_ok=True)
    output_csv_file = os.path.join(cfg.out_dir, "eval_{:05d}_{:05d}_part.csv".format(cfg.start_idx, cfg.end_idx))
    acc_results_df.to_csv(output_csv_file, index=False)
    logger.info("Metrics computation done with success and results saved to {}.".format(output_csv_file))


if __name__ == "__main__":
    eval_app()