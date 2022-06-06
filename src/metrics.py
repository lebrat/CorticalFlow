import numpy as np
import argparse, sys
import torch
from pytorch3d.ops import sample_points_from_meshes, knn_points, knn_gather
import trimesh as tri
from pytorch3d.structures import Meshes
import warnings
try:
    import pymeshlab as pyml
except ModuleNotFoundError:
    warnings.warn("Self intersecting faces will not be computed. Please install PyMeshLab")

def compute_metrics(gt_pcl, gt_normals, pred_pcl, pred_normals):
    # transform in batches
    gt_pcl = gt_pcl.unsqueeze(0) if gt_pcl.ndim == 2 else gt_pcl
    gt_normals = gt_normals.unsqueeze(
        0) if gt_normals.ndim == 2 else gt_normals
    pred_pcl = pred_pcl.unsqueeze(0) if pred_pcl.ndim == 2 else pred_pcl
    pred_normals = pred_normals.unsqueeze(
        0) if pred_normals.ndim == 2 else pred_normals

    # compute metrics
    metrics = {}
    # For each predicted point, find its neareast-neighbor GT point
    knn_p2g, knn_g2p = knn_points(pred_pcl, gt_pcl, K=1), knn_points(gt_pcl, pred_pcl, K=1)
    knn_p2g_sq_dists, knn_g2p_sq_dists = knn_p2g.dists.squeeze(dim=-1), knn_g2p.dists.squeeze(dim=-1)
    knn_p2g_dists, knn_g2p_dists = knn_p2g_sq_dists.sqrt(), knn_g2p_sq_dists.sqrt()
    metrics['CHAMFER'] = (knn_p2g_sq_dists.mean(dim=-1) + knn_g2p_sq_dists.mean(dim=-1)).cpu().numpy()
    metrics['HAUSDORFF'] = torch.stack([knn_p2g_dists.max(dim=-1).values, knn_g2p_dists.max(dim=-1).values], dim=1).max(dim=-1).values.cpu().numpy()    
    metrics['HAUSDORFF_90'] = torch.stack([torch.quantile(knn_p2g_dists, 0.9, dim=-1), torch.quantile(knn_g2p_dists, 0.9, dim=-1)], dim=1).max(dim=-1).values.cpu().numpy()
    gt_noramls_near_pred = knn_gather(gt_normals, knn_p2g.idx)[..., 0, :]
    pred_to_gt_dot = (pred_normals * gt_noramls_near_pred).sum(dim=-1).abs().mean(dim=-1)
    pred_normals_near_gt = knn_gather(pred_normals, knn_g2p.idx)[..., 0, :]
    gt_to_pred_dot = (gt_normals * pred_normals_near_gt).sum(dim=-1).abs().mean(dim=-1)
    metrics['NORMAL_CONSISTENCY'] = (0.5 * (pred_to_gt_dot + gt_to_pred_dot)).cpu().numpy()

    return metrics


def SI(pt3d_meshes):
    if 'pymeshlab' not in sys.modules:
        return np.ones(len(pt3d_meshes)) * np.nan
    else:
        verts_list, faces_list = pt3d_meshes.verts_list(), pt3d_meshes.faces_list()
        fracSI_array = np.zeros(len(verts_list))
        
        for i, (vertices, faces) in enumerate(zip(verts_list, faces_list)):        
            ms = pyml.MeshSet()
            ms.add_mesh(pyml.Mesh(vertices.cpu().numpy(), faces.cpu().numpy()))    
            faces = ms.compute_topological_measures()['faces_number']
            ms.select_self_intersecting_faces()
            ms.delete_selected_faces()
            nnSI_faces = ms.compute_topological_measures()['faces_number']
            SI_faces = faces-nnSI_faces
            fracSI = (SI_faces/faces)*100
            fracSI_array[i] = fracSI

        return fracSI_array
