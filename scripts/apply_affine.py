import argparse, os, sys, inspect 
import trimesh
import numpy as np
import nibabel as nib
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from src.utils import import_mesh, export_mesh, TicToc

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('surf_affine_file_list')
    args = parser.parse_args()
    
    timer = TicToc()
    with open(args.surf_affine_file_list) as file:
        timer.tic()
        for i, line in enumerate(file):            
            # read data            
            affine_path, surf_path = line.strip().split('\t')
            vertices, faces = import_mesh(surf_path.strip())
            affine_matrix = np.loadtxt(affine_path.strip())
            
            # apply affine to vertices
            affine_matrix = np.asanyarray(affine_matrix, dtype=np.float64)
            vertices = np.column_stack([np.asanyarray(vertices, dtype=np.float64), np.ones((vertices.shape[0], 1))]) 
            vertices = np.matmul(affine_matrix, vertices.T).T[:, :-1]
            vertices = np.ascontiguousarray(vertices)

            # save the resulting surface
            dirname = os.path.dirname(surf_path)
            surf_name, surf_ext = os.path.basename(surf_path).split('.')
            filename_no_ext = os.path.join(dirname, surf_name.strip()+'_native')
            export_mesh(vertices, faces, filename_no_ext, 'freesurfer', white_or_pial=surf_ext.strip())            

            if (i+1) % 100 == 0:                 
                print('{}/{} transformed in {:.4f} secs'.format(i, 13536, timer.toc()))