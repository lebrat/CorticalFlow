import time
import torch
import numpy as np
import nibabel as nib
import trimesh as tri


def import_mesh(filename_with_ext):
    mesh_ext = filename_with_ext.split('.')[-1].lower().strip()
    assert mesh_ext in ['stl', 'obj', 'npz', 'white', 'pial'], "{} format is not supported".format(mesh_ext)

    vertices, faces = None, None
    if mesh_ext in ['stl', 'obj']:
        mesh = tri.load(filename_with_ext, process=False)
        vertices, faces = mesh.vertices, mesh.faces
    elif mesh_ext == 'npz':
        mesh = np.load(filename_with_ext)
        vertices, faces = mesh['vertices'], mesh['faces']
    else:
        vertices, faces = nib.freesurfer.io.read_geometry(filename_with_ext)        

    vertices = np.array(vertices).astype(np.float32)
    faces = np.array(faces).astype(np.int32)
    return vertices, faces


def export_mesh(vertices, faces, filename_no_ext, mesh_fmt, white_or_pial=None):    
    assert vertices.ndim == 2 and vertices.shape[-1] == 3  and faces.ndim == 2 and faces.shape[-1] == 3
    if mesh_fmt == 'stl':
        out_mesh = trimesh.Trimesh(vertices, faces, process=False)
        out_mesh.export(filename_no_ext + '.stl')         
    elif mesh_fmt == 'freesurfer':
        assert white_or_pial in ['white', 'pial'], "for freesurfer white_or_pial should be passed as white or pial"
        nib.freesurfer.io.write_geometry(filename_no_ext + '.' + white_or_pial, vertices, faces)
    elif mesh_fmt == 'npz':
        np.savez_compressed(filename_no_ext + '.npz', vertices=vertices, faces=faces)
    else:
        raise ValueError("{} is not supported. try slt, freesurfer, or npz".format(mesh_fmt))


class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, list_datum):
        super(DatasetWrapper, self).__init__()
        self.datums = list_datum

    def __len__(self):
        return len(self.datums)

    def __getitem__(self, idx):
        return self.datums[idx]


class TicToc:
    """
    TicToc class for time pieces of code.
    """

    def __init__(self):
        self._TIC_TIME = {}
        self._TOC_TIME = {}

    def tic(self, tag=None):
        """
        Timer start function
        :param tag: Label to save time
        :return: current time
        """
        if tag is None:
            tag = 'default'
        self._TIC_TIME[tag] = time.time()
        return self._TIC_TIME[tag]

    def toc(self, tag=None):
        """
        Timer ending function
        :param tag: Label to the saved time
        :param fmt: if True, formats time in H:M:S, if False just seconds.
        :return: elapsed time
        """
        if tag is None:
            tag = 'default'
        self._TOC_TIME[tag] = time.time()

        if tag in self._TIC_TIME:
            d = (self._TOC_TIME[tag] - self._TIC_TIME[tag])
            return d
        else:
            print("No tic() start time available for tag {}.".format(tag))

    # Timer as python context manager
    def __enter__(self):
        self.tic('CONTEXT')

    def __exit__(self, type, value, traceback):
        self.toc('CONTEXT')



def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''
    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p


def save_nib_image(path, voxel_grid, affine=np.eye(4), header=None):
    nib_img = nib.Nifti1Image(voxel_grid, affine, header)
    nib.save(nib_img, path)


def cycle(dl):
    while True:
        for x in dl: yield x