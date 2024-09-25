import os
import mrcfile
import argparse
import numpy as np
import os.path as osp


def save_mrc(vol,
             path,
             voxel_size: Union[int, float, Tuple, np.recarray] = None,
             origin: Union[int, float, Tuple, np.recarray] = None):
    """
    Save volumetric data to mrc file, set voxel_size, origin.
    See Also: https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.mrcobject.MrcObject.voxel_size
    Args:
        vol: density volume
        path: save path
        voxel_size: a single number, a 3-tuple (x, y ,z) or a modified version of the voxel_size array, default 1.
        origin: a single number, a 3-tuple (x, y ,z) or a modified version of the origin array, default 0.

    """
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(vol)

        if voxel_size is not None:
            m.voxel_size = voxel_size

        if origin is not None:
            m.header.origin = origin


def _get_file_name(file_path):
    return osp.splitext(osp.basename(file_path))[0]


def center_origin(mrc_file_path, mrc_file_centered):
    """
    Centers the origin of PDB and MRC file

    This function moves the origin of coordinates for both PDB and MRC files to the
    center of the MRC three-dimensional data matrix, so that the center of the 3D
    data matrix becomes (0,0,0). It then saves the adjusted files in the current
    directory with a '_centered' suffix.

    Usage:
    center_origin <reference_structure_path.pdb> <consensus_map_path.mrc>

    Args:
    reference_structure_path (str): The path to the input PDB file.
    consensus_map_path (str): The path to the input MRC file.
    """
    with mrcfile.open(mrc_file_path) as m:
        if m.voxel_size.x == m.voxel_size.y == m.voxel_size.z and np.all(np.asarray(m.data.shape) == m.data.shape[0]):
            new_origin = (- m.data.shape[0] // 2 * m.voxel_size.x, ) * 3
        else:
            print("The voxel sizes or shapes differ across the three axes in the three-dimensional data.")
            new_origin = (- m.data.shape[2] // 2 * m.voxel_size.x, - m.data.shape[1] // 2 * m.voxel_size.y,
                          - m.data.shape[0] // 2 * m.voxel_size.z)
        save_mrc(m.data.copy(), mrc_file_centered + _get_file_name(mrc_file_path) + "_centered.mrc",
                 m.voxel_size, new_origin)
        print(f"Result centered MRC saved to {_get_file_name(mrc_file_path)}_centered.mrc.")




parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--volumes_path', type=str, required=True)
parser_arg.add_argument('--centered_volumes_path', type=str, required=True)

args = parser_arg.parse_args()
volumes_path = args.volumes_path
centered_volumes_path= args.centered_volumes_path

all_volumes = [volumes_path + f for f in os.listdir(volumes_path) if "mrc" in f]

for i, vol_path in enumerate(all_volumes):
    center_origin(vol_path, centered_volumes_path)



