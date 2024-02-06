import torch
import einops



class Gaussian:
    mus: Union[torch.Tensor, np.ndarray]
    sigmas: Union[torch.Tensor, np.ndarray]
    amplitudes: Union[torch.Tensor, np.ndarray]


class Grid:
    coords: torch.Tensor  # (N, 1 or 2 or 3)
    shape: Tuple  # (side_shape, ) * 1 or 2 or 3



class BaseGrid(nn.Module):
    """Base grid.
    Range from origin (in Angstrom, default (0, 0, 0)), to origin + (side_shape - 1) * voxel_size, almost all data from
    RCSB or EMD follow this convention

    """

    def __init__(self, side_shape, voxel_size, origin=None):
        super().__init__()
        # NUmber of pixels on each side
        self.side_shape = side_shape
        #Size of the voxel
        self.voxel_size = voxel_size

        # Place of the origin
        if origin is None:
            origin = 0
        self.origin = origin

        # integer indices -> angstrom coordinates
        line_coords = torch.linspace(origin, (side_shape - 1) * voxel_size + origin, side_shape)
        self.register_buffer("line_coords", line_coords)

        [xx, yy] = torch.meshgrid([line_coords, line_coords], indexing="ij")
        plane_coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        self.register_buffer("plane_coords", plane_coords)
        self.plane_shape = (side_shape, ) * 2

        [xx, yy, zz] = torch.meshgrid([line_coords, line_coords, line_coords], indexing="ij")
        vol_coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        self.register_buffer("vol_coords", vol_coords)
        self.vol_shape = (side_shape, ) * 3

    def line(self):
        return Grid(coords=self.line_coords, shape=(self.side_shape, ))

    def plane(self):
        return Grid(coords=self.plane_coords, shape=self.plane_shape)

    def vol(self):
        return Grid(coords=self.vol_coords, shape=self.vol_shape)



class EMAN2Grid(BaseGrid):
    """EMAN2 style grid.
    origin set to -(side_shape // 2) * voxel_size

    """

    def __init__(self, side_shape, voxel_size):
        origin = -side_shape // 2 * voxel_size
        super().__init__(side_shape=side_shape, voxel_size=voxel_size, origin=origin)



# For code simplicity, following functions' input args must have a batch dim, notation:
# b: batch_size; nc: num_centers; np: num_pixels; nx, ny, nz: side_shape x, y, z
# gaussian rot with rot_mat then projection along z axis to plane defined by plane or line(x, y)
def batch_projection(gauss: Gaussian, rot_mats: torch.Tensor, line_grid: Grid) -> torch.Tensor:
    """A quick version of e2gmm projection.

    Parameters
    ----------
    gauss: (b/1, num_centers, 3) mus, (b/1, num_centers) sigmas and amplitudes
    rot_mats: (b, 3, 3)
    line_grid: (num_pixels, 3) coords, (nx, ) shape

    Returns
    -------
    proj: (b, y, x) projections
    """

    #Performs a rotation of the atom locations.
    centers = einops.einsum(rot_mats, gauss.mus, "b c31 c32, b nc c32 -> b nc c31")


    sigmas = einops.rearrange(gauss.sigmas, 'b nc -> b 1 nc')
    sigmas = 2 * sigmas**2

    #Computing the first marginal
    proj_x = einops.rearrange(line_grid.coords, "nx -> 1 nx 1") - einops.rearrange(centers[..., 0], "b nc -> b 1 nc")
    proj_x = torch.exp(-proj_x**2 / sigmas)

    #Computing the second marginal
    proj_y = einops.rearrange(line_grid.coords, "ny -> 1 ny 1") - einops.rearrange(centers[..., 1], "b nc -> b 1 nc")
    proj_y = torch.exp(-proj_y**2 / sigmas)

    #Computing the sum over the atom positions, multiplied by the amplitudes
    proj = einops.einsum(gauss.amplitudes, proj_x, proj_y, "b nc, b nx nc, b ny nc -> b nx ny")
    proj = einops.rearrange(proj, "b nx ny -> b ny nx")
    return proj




