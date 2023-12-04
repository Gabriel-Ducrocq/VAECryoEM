import matplotlib.pyplot as plt
import yaml
import torch
import numpy as np
import Bio.PDB as bpdb
from Bio.PDB import PDBIO
from Bio.PDB.PDBParser import PDBParser
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from protein.main import rotate_pdb_structure_matrix, rotate_residues


class ResSelect(bpdb.Select):
    def accept_residue(self, res):
        if res.get_resname() == "LBV":
            return False
        else:
            return True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Renderer():
    def __init__(self, pixels_x, pixels_y, N_atoms=4530, period= 300/128, std = 1, defocus= 5000, spherical_aberration=21,
                 accelerating_voltage=300 , amplitude_contrast_ratio = 0.06, device="cpu", use_ctf=True,
                 latent_type="continuous", latent_dim = 10):
        self.std_blob = std
        self.len_x = pixels_x.shape[1]
        self.len_y = pixels_y.shape[1]
        assert self.len_x == self.len_y, "Number of pixels different on x and y"
        assert self.len_x % 2 == 0, "Number of pixel is not a multiple of 2"
        self.pixels_x = torch.tensor(pixels_x, dtype=torch.float32, device=device)
        self.pixels_y = torch.tensor(pixels_y, dtype=torch.float32, device=device)
        self.grid_x = self.pixels_x.repeat(self.len_y, 1)
        self.grid_y = torch.transpose(self.pixels_y, dim0=-2, dim1=-1).repeat(1, self.len_x)
        self.grid = torch.concat([self.grid_x[:, :, None], self.grid_y[:, :, None]], dim=2)
        self.N_atoms = N_atoms
        self.torch_sqrt_2pi= torch.sqrt(torch.tensor(2*np.pi, device=device))
        self.defocus = defocus
        self.spherical_aberration = spherical_aberration
        self.accelerating_voltage = accelerating_voltage # see the paper cited by cryoSparc site on CTF.
        self.amplitude_contrast_ratio = amplitude_contrast_ratio
        self.grid_period = period
        self.device = device
        self.use_ctf = use_ctf
        self.frequencies = torch.tensor([k/(eval(period)*self.len_x) for k in range(-int(self.len_x/2), int(self.len_x/2))],
                                        device=device)

        freqs = self.frequencies[:, None]**2 + self.frequencies[None, -int(self.len_y/2 + 1):]**2
        self.ctf_grid = self.compute_ctf_np(freqs, accelerating_voltage, spherical_aberration, amplitude_contrast_ratio,
                                            defocus)

        self.latent_type = latent_type
        self.latent_dim = latent_dim

    def compute_ctf_np(self,
            freqs: np.ndarray,
            volt: float,
            cs: float,
            w: float,
            df: float,
            phase_shift: float = 0,
            bfactor = None,
    ) -> np.ndarray:
        """
        Compute the 2D CTF
        Input:
            freqs (np.ndarray) Nx2 array of 2D spatial frequencies
            dfu (float): DefocusU (Angstrom)
            dfv (float): DefocusV (Angstrom)
            dfang (float): DefocusAngle (degrees)
            volt (float): accelerating voltage (kV)
            cs (float): spherical aberration (Å)
            w (float): amplitude contrast ratio
            phase_shift (float): degrees
            bfactor (float): envelope fcn B-factor (Angstrom^2)
        """
        # convert units
        volt = volt * 1000
        phase_shift = phase_shift * np.pi / 180

        # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
        lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt ** 2)
        s2 = freqs
        gamma = (
                2 * np.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
                - phase_shift
        )

        ctf = torch.sqrt(torch.tensor(1 - w ** 2, device=self.device)) * torch.sin(torch.tensor(gamma, device=self.device))\
                         - w * torch.cos(torch.tensor(gamma, device=self.device))
        if bfactor is not None:
            ctf *= np.exp(-bfactor / 4 * s2)

        return ctf

    def compute_gaussian_kernel(self, x, pixels_pos):
        """
        Computes the values of the gaussian kernel for one axis only but all heavy atoms and samples in batch
        :param x: (N_batch, 1): the coordinate of all heavy atoms on one axis for all samples in batch.
        :return: (N_batch, N_atoms, N_pix)
        """
        batch_size = x.shape[0]
        if self.latent_type == "continuous":
            scaled_distances = -(1/2)*(torch.broadcast_to(pixels_pos, (batch_size, self.N_atoms, -1)) -
                                   x[:, :, None])**2/self.std_blob**2
        else:
            scaled_distances = -(1/2)*(torch.broadcast_to(pixels_pos, (batch_size, self.latent_dim, self.N_atoms, -1)) -
                                   x[:, :, :, None])**2/self.std_blob**2

        axis_val = torch.exp(scaled_distances)/self.torch_sqrt_2pi
        return axis_val

    def ctf_corrupting(self, image):
        """
        Corrupts the image with the CTF.
        :param image: torch tensor (N_batch, N_pixels_s, N_pixels_y), non corrupted image.
        :return:  torch tensor (N_batch, N_pixels_s, N_pixels_y), corrupted image
        """
        fourier_images = torch.fft.rfft2(image)
        corrupted_fourier = fourier_images*self.ctf_grid
        corrupted_images = torch.fft.irfft2(corrupted_fourier)
        return corrupted_images

    def compute_x_y_values_all_atoms(self, atom_positions, rotation_matrices, translation_vectors,
                                     latent_type="continuous"):
        """

        :param atom_position: (N_batch, N_atoms, 3)
        :param rotation_matrices: (N_batch, 3, 3)
        :param translation_vectors: (N_batch, 3)
        :return:
        """
        transposed_atom_positions = torch.transpose(atom_positions, dim0=-2, dim1=-1)
        if latent_type=="continuous":
            #Rotation pose
            rotated_transposed_atom_positions = torch.matmul(rotation_matrices, transposed_atom_positions)
            #Translation pose
            rotated_transposed_atom_positions += translation_vectors[:, :, None]
            rotated_atom_positions = torch.transpose(rotated_transposed_atom_positions, -2, -1)
            all_x = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 0], self.pixels_x)
            all_y = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 1], self.pixels_y)
            #prod = torch.einsum("bki,bkj->bkij", (all_x, all_y))
            #projected_densities = torch.sum(prod, dim=1)
            projected_densities = torch.einsum("bki,bkj->bij", (all_x, all_y))
        else:
            rotated_transposed_atom_positions = torch.matmul(rotation_matrices[:, None, :, :], transposed_atom_positions)
            rotated_atom_positions = torch.transpose(rotated_transposed_atom_positions, -2, -1)
            all_x = self.compute_gaussian_kernel(rotated_atom_positions[:, :, :, 0], self.pixels_x)
            all_y = self.compute_gaussian_kernel(rotated_atom_positions[:, :, :, 1], self.pixels_y)
            #prod = torch.einsum("blki,blkj->blkij", (all_x, all_y))
            #projected_densities = torch.sum(prod, dim=-3)
            projected_densities = torch.einsum("blki,blkj->blij", (all_x, all_y))

        if self.use_ctf:
            projected_densities = self.ctf_corrupting(projected_densities)

        return projected_densities

def read_pdb(path):
    """
    Reads a pdb file in a structure object of biopdb
    :param path: str, path to the pdb file.
    :return: structure object from BioPython
    """
    parser = PDBParser(PERMISSIVE=0)
    structure = parser.get_structure("A", path)
    return structure

def compute_center_of_mass(structure):
    """
    Computes the center of mass of a protein
    :param structure: PDB structure
    :return: np.array(1,3)
    """
    all_coords = np.concatenate([atom.coord[:, None] for atom in structure.get_atoms()], axis=1)
    center_mass = np.mean(all_coords, axis=1)
    return center_mass[None, :]

def center_protein(structure, center_vector):
    """
    Center the protein given ALL its atoms
    :param structure: pdb structure in BioPDB format
    :param center_vector: vector used for the centering of all structures
    :return: PDB structure in BioPDB format, centered structure
    """
    all_coords = np.concatenate([atom.coord[:, None] for atom in structure.get_atoms()], axis=1)
    for index, atom in enumerate(structure.get_atoms()):
        atom.set_coord(all_coords[:, index] - center_vector)

    return structure

def get_backbone(structure):
    N_residue = 0
    residues_indexes = []
    absolute_positions = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residues_indexes.append(N_residue)
                name = residue.get_resname()
                if name not in ["LBV", "NAG", "MAN", "DMS", "BMA"]:
                    x, y, z = get_atom_positions(residue, name)
                    absolute_positions.append(x)
                    absolute_positions.append(y)
                    absolute_positions.append(z)

                    N_residue += 1

    return np.vstack(absolute_positions)

def get_atom_positions(residue, name):
    x = residue["CA"].get_coord()
    y = residue["N"].get_coord()
    if name == "GLY":
        z = residue["C"].get_coord()
        return x,y,z

    z = residue["C"].get_coord()
    return x,y,z


base_structure = read_pdb("../data/dataset/Phy/base_structure.pdb")
centering_structure = read_pdb("../data/dataset/Phy/base_structure.pdb")
center_of_mass = compute_center_of_mass(centering_structure)
centered_based_structure = center_protein(base_structure, center_of_mass[0])
atom_positions = torch.tensor(get_backbone(centered_based_structure), dtype=torch.float32, device=device)


#azimuth = torch.tensor(np.arange(0, 360, 15), dtype=torch.float32)
#inclination = torch.tensor(np.arange(0, 195, 15), dtype=torch.float32) # I am going to 195 intead of 180 to include 180
#angles = torch.tensor(np.arange(0, 195, 15), dtype=torch.float32)
#angles *= np.pi/180

#z = torch.repeat_interleave(torch.cos(inclination), repeats=24, dim=0)
#y = torch.reshape(torch.einsum("b, l -> bl", torch.sin(inclination), torch.sin(azimuth)), (13*24, ))
#x = torch.reshape(torch.einsum("b, l -> bl", torch.sin(inclination), torch.cos(azimuth)), (13*24,))

#axis_rot = torch.concatenate([x[:, None], y[:, None], z[:, None]], dim=-1)
#axis_angle_rot = torch.einsum("bl, k-> bkl", axis_rot, angles)
#axis_angle_rot = torch.reshape(axis_angle_rot, (13*24*13, 3))

azimuth = torch.tensor(np.arange(0, 370, 10), dtype=torch.float32)
inclination = torch.tensor(np.arange(0, 190, 10), dtype=torch.float32) # I am going to 190 intead of 180 to include 180
angles = torch.tensor(np.arange(0, 190, 10), dtype=torch.float32)
angles *= np.pi/180

print("Computing the coordinates")
z = torch.repeat_interleave(torch.cos(inclination), repeats=37, dim=0)
y = torch.reshape(torch.einsum("b, l -> bl", torch.sin(inclination), torch.sin(azimuth)), (19*37, ))
x = torch.reshape(torch.einsum("b, l -> bl", torch.sin(inclination), torch.cos(azimuth)), (19*37,))

print("Computing the rotation matrices")
axis_rot = torch.concatenate([x[:, None], y[:, None], z[:, None]], dim=-1)
axis_angle_rot = torch.einsum("bl, k-> bkl", axis_rot, angles)
axis_angle_rot = torch.reshape(axis_angle_rot, (19*37*19, 3))
rotation_matrices = axis_angle_to_matrix(axis_angle_rot)
#rotated_structures = torch.einsum("bkl, il -> bik", rotation_matrices, atom_positions)

pixels_x = np.linspace(-70, 70,num=140).reshape(1, -1)
pixels_y = np.linspace(-70, 70,num=140).reshape(1, -1)

path_settings = "../data/dataset/Phy/parameters.yaml"
path_images_yaml = "../data/dataset/Phy/images.yaml"
with open(path_settings, "r") as file:
    experiment_settings = yaml.safe_load(file)

with open(path_images_yaml, "r") as file:
    image_settings = yaml.safe_load(file)

#### BE CAREFUL, NO CTF !!!
renderer = Renderer(pixels_x, pixels_y, N_atoms=experiment_settings["N_residues"] * 3,
                    period=image_settings["renderer"]["period"], std=1, defocus=image_settings["renderer"]["defocus"],
                    spherical_aberration=image_settings["renderer"]["spherical_aberration"],
                    accelerating_voltage=image_settings["renderer"]["accelerating_voltage"],
                    amplitude_contrast_ratio=image_settings["renderer"]["amplitude_contrast_ratio"],
                    device=device, use_ctf=True,
                    latent_type=experiment_settings["latent_type"], latent_dim=experiment_settings["latent_dimension"])

#for i in range(4056):
#all_images = []
#for i in range(0, 10):
#    print(i)
#    image = renderer.compute_x_y_values_all_atoms(atom_positions, rotation_matrices[i*405: (i+1)*405]
#                                                  , torch.zeros(1, 3))
#    all_images.append(image)

#image = renderer.compute_x_y_values_all_atoms(atom_positions, rotation_matrices[4050:4056]
#                                                  , torch.zeros(6, 3))


#for i in range(4056):
all_images = []
for i in range(0, 100):
    print(i)
    image = renderer.compute_x_y_values_all_atoms(atom_positions, rotation_matrices[i*133: (i+1)*133]
                                                  , torch.zeros(1, 3))
    all_images.append(image)

image = renderer.compute_x_y_values_all_atoms(atom_positions, rotation_matrices[13300:13357]
                                                  , torch.zeros(57, 3))


all_images.append(image)
all_images = torch.concat(all_images, dim=0)
plt.imshow(all_images[0].detach().numpy())
plt.show()
torch.save(all_images, "../data/dataset/Phy/ImagePoseEstim")
#true_images = torch.load("../data/dataset/Phy/ImageDataSetNoNoiseNoCTF")
true_images = torch.load("../data/dataset/Phy/ImageDataSet")
true_poses = torch.load("../data/dataset/Phy/poses")

#all_images = torch.load("../data/dataset/Phy/ImagePoseEstim")
##ture_im_index = 1010 works quite well, even on noisy data :)
##true_im_index = 9501 works also quite well, in spite of the conformational change in the picture
##Same for true_index = 9900
## true_im_index = 1 does not work well at all, even though the images look quite similar... It is because it is hard to
## distinguish between two structures rotated by pi radians w.r.t its axis of rotation
## Same for 51 !
true_im_index = 9902
msds = torch.sum((all_images - true_images[true_im_index:true_im_index+ 1][None, :, :])**2, dim=(-1, -2))
print(msds)
pose_index = torch.argmin(msds)
K = 20
all_indices = torch.topk(msds, k=K, largest=False)
print("all indices:", all_indices)
print("Top:", pose_index)
pred_rot_mat = rotation_matrices[pose_index]
#torch.load("../data/dataset/Phy/ImageDataSetNoNoiseNoCTF")
print("PRED", pred_rot_mat)
print(true_poses[true_im_index])

test_true_pose = renderer.compute_x_y_values_all_atoms(atom_positions, true_poses[true_im_index][None, :, :],
                                                       torch.zeros(1, 3))


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
ax1.imshow(all_images[pose_index].detach().numpy())
ax2.imshow(all_images[9782].detach().numpy())
ax3.imshow(true_images[true_im_index].detach().numpy())
ax4.imshow(test_true_pose[0].detach().numpy())
plt.show()

print("Squared Frobenius norm", torch.sum((pred_rot_mat - true_poses[true_im_index])**2),
      torch.sum((rotation_matrices[9782] - true_poses[true_im_index])**2))

true_axis_angle = matrix_to_axis_angle(true_poses[true_im_index])
true_angle = torch.sqrt(torch.sum(true_axis_angle**2))
true_axis = true_axis_angle/true_angle


all_pred_dot = []
all_pred_angle = []
print(all_indices[-1])
for ind in all_indices[-1][0]:
    predicted_axis_angle = matrix_to_axis_angle(rotation_matrices[ind])
    predicted_angle = torch.sqrt(torch.sum(predicted_axis_angle**2))
    predicted_axis = predicted_axis_angle/predicted_angle
    dot = torch.sum(predicted_axis * true_axis)
    all_pred_dot.append(dot)
    all_pred_angle.append(predicted_angle)

#predicted_axis_angle_bis = matrix_to_axis_angle(rotation_matrices[9782])
#predicted_angle_bis = torch.sqrt(torch.sum(predicted_axis_angle_bis**2))
#predicted_axis_bis = predicted_axis_angle_bis/predicted_angle_bis

print("Axis", all_pred_dot)
print("Angle", all_pred_angle)
print("true angle", true_angle)
#print("Axis", predicted_axis, predicted_axis_bis, true_axis, torch.sum(predicted_axis*true_axis), torch.sum(predicted_axis_bis*true_axis))
#print("Angles", predicted_angle, predicted_angle_bis, true_angle)
print(pred_rot_mat.shape)
io = PDBIO()
io.set_structure(centered_based_structure)
io.save(f"../data/dataset/Phy/recovered_pose_{pose_index+1}.pdb", ResSelect())
centered_based_structure = read_pdb(f"../data/dataset/Phy/recovered_pose_{pose_index+1}.pdb")
pred_per_residue = torch.repeat_interleave(rotation_matrices[2112, None,:, :], 1006, dim=0)
rotate_residues(centered_based_structure, pred_per_residue, np.eye(3))
io = PDBIO()
io.set_structure(centered_based_structure)
io.save(f"../data/dataset/Phy/recovered_pose_{true_im_index+1}.pdb")












