import torch
import numpy as np
from time import time
import matplotlib.pyplot as plt


class Lattice:
    def __init__(
        self, D: int, extent: float = 0.5, ignore_DC: bool = True, device=None
    ):
        assert D % 2 == 1, "Lattice size must be odd"
        x0, x1 = np.meshgrid(
            np.linspace(-extent, extent, D, endpoint=True),
            np.linspace(-extent, extent, D, endpoint=True),
        )
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(D**2)], 1).astype(
            np.float32
        )
        self.coords = torch.tensor(coords, device=device)
        self.extent = extent
        self.D = D
        self.D2 = int(D / 2)

        # todo: center should now just be 0,0; check Lattice.rotate...
        # c = 2/(D-1)*(D/2) -1
        # self.center = torch.tensor([c,c]) # pixel coordinate for img[D/2,D/2]
        self.center = torch.tensor([0.0, 0.0], device=device)

        self.square_mask = {}
        self.circle_mask = {}

        self.freqs2d = self.coords[:, 0:2] / extent / 2
        self.device = device


        self.x0 = np.linspace(-extent, extent, D, endpoint=True)
        self.x1 = np.linspace(-extent, extent, D, endpoint=True)


class RendererFourier():
        def __init__(self, D, std = 1, device= "cpu", sigma=1):
            self.D = D
            if D % 2 == 0:
                self.D += 1

            self.lattice = Lattice(self.D, device=device)
            self.sigma = sigma

            self.x0 = 2*torch.pi*self.lattice.x0 /self.lattice.extent / 2
            self.x1 = 2*torch.pi*self.lattice.x1 /self.lattice.extent / 2
            self.exp0 = self.compute_exponential(self.x0)
            self.exp1 = self.compute_exponential(self.x1)

        def compute_fourier(self, atom_positions, rotation_matrices):
            """
            :param: atom_positions, torch.tensor(N_batch, N_atoms, 3)
            :param rotation_matrices: torch.tensor(N_batch, 3, 3), right multiplication convention ! So transpose before feeding to the function !
            """
            N_batch = rotation_matrices.shape[0]
            #coord is of shape (N_batch, D**2, 3)
            #There is a 2pi coeff here for taking into account the different convention of Fourier transform I do and the one
            #used by the FFT in torch
            start = time()
            coords = 2*torch.pi*self.lattice.coords / self.lattice.extent / 2 @ rotation_matrices
            end = time()
            print("Time coords rotating", end -start)
            #dot_prod is (batch, D**2, atoms)
            start = time()
            dot_prod = torch.einsum("bkj, baj-> bka", coords, atom_positions)
            end = time()
            print("Time dot_prod", end -start)
            #norm_squared is (N_batch, D**2, 1)
            start = time()
            norm_squared = torch.sum(coords**2, dim=-1)[:, :, None]
            end = time()
            print("Time norm_squred", end -start)
            print("TEST SHAPES", dot_prod.shape, norm_squared.shape)
            start = time()
            test_real = torch.cos(dot_prod)
            test_imag = -torch.sin(dot_prod)
            #test_real_imag = torch.stack([test_real, test_imag], dim=-1)
            #test = -1j*dot_prod
            end = time()
            print("test time", end - start)
            start = time()
            fourier_per_coord_per_atom = torch.exp(-1j*dot_prod - 0.5*(self.sigma**2)*norm_squared)
            end = time()
            print("Time exponential", end - start)
            #Fourier_per_coord is (N_batch, D**2)
            fourier_per_coord = torch.sum(fourier_per_coord_per_atom, dim=-1)
            ##TRANSPOSING HERE BE CAREFUL !!!
            start = time()
            fourier_per_coord_2d = torch.transpose(torch.reshape(fourier_per_coord, (N_batch, self.D, self.D)), dim0=-2, dim1=-1)
            end = time()
            print("FOURIER time", end - start)
            # BE CAREFUL: I AM REMOVING THE LAST ROWS AND COLUMNS BECAUSE WE ALREADY HAVE THE NYQUIST FREQUENCY !
            fourier_per_coord_2d = fourier_per_coord_2d[: , :-1, :-1]
            images = torch.fft.ifft2(torch.fft.ifftshift(fourier_per_coord_2d, dim = (-2, -1)))
            images = torch.fft.fftshift(images, dim=(-2, -1))
            #images = torch.fft.ifft2(fourier_per_coord_2d)
            return images


        def compute_complex_exponential(self, atom_coordinates, freq_coordinates):
            """
            Computes the complex exponential appearing in the Fourier transform
            :param: atom_coordinates, torch.tensor(N_batch, N_atoms, 1)
            :param: freq_coordinates, torch.tensor(N_pix) or torch.tensor(N_pix+1) 
            """
            args = atom_coordinates[:, :, None]*freq_coordinates[None, None, :]
            real = torch.cos(args)
            imaginary = torch.sin(args)
            complex_output = torch.concat([real[:, :, :, None], imaginary[:, :, :, None]], dim=-1)
            return torch.view_as_complex(complex_output)
            #return torch.exp(-1j*atom_coordinates[:, :, None]*freq_coordinates[None, None, :])


        def compute_exponential(self, freq_coordinates):
            """
            Computes the regular exponential appearing in the Fourier transform
            :param: freq_coordinates, torch.tensor(N_pix) or torch.tensor(N_pix+1) 
            """
            freq_coordinates = torch.tensor(freq_coordinates, dtype=torch.float32)
            return torch.exp(-0.5*self.sigma**2*freq_coordinates**2)

        def compute_fourier_test(self, atom_positions, rotation_matrices):
            """
            :param: atom_positions, torch.tensor(N_batch, N_atoms, 3)
            :param rotation_matrices: torch.tensor(N_batch, 3, 3), right multiplication convention ! So transpose before feeding to the function !
            """
            N_batch = rotation_matrices.shape[0]
            #coord is of shape (N_batch, D**2, 3)
            #There is a 2pi coeff here for taking into account the different convention of Fourier transform I do and the one
            #used by the FFT in torch
            rotated_structures = atom_positions @ torch.transpose(rotation_matrices, dim0=-2, dim1=-1)
            #first_exp and second_exp are torch.tensor(N_batch. N_atom, Npix or Npix+1)
            first_exp = self.compute_complex_exponential(rotated_structures[:, :, 0], self.x0)*self.exp0
            second_exp = self.compute_complex_exponential(rotated_structures[:, :, 1], self.x1)*self.exp1
            start = time()
            fourier_images =torch.einsum("bak, bal->bkl", first_exp, second_exp)
            end = time()
            print("TIME INSIDE FOURIER", end - start)
            images = torch.fft.ifft2(torch.fft.ifftshift(fourier_images, dim = (-2, -1)))
            images = torch.fft.fftshift(images, dim=(-2, -1))
            return images












class Renderer():
    def __init__(self, pixels_x, pixels_y, N_atoms, dfU, dfV, dfang, spherical_aberration=21,
                 accelerating_voltage=300 , amplitude_contrast_ratio = 0.06, device="cpu", use_ctf=True,
                 latent_type="continuous", latent_dim = 10, std = 1):
        self.std_blob = std
        print("STD renderer", std)
        self.len_x = pixels_x.shape[1]
        self.len_y = pixels_y.shape[1]
        assert self.len_x == self.len_y, "Number of pixels different on x and y"
        assert self.len_x % 2 == 0, "Number of pixel is not a multiple of 2"
        self.pixels_x = torch.tensor(pixels_x, dtype=torch.float32, device=device)
        self.pixels_y = torch.tensor(pixels_y, dtype=torch.float32, device=device)
        self.pixels_z = torch.tensor(pixels_x, dtype=torch.float32, device=device)
        self.N_atoms = N_atoms
        self.torch_sqrt_2pi= torch.sqrt(torch.tensor(2*np.pi, device=device))
        self.dfU = torch.ones(1, device=device)*dfU
        self.dfV = torch.ones(1, device=device)*dfV
        self.dfang = torch.ones(1, device=device)*dfang
        self.spherical_aberration = torch.ones(1, device=device)*spherical_aberration
        self.accelerating_voltage = torch.ones(1, device=device)*accelerating_voltage # see the paper cited by cryoSparc site on CTF.
        self.amplitude_contrast_ratio = torch.ones(1, device=device)*amplitude_contrast_ratio
        self.pixel_span = (self.pixels_x[:, -1] - self.pixels_x[:, 0])/self.len_x
        self.device = device
        self.use_ctf = use_ctf
        self.latent_type = latent_type
        self.latent_dim = latent_dim

        freqs = (
            torch.stack(
                self.meshgrid_2d(-0.5, 0.5, self.len_x, endpoint=False),
                -1,
            )
            / self.pixel_span
        )
        freqs = freqs.reshape(-1, 2)
        freqs = freqs.to(device)
        ctf = self.compute_ctf(freqs, dfU, dfV, dfang, self.accelerating_voltage, self.spherical_aberration,
                               self.amplitude_contrast_ratio)
        self.ctf_grid = torch.reshape(ctf, (self.len_x, self.len_y))
        ## BE CAREFUL, the CTF potentially works only for an even number of pixels along one dimension !
        self.ctf_grid = torch.fft.ifftshift(self.ctf_grid)


    def meshgrid_2d(self, lo, hi, n, endpoint=False):
        """
        Torch-compatible implementation of:
        np.meshgrid(
                np.linspace(-0.5, 0.5, D, endpoint=endpoint),
                np.linspace(-0.5, 0.5, D, endpoint=endpoint),
            )
        Torch doesn't support the 'endpoint' argument (always assumed True)
        and the behavior of torch.meshgrid is different unless the 'indexing' argument is supplied.
        """
        if endpoint:
            values = torch.linspace(lo, hi, n, device=self.device)
        else:
            values = torch.linspace(lo, hi, n + 1, device=self.device)[:-1]

        return torch.meshgrid(values, values, indexing="xy")

    def compute_ctf(self,
            freqs: torch.Tensor,
            dfu: torch.Tensor,
            dfv: torch.Tensor,
            dfang: torch.Tensor,
            volt: torch.Tensor,
            cs: torch.Tensor,
            w: torch.Tensor,
            phase_shift = None,
            scalefactor = None,
            bfactor= None,
    ) -> torch.Tensor:
        """

        This code is based on the cryoDRGN code.
        Compute the 2D CTF

        Input:
            freqs: Nx2 array of 2D spatial frequencies
            dfu: DefocusU (Angstrom)
            dfv: DefocusV (Angstrom)
            dfang: DefocusAngle (degrees)
            volt: accelerating voltage (kV)
            cs: spherical aberration (mm)
            w: amplitude contrast ratio
            phase_shift: degrees
            scalefactor : scale factor
            bfactor: envelope fcn B-factor (Angstrom^2)
        """
        # convert units
        volt = volt * 1000
        cs = cs * 10 ** 7
        dfang = dfang * np.pi / 180
        if phase_shift is None:
            phase_shift = torch.tensor(0)
        phase_shift = phase_shift * np.pi / 180

        # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
        lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt ** 2)
        x = freqs[..., 0]
        y = freqs[..., 1]
        ang = torch.arctan2(y, x)
        s2 = x ** 2 + y ** 2
        df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
        gamma = (
                2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
                - phase_shift
        )
        ctf = torch.sqrt(1 - w ** 2) * torch.sin(gamma) - w * torch.cos(gamma)
        if scalefactor is not None:
            ctf *= scalefactor
        if bfactor is not None:
            ctf *= torch.exp(-bfactor / 4 * s2)
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
        fourier_images = torch.fft.fft2(image)
        #test = torch.fft.fftshift(fourier_images, dim=(-2, -1))
        #test_no_nyquist = test[:, 1:, 1:]
        #print("SHAPE", test_no_nyquist.shape)
        #print("TEST SYMMETRY", test_no_nyquist - torch.transpose(test_no_nyquist, dim0=-2, dim1=-1))
        #ctf_grid_test = torch.fft.fftshift(self.ctf_grid)
        #print("TEST CTF SYMMETRY", ctf_grid_test - torch.transpose(ctf_grid_test, dim0=-2, dim1=-2))

        corrupted_fourier = fourier_images*self.ctf_grid
        corrupted_images = torch.fft.ifft2(corrupted_fourier).real
        return corrupted_images

    def compute_x_y_values_all_atoms(self, atom_positions, rotation_matrices, translation_vectors,
                                     latent_type="continuous", volume=False):
        """

        :param atom_position: (N_batch, N_atoms, 3)
        :param rotation_matrices: (N_batch, 3, 3)
        :param translation_vectors: (N_batch, 3)
        :return:
        """
        transposed_atom_positions = torch.transpose(atom_positions, dim0=-2, dim1=-1)
        if latent_type=="continuous":
            #Rotation pose
            #TRYING TO ACCELERATE THIS PART
            rotated_transposed_atom_positions = torch.matmul(rotation_matrices, transposed_atom_positions)
            #Translation pose
            rotated_transposed_atom_positions += translation_vectors[:, :, None]
            #Trying to accelerate with the commmented code ?
            #rotated_atom_positions = torch.einsum("bkl, bal->bak", rotation_matrices, atom_positions)
            #rotated_atom_positions += translation_vectors[:, None, :]
            rotated_atom_positions = torch.transpose(rotated_transposed_atom_positions, -2, -1)
            all_x = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 0], self.pixels_x)
            all_y = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 1], self.pixels_y)
            #prod = torch.einsum("bki,bkj->bkij", (all_x, all_y))
            #projected_densities = torch.sum(prod, dim=1)
            if volume:
                all_z = self.compute_gaussian_kernel(rotated_atom_positions[:, :, 2], self.pixels_z)
                xy = torch.einsum("bki,bkj->bkij", (all_x, all_y))
                volume = torch.einsum("bkij, bkl -> bijl", (xy, all_z))
                return volume
            else:
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