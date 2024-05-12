import sys
import os
path = os.path.abspath("model")
sys.path.append(path)
print("AAAA")
import starfile
print("BBBB")
from ctf import CTF
print("CCCC")
import argparse
print("DDDDD")
import pickle as pkl
print("EEEEEE")
import cryostar
print("FFFFFF")
import torch
print("GGGGGGG")
import pickle
from cryostar.utils.ctf import parse_ctf_star
import random
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt


parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--star_file', type=str, required=True)
parser_arg.add_argument('--ctf_pickle', type=bool, required=False)



class Lattice_cryoDRGN:
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
        self.ignore_DC = ignore_DC
        self.device = device

    def get_downsample_coords(self, d: int) -> Tensor:
        assert d % 2 == 1
        extent = self.extent * (d - 1) / (self.D - 1)
        x0, x1 = np.meshgrid(
            np.linspace(-extent, extent, d, endpoint=True),
            np.linspace(-extent, extent, d, endpoint=True),
        )
        coords = np.stack([x0.ravel(), x1.ravel(), np.zeros(d**2)], 1).astype(
            np.float32
        )
        return torch.tensor(coords, device=self.device)

    def get_square_lattice(self, L: int) -> Tensor:
        b, e = self.D2 - L, self.D2 + L + 1
        center_lattice = (
            self.coords.view(self.D, self.D, 3)[b:e, b:e, :].contiguous().view(-1, 3)
        )
        return center_lattice

    def get_square_mask(self, L: int) -> Tensor:
        """Return a binary mask for self.coords which restricts coordinates to a centered square lattice"""
        if L in self.square_mask:
            return self.square_mask[L]
        assert (
            2 * L + 1 <= self.D
        ), "Mask with size {} too large for lattice with size {}".format(L, self.D)
        logger.info("Using square lattice of size {}x{}".format(2 * L + 1, 2 * L + 1))
        b, e = self.D2 - L, self.D2 + L
        c1 = self.coords.view(self.D, self.D, 3)[b, b]
        c2 = self.coords.view(self.D, self.D, 3)[e, e]
        m1 = self.coords[:, 0] >= c1[0]
        m2 = self.coords[:, 0] <= c2[0]
        m3 = self.coords[:, 1] >= c1[1]
        m4 = self.coords[:, 1] <= c2[1]
        mask = m1 * m2 * m3 * m4
        self.square_mask[L] = mask
        if self.ignore_DC:
            raise NotImplementedError
        return mask

    def get_circular_mask(self, R: float) -> Tensor:
        """Return a binary mask for self.coords which restricts coordinates to a centered circular lattice"""
        if R in self.circle_mask:
            return self.circle_mask[R]
        assert (
            2 * R + 1 <= self.D
        ), "Mask with radius {} too large for lattice with size {}".format(R, self.D)
        logger.debug("Using circular lattice with radius {}".format(R))
        r = R / (self.D // 2) * self.extent
        mask = self.coords.pow(2).sum(-1) <= r**2
        if self.ignore_DC:
            assert self.coords[self.D**2 // 2].sum() == 0.0
            mask[self.D**2 // 2] = 0
        self.circle_mask[R] = mask
        return mask

    def rotate(self, images: Tensor, theta: Tensor) -> Tensor:
        """
        images: BxYxX
        theta: Q, in radians
        """
        images = images.expand(len(theta), *images.shape)  # QxBxYxX
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        rot = torch.stack([cos, sin, -sin, cos], 1).view(-1, 2, 2)
        grid = self.coords[:, 0:2] / self.extent @ rot  # grid between -1 and 1
        grid = grid.view(len(rot), self.D, self.D, 2)  # QxYxXx2
        offset = self.center - grid[:, self.D2, self.D2]  # Qx2
        grid += offset[:, None, None, :]
        rotated = F.grid_sample(images, grid)  # QxBxYxX
        return rotated.transpose(0, 1)  # BxQxYxX

    def translate_ft(self, img, t, mask=None):
        """
        Translate an image by phase shifting its Fourier transform

        Inputs:
            img: FT of image (B x img_dims x 2)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims x 2)

        img_dims can either be 2D or 1D (unraveled image)
        """
        # F'(k) = exp(-2*pi*k*x0)*F(k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img.unsqueeze(1)  # Bx1xNx2
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * -2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return torch.stack(
            [img[..., 0] * c - img[..., 1] * s, img[..., 0] * s + img[..., 1] * c], -1
        )

    def translate_ht(self, img, t, mask=None):
        """
        Translate an image by phase shifting its Hartley transform

        Inputs:
            img: HT of image (B x img_dims)
            t: shift in pixels (B x T x 2)
            mask: Mask for lattice coords (img_dims x 1)

        Returns:
            Shifted images (B x T x img_dims)

        img must be 1D unraveled image, symmetric around DC component
        """
        # H'(k) = cos(2*pi*k*t0)H(k) + sin(2*pi*k*t0)H(-k)
        coords = self.freqs2d if mask is None else self.freqs2d[mask]
        img = img.unsqueeze(1)  # Bx1xN
        t = t.unsqueeze(-1)  # BxTx2x1 to be able to do bmm
        tfilt = coords @ t * 2 * np.pi  # BxTxNx1
        tfilt = tfilt.squeeze(-1)  # BxTxN
        c = torch.cos(tfilt)  # BxTxN
        s = torch.sin(tfilt)  # BxTxN
        return c * img + s * img[:, :, torch.arange(len(coords) - 1, -1, -1)]





def compute_ctf_cryodrgn(freqs: torch.Tensor, dfu: torch.Tensor, dfv: torch.Tensor, dfang: torch.Tensor, volt: torch.Tensor, cs: torch.Tensor, w: torch.Tensor, phase_shift: torch.Tensor = None,scalefactor: torch.Tensor = None, bfactor: torch.Tensor = None,) -> torch.Tensor:
    """
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
    cs = cs * 10**7
    dfang = dfang * np.pi / 180
    if phase_shift is None:
        phase_shift = torch.tensor(0)
    phase_shift = phase_shift * np.pi / 180
    # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
    lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt**2)
    x = freqs[..., 0]
    y = freqs[..., 1]
    ang = torch.arctan2(y, x)
    s2 = x**2 + y**2
    df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
    gamma = (
        2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2) 
        - phase_shift

    )
    ctf = torch.sqrt(1 - w**2) * torch.sin(gamma) - w * torch.cos(gamma)

    if scalefactor is not None:
        ctf *= scalefactor
    if bfactor is not None:
        ctf *= torch.exp(-bfactor / 4 * s2)
    return ctf





def run(path_star, path_pickle):

	print("A")
	ctf_cryosphere_obj = CTF.from_starfile(path_star)
	ctf_cryosphere = CTF.from_starfile(path_star)
	print("B")
	#with open(path_pickle, "rb") as f:
	#	ctf_cryodrgn = torch.tensor(pickle.load(f))

	ctf_cryostar = parse_ctf_star(path_star)

	print("B")
	l = len(ctf_cryosphere.dfU)
	ctf_cryosphere_array = torch.concat([ctf_cryosphere.Npix, ctf_cryosphere.Apix, ctf_cryosphere.dfU, ctf_cryosphere.dfV, ctf_cryosphere.dfang, ctf_cryosphere.volt, ctf_cryosphere.cs, ctf_cryosphere.w, ctf_cryosphere.phaseShift], dim=1)

	ctf_cryodrgn = ctf_cryosphere_array
	rel_err_cryodrgn = torch.max(torch.abs(ctf_cryosphere_array - ctf_cryodrgn)[:, :-1]/ctf_cryodrgn[:, :-1])
	rel_err_cryostar = torch.max(torch.abs(ctf_cryosphere_array - ctf_cryostar)[:, :-1]/ctf_cryostar[:, :-1])


	D = int(ctf_cryodrgn.detach().numpy()[0, 0])
	lattice = Lattice_cryoDRGN(int(ctf_cryodrgn[0, 0]), extent=0.5)

	B = 128
	sample_indexes = random.sample([i for i in range(len(ctf_cryodrgn))], B)
	freqs = lattice.freqs2d.unsqueeze(0).expand(
	    B, *lattice.freqs2d.shape
	) / ctf_cryodrgn[sample_indexes, 1].view(B, 1, 1)
	ctf_cryodrgn = compute_ctf_cryodrgn(freqs, *torch.split(ctf_cryodrgn[sample_indexes, 2:], 1, 1)).view(B, D, D)
	#ctf_cryodrgn = compute_ctf_cryodrgn(ctf_cryosphere_obj.freqs, *torch.split(ctf_cryodrgn[sample_indexes, 2:], 1, 1)).view(B, D, D)
	print(ctf_cryodrgn.shape)
	ctf_cryodrgn_flattened = ctf_cryodrgn.view(B, -1)

	ctf_cryosphere = ctf_cryosphere.compute_ctf(sample_indexes)
	print(ctf_cryosphere.shape)

	print("cryosphere", ctf_cryosphere)
	print("\n\n\n\n")
	print("cryodrgn", ctf_cryodrgn)
	print("\n\n\n\n")
	print("Difference CTF", torch.max(torch.abs(ctf_cryosphere - ctf_cryodrgn)))
	print("FREQSSS")
	print(freqs)
	print("\n")
	print("Difference freqs", freqs - ctf_cryosphere_obj.freqs)
	print("\n")
	print(ctf_cryosphere_obj.freqs)
	print(freqs[0, :, :])
	plt.imshow(ctf_cryosphere.detach().numpy()[0], cmap="gray")
	plt.savefig("cryosphere_ctf.png")
	plt.show()
	plt.close()

	plt.imshow(ctf_cryodrgn.detach().numpy()[0], cmap="gray")
	plt.savefig("cryodrgn_ctf.png")
	plt.show()
	plt.close()






if __name__ == '__main__':
	args = parser_arg.parse_args()
	path_star = args.star_file
	path_pickle= args.ctf_pickle
	print("AAAAAAAAAA")
	run(path_star, path_pickle)







"""
    for i, header in enumerate([
            "rlnDefocusU",
            "rlnDefocusV",
            "rlnDefocusAngle",
            "rlnVoltage",
            "rlnSphericalAberration",
            "rlnAmplitudeContrast",
            "rlnPhaseShift",
    ]):

print("Npix", ctf_cryosphere.Npix - ctf_cryodrgn[:, 0])
print("Apix", ctf_cryosphere.apix - ctf_cryodrgn[:, 1])
print("dfU",  ctf.ctf_cryosphere.dfU - ctf_cryodrgn[:, 2])



		self.register_buffer("Npix", torch.tensor(side_shape, dtype=torch.float32, device=device)[:, None])
		self.register_buffer("Apix", torch.tensor(apix, dtype=torch.float32, device=device)[:, None])
		self.register_buffer("dfU", torch.tensor(defocusU[:, None], dtype=torch.float32, device=device))
		self.register_buffer("dfV", torch.tensor(defocusV[:, None], dtype=torch.float32, device=device))
		self.register_buffer("dfang", torch.tensor(defocusAngle[:, None], dtype=torch.float32, device=device))
		self.register_buffer("volt", torch.tensor(voltage[:, None], dtype=torch.float32, device=device))
		self.register_buffer("cs", torch.tensor(sphericalAberration[:, None], dtype=torch.float32, device=device))
		self.register_buffer("w", torch.tensor(amplitudeContrastRatio[:, None], dtype=torch.float32, device=device))
		self.register_buffer("phaseShift", phaseShift[:, None])
		self.register_buffer("scalefactor", scalefactor[:, None])
		self.register_buffer("bfactor", bfactor[:, None])


"""