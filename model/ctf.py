import torch
import starfile
import numpy as np
import matplotlib.pyplot as plt


class CTF(torch.nn.Module):
	"""
	Class describing the ctf, built from starfile
	"""
	def __init__(self, side_shape, apix, defocusU, defocusV, defocusAngle, voltage, sphericalAberration, amplitudeContrastRatio, phaseShift=None ,scalefactor = None,
		bfactor= None):
		"""
		device: str, device to use.
		side shape: number of pixels on a side.
		apix: size of a pixel in Å.
		defocusU: defocusU.
		defocusV: defocusV.
		defocusAngle: defocus angle in degrees.
		voltage: accelerating voltage in keV.
		sphericalAberration: spherical aberration in mm.
		AmplitudeContrastRatio: amplitude contrat ratio.
		phaseShift: phase shift in degrees.
		"""
		super().__init__()
		if phaseShift is None:
			phaseShift = torch.zeros_like(defocusU, dtype=torch.float32)
		else:
			phaseShift = torch.tensor(phaseShift, dtype=torch.float32)

		if scalefactor is None:
			scalefactor = torch.ones_like(defocusU, dtype=torch.float32)
		else:
			scalefactor = torch.tensor(scalefactor, dtype=torch.float32)

		if bfactor is None:
			bfactor = torch.zeros_like(defocusU, dtype=torch.float32)
		else:
			bfactor = torch.tensor(bfactor, dtype=torch.float32)


		saved_args = locals()
		assert len(set({len(val) for arg_name,val in saved_args.items() if arg_name not in ["self", "__class__", "side_shape", "apix"]})) == 1, "CTF values do not have the same shape."

		self.register_buffer("Npix", torch.ones((1, ))*side_shape)
		self.register_buffer("Apix", torch.ones((1, ))*apix)
		self.register_buffer("dfU", torch.tensor(defocusU[:, None], dtype=torch.float32))
		self.register_buffer("dfV", torch.tensor(defocusV[:, None], dtype=torch.float32))
		self.register_buffer("dfang", torch.tensor(defocusAngle[:, None], dtype=torch.float32))
		self.register_buffer("volt", torch.tensor(voltage[:, None], dtype=torch.float32))
		self.register_buffer("cs", torch.tensor(sphericalAberration[:, None], dtype=torch.float32))
		self.register_buffer("w", torch.tensor(amplitudeContrastRatio[:, None], dtype=torch.float32))
		self.register_buffer("phaseShift", phaseShift[:, None])
		self.register_buffer("scalefactor", scalefactor[:, None])
		self.register_buffer("bfactor", bfactor[:, None])
		self.npix = side_shape
		self.apix = apix
		#In this stack, freqs[0, :] corresponds to constant x values, freqs[:, 0] corresponds to contant y values.
		freqs = (
		    torch.stack(
		        self.meshgrid_2d(-0.5, 0.5, int(self.npix), endpoint=False),
		        -1,
		    )
		    / self.apix)

		self.freqs = freqs.reshape(-1, 2)
		#In freqs, x is the first coordinate, y is the second and we are in x major
		#ctf = self.compute_ctf(freqs, self.dfU, self.dfV, self.dfang, self.volt, self.cs, self.w, self.phaseShift, None, None)


	@classmethod
	def from_starfile(cls, file, **kwargs):
		df = starfile.read(file)

		overrides = {}

		#First we find the values of the CTF im the optics block of the star file.
		try:
			side_n_pix = int(df["optics"].loc[0, "rlnImageSize"])
			apix = df["optics"].loc[0, "rlnImagePixelSize"]
		except Exception:
		    assert "side_shape" in kwargs and "apix" in kwargs, "side_shape, apix must be provided."
		    side_n_pix = kwargs["side_shape"]
		    apix = kwargs["apix"]

		if "optics" in df:
		    assert len(df["optics"]) == 1, "Currently only support one optics group."
		    overrides["rlnVoltage"] = df["optics"].loc[0, "rlnVoltage"]
		    overrides["rlnSphericalAberration"] = df["optics"].loc[0, "rlnSphericalAberration"]
		    overrides["rlnAmplitudeContrast"] = df["optics"].loc[0, "rlnAmplitudeContrast"]

		#Second, if there are particles in the file we find defocus U, V and angle for each one of them. Otherwise we just find the 
		#values in the optics block and there is only one ctf. Notte that the CTF parameters in the optics groups have precedence on
		# the ones in the data block.

		if "particles" in df:
			df = df["particles"]

		num = len(df)
		ctf_params = np.zeros((num, 9))
		ctf_params[:, 0] = side_n_pix
		ctf_params[:, 1] = apix
		for i, header in enumerate([
		"rlnDefocusU",
		"rlnDefocusV",
		"rlnDefocusAngle",
		"rlnVoltage",
		"rlnSphericalAberration",
		"rlnAmplitudeContrast",
		"rlnPhaseShift",
		]):
			if header in overrides:
				ctf_params[:, i + 2] = overrides[header]
			else:
				ctf_params[:, i + 2] = df[header].values if header in df else None

		return cls(side_n_pix, apix, *ctf_params.T)

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
		    values = torch.linspace(lo, hi, n)
		else:
		    values = torch.linspace(lo, hi, n + 1)[:-1]

		return torch.meshgrid(values, values, indexing="xy")

	def compute_ctf(self, indexes
		) -> torch.Tensor:
		"""

		This code is based on the cryoDRGN code.
		Compute the 2D CTF

		Input:
		    indexes: torch.tensor(batch_size) of indexes of the images in this batch.
		"""
		# convert units
		volt = self.volt[indexes]
		cs = self.cs[indexes]
		dfu = self.dfU[indexes]
		dfv = self.dfV[indexes]
		dfang = self.dfang[indexes]
		w = self.w[indexes]
		phase_shift = self.phaseShift[indexes]
		bfactor = self.bfactor[indexes]
		scalefactor = self.scalefactor[indexes]


		volt = volt * 1000
		cs = cs * 10 ** 7
		dfang = dfang * np.pi / 180
		phase_shift = phase_shift * np.pi / 180

		# lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
		lam = 12.2639 / torch.sqrt(volt + 0.97845e-6 * volt ** 2)
		x = self.freqs[..., 0]
		y = self.freqs[..., 1]
		#Since we take arctan between y and x and not x and y, we are still in x ordering but x is the second coordinate now !
		ang = torch.arctan2(y, x)
		s2 = x ** 2 + y ** 2

		df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
		gamma = (
		        2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam ** 3 * s2 ** 2)
		        - phase_shift
		)
		ctf = torch.sqrt(1 - w ** 2) * torch.sin(gamma) - w * torch.cos(gamma)
		if scalefactor is not None:
			scalefactor = self.scalefactor[indexes]
			ctf *= scalefactor
		if bfactor is not None:
			bfactor = self.bfactor[indexes]
			ctf *= torch.exp(-bfactor / 4 * s2)


		#But in this project, the images are (y_coords, x_coords), see renderer.project so we transpose:
		ctf = ctf.reshape((len(indexes), self.npix, self.npix))
		return torch.transpose(ctf, dim0=-2, dim1=-1)

	def compute_ctf_inefficient(self,
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
		#Since we take arctan between y and x and not x and y, we are still in x ordering but x is the second coordinate now !
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


		#But in this project, the images are (y_coords, x_coords), see renderer.project so we transpose:
		ctf = ctf.reshape((N_images, Npix, Npix))
		return torch.transpose(ctf, dim0=-2, dim1=-1)
  






