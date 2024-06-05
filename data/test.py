import torch
import matplotlib.pyplot as plt

mu = -1*torch.ones(1)
std = 1*torch.ones(1)
denom = torch.sqrt(2*std**2*torch.pi)
def pdf(t):
    return torch.exp(-0.5*(t-mu)**2/std**2)/denom

def pdf_freqs(t):
    return torch.exp(-2*1j*torch.pi*mu*t - 2*torch.pi**2*std**2*t**2)

#def pdf_freqs(t):
#    return torch.exp(-1j*mu*t - 0.5*std**2*t**2)


voxel_size = 1
side_n_pixels = 10
origin = -side_n_pixels//2 * voxel_size

x = torch.linspace(origin, (side_n_pixels - 1) * voxel_size + origin, side_n_pixels)
freqs = torch.fft.fftshift(torch.fft.fftfreq(side_n_pixels, voxel_size))

y = torch.stack([pdf(xx) for xx in x], dim=0)[:, 0]
image_real = torch.einsum("p, q -> p q", y , y)
plt.imshow(image_real.detach().numpy())
plt.show()
#y = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(y)))
y_freqs = torch.stack([pdf_freqs(ff) for ff in freqs], dim=0)[:, 0]
image_fourier = torch.einsum("p, q -> p q", y_freqs, y_freqs)
image_pred = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(image_fourier)).real)
plt.imshow(image_pred.detach().numpy())
plt.show()
#y_freqs = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(y_freqs)))
print(torch.abs(y_freqs.real - y.real)/torch.abs(y.real))
plt.plot(torch.abs(y_freqs.real - y.real)/torch.abs(y_freqs.real).detach().numpy())
plt.show()
error_real = torch.abs(y.real - y_freqs.real)/torch.abs(y_freqs.real)
error_imag = torch.abs(y.imag - y_freqs.imag)/torch.abs(y_freqs.imag)
print("Error real", torch.abs(y.real - y_freqs.real)/torch.abs(y_freqs.real))
print("Error imag", torch.abs(y.imag - y_freqs.imag)/torch.abs(y_freqs.imag))
plt.plot(error_real.detach().numpy())
plt.show()

plt.plot(error_imag.detach().numpy())
plt.show()








