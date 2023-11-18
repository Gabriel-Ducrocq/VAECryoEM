import torch
import torch.nn as nn

class GaussianPyramid(nn.Module):
    def __init__(self, kernel_size, kernel_variance, num_octaves, octave_scaling):
        """
        Initialize a set of gaussian filters.

        Parameters
        ---------
        kernel_size: int
        kernel_variance: float
        num_octaves: int
        octave_scaling: int
        """
        super(GaussianPyramid,self).__init__()
        #Size of the kernel
        self.kernel_size = kernel_size
        #Variance of the kernel
        self.variance = kernel_variance
        # Number of scales
        self.num_dec = num_octaves
        #Scaling for the kernels
        self.scaling = octave_scaling

        weighting = torch.ones([num_octaves], dtype=torch.float32)
        self.register_buffer('weighting', weighting)
        #Create kernels and get a tensor (N_scales, 1, kernel_size, kernel_size)
        self.kernels = self.generateGaussianKernels(kernel_size, kernel_variance, num_octaves + 1, octave_scaling)

        #Initialize a conv2d layer with the same number of kernels
        self.gaussianPyramid = torch.nn.Conv2d(1, num_octaves + 1,
                                               kernel_size=kernel_size,
                                               padding='same', padding_mode='reflect', bias=False)

        #Set the weights to the kernel defined above, so that we convolve with the right Gaussian filters
        self.gaussianPyramid.weight = torch.nn.Parameter(self.kernels)
        #Desactivate the gradient on this
        self.gaussianPyramid.weight.requires_grad = False

    def generateGaussianKernels(self, size, var, scales=1, scaling=2):
        """
        Generate a list of gaussian kernels

        Parameters
        ----------
        size: int
        var: float
        scales: int
        scaling: int

        Returns
        -------
        kernels: list of torch.Tensor
        """
        ##Create two grids representing the coordinates along x and y axis
        coords = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
        #Stack them on a batch dim created on the fly
        xy = torch.stack(torch.meshgrid(coords, coords),dim=0)
        #Compute the exp(x**2+y**2) for every coordinates, different variances and scalings
        kernels = [torch.exp(-(xy ** 2).sum(0) / (2 * var * scaling ** i)) for i in range(scales)]
        #Stack all the kernels
        kernels = torch.stack(kernels,dim=0)
        #Normalize the kernels
        kernels /= kernels.sum((1, 2), keepdims=True)

        #Get dimension (N_scales, 1, kernel_size, kernel_size)
        kernels = kernels[:, None, ...]
        return kernels

    def forward(self, x):
        return self.gaussianPyramid(x[:, None, :, :])


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """
    3x3 convolution with padding.

    Parameters
    ----------
    in_planes: int
    out_planes: int
    stride: int
    bias: bool

    Returns
    -------
    out: torch.nn.Module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, triple=False):
        """
        Initialization of a double convolutional block.

        Parameters
        ----------
        in_size: int
        out_size: int
        batch_norm: bool
        triple: bool
        """
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.triple = triple

        self.conv1 = conv3x3(in_size, out_size)
        self.conv2 = conv3x3(out_size, out_size)
        if triple:
            self.conv3 = conv3x3(out_size, out_size)

        self.relu = nn.ReLU(inplace=True)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_size)
            self.bn2 = nn.BatchNorm2d(out_size)
            if triple:
                self.bn3 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        if self.triple:
            out = self.relu(out)

            out = self.conv3(out)
            if self.batch_norm:
                out = self.bn3(out)

        out = self.relu(out)

        return out


class VGG16Like(nn.Module):
    def __init__(self, in_channels=3, batch_norm=False, flip_images=False, high_res=False):
        """
        Initialization of a VGG16-like encoder.

        Parameters
        ----------
        in_channels: int
        batch_norm: bool
        pretrained: bool
        flip_images: bool
        high_res: bool
        """
        super(VGG16Like, self).__init__()
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.flip_images = flip_images
        self.high_res = high_res
        self.feature_channels = [64, 128, 256, 256, 1024, 2048]
        self.net = []

        prev_chanel = in_channels
        for k, next_channel in enumerate(self.feature_channels):
            self.net.append(DoubleConvBlock(prev_chanel, next_channel, batch_norm=batch_norm))
            if k <= 2:
                self.net.append(nn.MaxPool2d(kernel_size=2))
            else:
                self.net.append(nn.AvgPool2d(kernel_size=2))

            prev_chanel = next_channel

        self.net.append(nn.MaxPool2d(kernel_size=2))

        self.net = nn.Sequential(*self.net)
        self.register_buffer('means', torch.tensor([0.45] * self.in_channels).reshape(1, self.in_channels))
        self.register_buffer('stds', torch.tensor([0.226] * self.in_channels).reshape(1, self.in_channels))


    ### CHECK THIS PART OF THE CODE
    def normalize_repeat(self, input):
        """
        Normalize input.

        Parameters
        ----------
        input: torch.Tensor

        Returns
        -------
        out: torch.Tensor
        """
        N = input.shape[0]
        C_in = self.in_channels
        C_out = self.in_channels
        # input: N, C_in, H, W
        # self.means/std: N, C_out
        means = torch.mean(input, (2, 3))  # N, C_in
        stds = torch.std(input, (2, 3))  # N, C_in
        alphas = (self.stds / stds).reshape(N, C_out, 1, 1)  # N, C_out, 1, 1
        c = (self.means.reshape(1, C_out, 1, 1) / alphas -
             means.reshape(N, C_in, 1, 1)).reshape(N, C_out, 1, 1)
        return alphas * (input.repeat(1, int(C_out/C_in), 1, 1) + c)

    def forward(self, input):
        input_augmented = input
        normalized_input = self.normalize_repeat(input_augmented)
        out = self.net(normalized_input)
        return out


class CNNEncoder(nn.Module):
    def __init__(self, kernel_size, kernel_variance, num_octaves, octave_scaling, in_channels=5, batch_norm=False,
                 flip_images=False, high_res=False, device="cpu"):
        super(CNNEncoder, self).__init__()
        self.vgg_like = VGG16Like(in_channels, batch_norm, flip_images, high_res)
        self.gaussian_pyramid = GaussianPyramid(kernel_size, kernel_variance, num_octaves, octave_scaling)
        self.fc = nn.Sequential(*[nn.Linear(2048, 512, device=device), nn.ReLU(), nn.Linear(512, 256, device=device), nn.ReLU(),
                    nn.Linear(256, 3*2, device=device)])

    def forward(self, images):
        """

        :param images: torch.tensor(N_batch, N_pix, N_pix)
        :return: torch.tensor(Batch_size, 6) a representation of rotation in R^6
        """
        filtered_images = self.gaussian_pyramid(images)
        print(self.vgg_like(filtered_images).shape)
        vgg_output = torch.flatten(self.vgg_like(filtered_images), start_dim=1)
        print(vgg_output.shape)
        print("OUTPUT CONV SHAPE", vgg_output.shape)
        print(vgg_output[:, :15])
        output = self.fc(vgg_output)
        return output






