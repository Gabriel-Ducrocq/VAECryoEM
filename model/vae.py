import torch as torch
from pytorch3d.transforms import rotation_6d_to_matrix


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, mask_start_values, N_domains=6, N_residues=1006, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous"):
        super(VAE, self).__init__()
        assert latent_type in ["continuous", "categorical"]
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.N_domains = N_domains
        self.N_residues = N_residues
        self.tau_mask = tau_mask
        self.latent_type = latent_type
        self.latent_dim = latent_dim

        self.slice_mean_rotation = slice(0, 6)
        self.slice_sigma_rotation = slice(6, 9)
        self.slice_mean_translation = slice(9, 12)
        self.slice_std_translation = slice(12, 15)

        self.lie_alg_l1 = torch.zeros((3, 3), device=self.device)
        self.lie_alg_l1[2, 1] = 1
        self.lie_alg_l1[1, 2] = -1

        self.lie_alg_l2 = torch.zeros((3, 3), device=self.device)
        self.lie_alg_l2[0, 2] = 1
        self.lie_alg_l2[2, 0] = -1

        self.lie_alg_l3 = torch.zeros((3, 3), device=self.device)
        self.lie_alg_l3[1, 0] = 1
        self.lie_alg_l3[0, 1] = -1

        self.elu = torch.nn.ELU()

        self.lie_alg_basis = torch.concat([self.lie_alg_l1[None, :, :], self.lie_alg_l2[None, :, :],
                                           self.lie_alg_l3[None, :, :]], dim=0)


        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

        self.mask_means_mean = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_mean"]["mean"], dtype=torch.float32,device=device)[None, :],
                                                requires_grad=True)

        self.mask_means_std = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_mean"]["std"], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.mask_std_mean = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_std"]["mean"], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.mask_std_std = torch.nn.Parameter(data=torch.tensor(mask_start_values["clusters_std"]["std"], dtype=torch.float32, device=device)[None, :],
                                              requires_grad=True)

        self.mask_proportions_mean = torch.nn.Parameter(torch.tensor(mask_start_values["clusters_proportions"]["mean"], dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)

        self.mask_proportions_std = torch.nn.Parameter(torch.tensor(mask_start_values["clusters_proportions"]["std"], dtype=torch.float32, device=device)[None, :],
                           requires_grad=True)

        self.mask_parameters = {"means":{"mean":self.mask_means_mean, "std":self.mask_means_std},
                                   "stds":{"mean":self.mask_std_mean, "std":self.mask_std_std},
                                   "proportions":{"mean":self.mask_proportions_mean, "std":self.mask_proportions_std}}

        self.elu = torch.nn.ELU()
    def sample_mask(self, N_batch):
        """
        Samples a mask
        :return: torch.tensor(N_batch, N_residues, N_domains) values of the mask
        """
        #cluster_proportions = torch.randn(self.N_domains, device=self.device)*self.mask_proportions_std + self.mask_proportions_mean
        #cluster_means = torch.randn(self.N_domains, device=self.device)*self.mask_means_std + self.mask_means_mean
        #cluster_std = self.elu(torch.randn(self.N_domains, device=self.device)*self.mask_std_std + self.mask_std_mean) + 1
        #proportions = torch.softmax(cluster_proportions, dim=1)
        #log_num = -0.5*(self.residues - cluster_means)**2/cluster_std**2 + \
        #      torch.log(proportions)

        #mask = torch.softmax(log_num/self.tau_mask, dim=1)
        cluster_proportions = torch.randn((N_batch, self.N_domains),
                                          device=self.device) * self.mask_proportions_std+ self.mask_proportions_mean
        cluster_means = torch.randn((N_batch, self.N_domains), device=self.device) * self.mask_means_std+ self.mask_means_mean
        cluster_std = self.elu(torch.randn((N_batch, self.N_domains), device=self.device)*self.mask_std_std + self.mask_std_mean) + 1
        proportions = torch.softmax(cluster_proportions, dim=-1)
        log_num = -0.5*(self.residues[None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
              torch.log(proportions[:, None, :])

        mask = torch.softmax(log_num / self.tau_mask, dim=-1)
        return mask

    def sample_latent(self, images):
        """
        Samples latent variables given an image. This latent variable is a tensor of size
        (N_batch, N_domain*15) where 15 comes from the fact that we sample a translation (3 dim), an uncertainty
        std (3 dim) on translation, a mean rotation matrix R\mu (6 dim before Gram Schmidt) and uncertainties stds
        (3 dim) on the Lie algebra.
        :param images: torch.tensor(N_batch, N_pix_x, N_pix_y)
        :return: torch.tensor(N_batch, N_domains, 3, 3) rotation matrix,
                torch.tensor(N_batch, N_domains, 3, 3) mean rotation matrix,
                torch.tensor(N_batch, N_domains, 3, 3) noise rotation matrix,
                torch.tensor(N_batch, N_domains, 3) rotation std
                torch.tensor(N_batch, N_domains, 3) translation
                torch.tensor(N_batch, N_domains, 3) mean translation
                torch.tensor(N_batch, N_domains, 3) std translation
        """

        N_batch = images.shape[0]
        output = self.encoder(images)
        output_per_domain = torch.reshape(output, (N_batch, self.N_domains, 15))
        mean_translation = output_per_domain[:, :, self.slice_mean_translation]
        sigma_translation = self.elu(output_per_domain[:, :, self.slice_std_translation]) + 1
        translation_per_domain = mean_translation + torch.randn_like(mean_translation)*sigma_translation


        mean_rotations = rotation_6d_to_matrix(output_per_domain[:, :, self.slice_mean_rotation])
        std_rot = self.elu(output_per_domain[:, :, self.slice_sigma_rotation]) + 1
        #We first sample in R^3
        noise_rot = torch.randn_like(std_rot)*std_rot
        #Then we get the norm of the vectors and normalize them
        theta = torch.sqrt(torch.sum(noise_rot**2, dim=-1))
        normalized_noise_rot = noise_rot/theta[:, :, None]
        #Then, we project the normalized vectors in the Lie algebra so(3) thanks to the isomorphism.
        normalized_matrices = normalized_noise_rot[:, :, 0, None, None] * self.lie_alg_l1[None, None, :, :] + \
                             normalized_noise_rot[:, :, 1, None, None] * self.lie_alg_l2[None, None, :, :] + \
                              normalized_noise_rot[:, :, 2, None, None] * self.lie_alg_l3[None, None, :, :]

        #normalized_matrices = torch.einsum("blk, kmn -> blmn", normalized_noise_rot, self.lie_alg_basis)
        #Finally, we use the exponential map (Rodrigues formula) to map from the Lie algebra so(3) to the
        # Lie group SO(3)

        uncertainty_matrices = torch.eye(3)[None, None, :, :] + torch.sin(theta)[:, :, None, None]*normalized_matrices + (1-torch.cos(theta))[:, :, None, None]\
                                        *torch.einsum("blmj, bljn -> blmn", normalized_matrices, normalized_matrices)

        #We then shift the noise matrix (centered in the neutral element) with the mean matrix, using the group
        #operation: left multiplication.
        sample_matrix = torch.einsum("blmj, bljn->blmn", mean_rotations, uncertainty_matrices)
        return sample_matrix, mean_rotations, noise_rot, std_rot, translation_per_domain, mean_translation, sigma_translation





