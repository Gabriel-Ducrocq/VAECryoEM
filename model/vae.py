import torch as torch


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, mask_start_values, N_domains=6, N_residues=1006, tau_mask=0.05):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.N_domains = N_domains
        self.N_residues = N_residues
        self.tau_mask = tau_mask

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

    def sample_mask(self):
        """
        Samples a mask
        :return: torch.tensor(N_residues, N_domains) values of the mask
        """
        cluster_proportions = torch.randn(self.N_domains, device=self.device)*self.mask_proportions_std + self.mask_proportions_mean
        cluster_means = torch.randn(self.N_domains, device=self.device)*self.mask_means_std + self.mask_means_mean
        cluster_std = torch.randn(self.N_domains, device=self.device)*self.mask_std_std + self.mask_std_mean
        proportions = torch.softmax(cluster_proportions, dim=1)
        log_num = -0.5*(self.residues - cluster_means)**2/cluster_std**2 + \
              torch.log(proportions)

        mask = torch.softmax(log_num/self.tau_mask, dim=1)
        return mask

    def sample_latent(self, images):
        """
        Samples latent variables given an image
        :param images: torch.tensor(N_batch, N_pix_x, N_pix_y)
        :return: torch.tensor(N_batch, latent_dim) latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std
        """
        latent_mean, latent_std = self.encoder(images)
        latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                            *latent_std

        return latent_variables, latent_mean, latent_std

    def decode(self, latent_variables):
        """
        Decode the latent variables
        :param latent_variables: torch.tensor(N_batch, latent_dim)
        :return: torch.tensor(N_batch, N_domains, 4) quaternions, torch.tensor(N_batch, N_domains, 3) translations
        """
        N_batch = latent_variables.shape[0]
        transformations = self.decoder(latent_variables)
        transformations_per_domain = torch.reshape(transformations, (N_batch, self.N_domains, 6))
        ones = torch.ones(size=(N_batch, self.N_domains, 1), device=self.device)
        quaternions_per_domain = torch.concat([ones, transformations_per_domain[:, :, 3:]], dim=-1)
        translations_per_domain = transformations_per_domain[:, :, :3]
        return quaternions_per_domain, translations_per_domain



