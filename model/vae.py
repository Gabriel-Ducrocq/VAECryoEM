import torch
import numpy as np


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, device, mask_start_values, N_chains, N_domains, N_residues, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous", amortized=True, N_images=None, chain_ids=["A"]):
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
        self.N_images = N_images
        self.N_chains = N_chains
        self.amortized = amortized
        self.total_n_domains = sum(self.N_domains.values())
        self.chain_ids = chain_ids

        self.dict_segments_means_means = torch.nn.ParameterDict({})
        self.dict_segments_means_stds = torch.nn.ParameterDict({})
        self.dict_segments_stds_means = torch.nn.ParameterDict({})
        self.dict_segments_stds_stds = torch.nn.ParameterDict({})
        self.dict_segments_proportions_means = torch.nn.ParameterDict({})
        self.dict_segments_proportions_stds = torch.nn.ParameterDict({})
        self.residues = {}

        for n_chain in self.chain_ids:
            assert mask_start_values["type"] == "uniform", "Currently, only uniform initialization of the segmentation available."
            bound_0 = self.N_residues[f"chain_{n_chain}"]/N_domains[f"chain_{n_chain}"]
            #Creating the variables for the mean
            self.segments_means_means = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_domains[f"chain_{n_chain}"])]), dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)

            self.segments_means_stds = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains[f"chain_{n_chain}"])*10.0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)


            #Creating the variables for the std
            self.segments_stds_means = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains[f"chain_{n_chain}"])*bound_0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)


            self.segments_stds_stds = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains[f"chain_{n_chain}"]) * 10.0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)


            self.segments_proportions_means = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains[f"chain_{n_chain}"]) * 0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.segments_proportions_stds = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains[f"chain_{n_chain}"]), dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.dict_segments_means_means[f"chain_{n_chain}"] = self.segments_means_means
            self.dict_segments_means_stds[f"chain_{n_chain}"] = self.segments_means_stds
            self.dict_segments_stds_means[f"chain_{n_chain}"] = self.segments_stds_means
            self.dict_segments_stds_stds[f"chain_{n_chain}"] = self.segments_stds_stds
            self.dict_segments_proportions_means[f"chain_{n_chain}"] = self.segments_proportions_means
            self.dict_segments_proportions_stds[f"chain_{n_chain}"] = self.segments_proportions_stds
            self.residues[f"chain_{n_chain}"] = torch.arange(0, self.N_residues[f"chain_{n_chain}"], 1, dtype=torch.float32, device=device)[:, None]


        self.elu = torch.nn.ELU()

        if amortized:
            assert N_images, "If using a non amortized version of the code, the number of images must be specified"
            self.latent_variables_mean = torch.nn.Parameter(torch.zeros(N_images, self.latent_dim, dtype=torch.float32, device=device), requires_grad=True)
            self.latent_variables_std = torch.nn.Parameter(torch.ones(N_images, self.latent_dim, dtype=torch.float32, device=device), requires_grad=False)

    def sample_mask(self, N_batch):
        """
        Samples a mask
        :return: dictionnary of segmentation: torch.tensor(N_batch, N_residues_in_chain, N_domains) values of the mask
        """
        chain_segments = {}
        for n_chain in self.chain_ids:
            segments_proportions_means = self.dict_segments_proportions_means[f"chain_{n_chain}"]
            segments_proportions_stds = self.dict_segments_proportions_stds[f"chain_{n_chain}"]

            segments_mean_means = self.dict_segments_means_means[f"chain_{n_chain}"]
            segments_mean_stds = self.dict_segments_means_stds[f"chain_{n_chain}"]

            segments_stds_means = self.dict_segments_stds_means[f"chain_{n_chain}"]
            segments_stds_stds = self.dict_segments_stds_stds[f"chain_{n_chain}"]

            cluster_proportions = torch.randn((N_batch, self.N_domains[f"chain_{n_chain}"]),
                                              device=self.device) * segments_proportions_stds + segments_proportions_means

            cluster_means = torch.randn((N_batch, self.N_domains[f"chain_{n_chain}"]), device=self.device) * segments_means_stds + segments_means_means

            cluster_std = self.elu(torch.randn((N_batch, self.N_domains[f"chain_{n_chain}"]), device=self.device)*segments_stds_stds + segments_stds_means) + 1
            proportions = torch.softmax(cluster_proportions, dim=-1)
            log_num = -0.5*(self.residues[f"chain_{n_chain}"][None, :, :] - cluster_means[:, None, :])**2/cluster_std[:, None, :]**2 + \
                  torch.log(proportions[:, None, :])

            segmentation = torch.softmax(log_num / self.tau_mask, dim=-1)
            chain_segments[f"chain_{n_chain}"] = segmentation


        return mask

    def sample_latent(self, images, indexes=None):
        """
        Samples latent variables given an image
        :param images: torch.tensor(N_batch, N_pix_x, N_pix_y)
        :param indexes: torch.tensor(N_batch, dtype=torch.int) the indexes of images in the batch
        :return: torch.tensor(N_batch, latent_dim) latent variables,
                torch.tensor(N_batch, latent_dim) latent_mean,
                torch.tensor(N_batch, latent_dim) latent std

        """
        if self.amortized:
            assert indexes, "If using a non-amortized version of the code, the indexes of the images must be provided"
            latent_variables = torch.randn_like(self.latent_variables_mean, dtype=torch.float32, device=self.device)*self.latent_variables_std[indexes, :] + self.latent_variables_mean[indexes, :]
            return latent_variables, self.latent_variables_mean[indexes, :], self.latent_variables_std[indexes, :] 
        else:
            latent_mean, latent_std = self.encoder(images)
            latent_variables = latent_mean + torch.randn_like(latent_mean, dtype=torch.float32, device=self.device)\
                                *latent_std

            return latent_variables, latent_mean, latent_std


    def decode(self, latent_variables):
        """
        Decode the latent variables
        :param latent_variables: torch.tensor(N_batch, latent_dim)
        :return: torch.tensor(N_batch, N_domains, 4) quaternions, torch.tensor(N_batch, N_domains, 3) translations
                OR torch.tensor(N_latent_dim, N_domains, 4)
        """
        N_batch = latent_variables.shape[0]
        transformations = self.decoder(latent_variables)
        transformations_per_domain = torch.reshape(transformations, (N_batch, self.total_n_domains, 6))
        ones = torch.ones(size=(N_batch, self.total_n_domains, 1), device=self.device)
        quaternions_per_domain = torch.concat([ones, transformations_per_domain[:, :, 3:]], dim=-1)
        translations_per_domain = transformations_per_domain[:, :, :3]

        quaternions_per_domain_per_chain = {}
        translations_per_domain_per_chain = {}
        n_domains = 0
        for n_chain in self.chain_ids:
            n_dom_chain = self.N_domains[f"chain_{n_chain}"]
            quaternions_per_domain_per_chain[f"chain_{n_chain}"] = quaternions_per_domain[:, n_domains:n_domains+n_dom_chain, :]
            translations_per_domain_per_chain[f"chain_{n_chain}"] = translations_per_domain[:, n_domains:n_domains+n_dom_chain, :]
            n_domains += n_dom_chain


        return quaternions_per_domain_per_chain, translations_per_domain_per_chain



