import torch
import numpy as np
from operator import itemgetter
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


class VAE(torch.nn.Module):
    def __init__(self, device, mask_start_values, N_images=150000, N_domains=6, N_residues=1006, tau_mask=0.05,
                 latent_dim = None, latent_type="continuous"):
        super(VAE, self).__init__()
        assert latent_type in ["continuous", "categorical"]
        self.device = device
        self.N_domains = N_domains
        self.N_residues = N_residues
        self.tau_mask = tau_mask
        self.latent_type = latent_type
        self.latent_dim = latent_dim

        self.residues = torch.arange(0, self.N_residues, 1, dtype=torch.float32, device=device)[:, None]

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


        if mask_start_values["type"] == "uniform":
            bound_0 = self.N_residues/N_domains
            self.mask_means_mean = torch.nn.Parameter(data=torch.tensor(np.array([bound_0/2 + i*bound_0 for i in range(N_domains)]), dtype=torch.float32, device=device)[None, :],
                                                      requires_grad=True)
            self.mask_means_std = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains)*10.0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)
            self.mask_std_mean = torch.nn.Parameter(data= torch.tensor(np.ones(N_domains)*bound_0, dtype=torch.float32, device=device)[None,:],
                                                    requires_grad=True)

            self.mask_std_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains) * 10.0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.mask_proportions_mean = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains) * 0, dtype=torch.float32, device=device)[None, :],
                requires_grad=True)

            self.mask_proportions_std = torch.nn.Parameter(
                data=torch.tensor(np.ones(N_domains), dtype=torch.float32, device=device)[None, :],
                requires_grad=True)



        else:
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


        #self.translation_per_domain = torch.nn.ParameterList([torch.nn.Parameter(data=torch.zeros((N_domains, 3), dtype=torch.float32, device=self.device), requires_grad=True) 
        #                                    for i in range(N_images)])

        #self.rotation_per_domain = torch.nn.ParameterList([torch.nn.Parameter(data=torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32, device=self.device).repeat(N_domains, 1), requires_grad=True)
        #                                     for i in range(N_images)])

        self.mean_translation_per_domain = torch.nn.Parameter(data=torch.zeros((N_images, N_domains, 3), dtype=torch.float32, device=self.device), requires_grad=True)
        self.std_translation_per_domain = torch.nn.Parameter(data=torch.ones((N_images, N_domains, 3), dtype=torch.float32, device=self.device), requires_grad=True) 
        self.mean_rotation_per_domain = torch.nn.Parameter(data=torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.float32, device=self.device).repeat(N_images, N_domains, 1), requires_grad=True)
        self.std_rotation_per_domain = torch.nn.Parameter(data=torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32, device=self.device).repeat(N_images, N_domains, 1), requires_grad=True)

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

    def sample_transformations(self, indexes):
        """
        Sample transformations from the approximate posterior
        """
        ## We first sample in R**3
        noise_rotation = torch.randn(size=(len(indexes), self.N_domains, 3), dtype=torch.float32, device=self.device)*self.elu(self.std_rotation_per_domain[indexes])+1
        #Next we map this sample to so(3) and use Rodrigues formula to map to SO(3).
        theta = noise_rotation.norm(p=2, dim=-1, keepdim=True)
        noise_rotation_normalized = noise_rotation/theta
        noise_lie_algebra = noise_rotation_normalized[:, :, 0, None, None]*self.lie_alg_l1 + noise_rotation_normalized[:, :, 1, None, None]*self.lie_alg_l2 + noise_rotation_normalized[:, :, 2, None, None]*self.lie_alg_l3
        #Noise matrix is tensor (N_batch, N_domains, 3, 3)
        noise_matrix = torch.eye(3, dtype=torch.float32, device=self.device)[None, None, :, :] + torch.sin(theta[...,None])*noise_lie_algebra + (1-torch.cos(theta[..., None]))*torch.einsum("bdij,bdjk->bdik", noise_lie_algebra, noise_lie_algebra)
        #mean_rotation_matrix is (N_batch, N_domains, 3, 3)
        mean_rotation_matrix = rotation_6d_to_matrix(self.mean_rotation_per_domain[indexes])
        sampled_rotation_matrix = torch.einsum("bdij, bdjk->bdik", mean_rotation_matrix, noise_matrix)

        sampled_translation = torch.randn_like(self.mean_translation_per_domain[indexes], dtype=torch.float32, device=self.device)*self.std_translation_per_domain[indexes] + self.mean_translation_per_domain[indexes]
        return sampled_translation, sampled_rotation_matrix




