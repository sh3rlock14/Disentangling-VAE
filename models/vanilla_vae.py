import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):

    def __init__(self,
                in_channels: int,
                latent_dim: int,
                hidden_dims: List = None,
                **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512] # Create a small model just for debug purposes


        # Build Encoder
        for h_dim in hidden_dims[0:3]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                            out_channels=h_dim,
                            kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        
        
        for h_dim in hidden_dims[3:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                    out_channels=in_channels * 3, # substitute 3 with an arbitrary integer 'k'
                    groups=in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1),
                    nn.Conv2d(in_channels * 3, h_dim, kernel_size=1),
                )
            )
            in_channels = h_dim
        
        
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []


    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        print(result.shape)

        result = torch.flatten(result, start_dim=1)
        #print("shape", result.shape)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu 
    

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [z, input, mu, log_var]
