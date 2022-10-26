from torch import rand_like
from .base_pointnet_pl import *
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv1d, BatchNorm1d, ReLU, LeakyReLU, Linear
import torch.nn.functional as F
from torch import exp, bmm

import pytorch_lightning as pl

class Encoder(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        self.ptnet = BasePointNetPL(config['latent_dim']*2,
                                  config['conv_out_dim'],
                                  config['conv_layers_size'],
                                  config['fc_layers_size'],
                                  config['transformers_position'],
                                  )
    
    def forward(self, x):
        x = self.ptnet(x)
        return x
    

class Decoder(pl.LightningModule):
    def __init__(self, config):
        pl.LightningModule.__init__(self)
        
        self.num_points = config['num_points']

        self.fc1 = Sequential(
            Linear(config['latent_dim'], 1024),
            LeakyReLU(),
            Linear(1024, 2048),
            LeakyReLU(),
            Linear(2048, self.num_points * 3),
        )

    def forward(self, x):
        x = self.fc1(x).view(x.shape[0], self.num_points, -1)
        return x


class PointNetVAE(pl.LightningModule):

    def __init__(self, config, **kwargs):
        pl.LightningModule.__init__(self)

        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, x):

        latent = self.encoder(x)

        if self.train:
            # TO DO: try to split the returning values of the encoder 
            self.z_mu = latent[..., :self.config['latent_dim']]
            self.z_var = latent[..., self.config['latent_dim']:]
            std = exp(self.z_var / 2)
            eps = rand_like(std)

            latent = eps.mul(std).add_(self.z_mu)
        
        return self.decoder(latent[..., :self.config['latent_dim']])