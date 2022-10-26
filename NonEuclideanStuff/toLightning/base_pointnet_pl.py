from torch import bmm, eye
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv1d, BatchNorm1d, ReLU, Linear
import torch.nn.functional as F

import pytorch_lightning as pl

class BasePointNetPL(pl.LightningModule):
    def __init__(self,
                latent_dim: int,
                conv_out_dim: int,
                conv_layers_size,
                fc_layers_size,
                transformers_position,
                end_in_batchnorm=False):

        super(BasePointNetPL, self).__init__()
        self.LD = latent_dim
        self.CD = conv_out_dim
        self.transformers_position = transformers_position

        assert self.CD % 2 == 0, "Convolutional output dimension must be even"

        self._conv_sizes = [3] + [k for k in conv_layers_size] + [self.CD]
        self._fc_sizes = [self.CD] + [k for k in fc_layers_size]

        # Convolutional Layers
        self.conv_layers = ModuleList([
            Sequential(
                Conv1d(self._conv_sizes[i], self._conv_sizes[i+1], 1),
                BatchNorm1d(self._conv_sizes[i+1]),
                ReLU()
            ) for i in range(len(self._conv_sizes)-1)
        ])

        # Transformer Layers
        self.transformers = ModuleList([
            BaseTransformer(self._conv_sizes[jj]) for jj in self.transformers_position

        ])

        # FC layers
        self.fc_layers = ModuleList([
            Sequential(
                Linear(self._fc_sizes[i], self._fc_sizes[i+1]),
                BatchNorm1d(self._fc_sizes[i+1]),
                ReLU()
            ) for i in range(len(self._fc_sizes)-1)] 
            +
            (
                [ Linear(self._fc_sizes[-1], self.LD),
                        BatchNorm1d(self.LD) ] if end_in_batchnorm else [Linear(self._fc_sizes[-1], self.LD)])
                )
        
    def forward(self, x):
        """
        Input: B x N x 3 PCs (not permuted)
        Output: B x LD  embedded shapes
        """

        x = x.permute(0,2,1) # B x 3 x N
        assert x.shape[1] == 3, "Permuted Input must be B x 3 x N"

        for i, layer in enumerate(self.conv_layers):
            if i in self.transformers_position:
                T = self.transformers[self.transformers_position.index(i)](x)
                x = layer(bmm(T, x))
            else:
                x = layer(x)
        
        # Pool over the points
        # x: B x C_D x N -pool-> B x C_D x 1 -squeeze-> B x C_D

        x = F.max_pool1d(x, x.shape[2]).squeeze(2)

        for j, layer in enumerate(self.fc_layers):
            x = layer(x)
        
        return x


class BaseTransformer(pl.LightningModule):

    def __init__(self,
                input_dim,
                conv_dims = (64, 128, 512),
                fc_dims = (512, 256)):

                super(BaseTransformer, self).__init__()


                # Set network dimensions
                self.input_features_dim = input_dim
                self.conv_dims = [self.input_features_dim] + [k for k in conv_dims]
                self.fc_dims = [k for k in fc_dims]

                # Convolutional Layers
                self.conv_layers = ModuleList([
                    Sequential(
                        Conv1d(self.conv_dims[i], self.conv_dims[i+1], 1),
                        BatchNorm1d(self.conv_dims[i+1]),
                        ReLU()
                    ) for i in range(len(self.conv_dims)-1)
                ])


                # Fully Connected Layers
                self.fc_layers = ModuleList([
                    Sequential(
                        Linear(self.fc_dims[i], self.fc_dims[i+1]),
                        ReLU()
                    ) for i in range(len(self.fc_dims)-1)
                ] + [ Linear(self.fc_dims[-1], self.input_features_dim**2) ])


                # Identity Matrix
                self.eye = eye(self.input_features_dim)

    def forward(self,x):
        """
        Input: B x F x N. F = 3 at the beginning (a permuted PCs batch is passed as input)

        Output: B x F x F set of transformation matrices
        """

        SF = x.shape[1] # Size of features per point

        # Convolutional Layers
        # TODO: remove enumerate
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)

        # Max Pooling
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)

        # FC layers
        for j, layer in enumerate(self.fc_layers):
            x = layer(x)

        x = x.view(-1, SF, SF) + self.eye.to(x.device)

        return x            
