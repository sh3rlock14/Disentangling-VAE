from models import BaseVAE
from numpy import product
from torch import nn, mean, prod, flatten, sum, exp, randn_like, clamp, randn, Tensor
from torch import Tensor as TTensor
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_
from .types_ import *


class BetaVAE(BaseVAE):


    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = TTensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                            out_channels = h_dim,
                            kernel_size = 4,
                            stride = 2,
                            padding = 1),
                    #nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        #self.weight_init()

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    
    def weight_init(self):
        for block in self._modules:
            if block in ['encoder', 'decoder', 'final_layer']:
                for m in self._modules[block]: 
                    self.kaiming_init(m)
            

    def kaiming_init(self, m):
        pass
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)



    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = exp(0.5 * logvar)
        eps = randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = mean(-0.5 * sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



class BetaVAE_Burgess(BaseVAE):

    num_iter = 0

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE_Burgess, self).__init__()

        self.latent_dim = latent_dim
        self.kernel_size = 4
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = TTensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32, 256, 256]
        
        # Build Encoder
        
        self.reshape = (hidden_dims[3], self.kernel_size, self.kernel_size)

        # Convolutional Part
        in_chs = in_channels
        for h_dim in hidden_dims[0:4]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_chs,
                        out_channels = h_dim,
                        kernel_size = 4,
                        stride = 2,
                        padding = 1
                    ),
                    nn.ReLU())
                    )
            in_chs = h_dim

        # Flatten
        modules.append(
            nn.Sequential(
                nn.Flatten(),
        )
        )
        
        # Linear Part
        in_fts = int(product(self.reshape))
        for h_dim in hidden_dims[4:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(
                        in_fts,
                        out_features = h_dim),
                    nn.ReLU()
                ),
                
            )
            in_fts = h_dim
        
        
        self.encoder = nn.Sequential(*modules)
        
        # Linear Part - mean and var
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)



        # Build Decoder
        modules = []

        #self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        # Linear Part

        in_fts = latent_dim
        for i in range(2):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_fts,
                              hidden_dims[i]),
                    nn.ReLU()
                )
            )
            in_fts = hidden_dims[i]
        
        
        modules.append(nn.Sequential(
            nn.Linear(in_fts, int(product(self.reshape))),
        ))
        

        # Unflatten

        modules.append(nn.Sequential(
            nn.Unflatten(1, self.reshape),
        ))

        

        # Convolutional Part

        conv_hidden_dims = hidden_dims[2:]
    
        for i in range(len(conv_hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels = conv_hidden_dims[i],
                        out_channels = conv_hidden_dims[i+1],
                        kernel_size = 4,
                        stride = 2,
                        padding = 1
                    ),
                    nn.ReLU()
                )
            )

        
        
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], 
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1),
            nn.Sigmoid(),
        )

    

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        #print(result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if self.training:
            std = exp(0.5 * logvar)
            eps = randn_like(std)
            return eps * std + mu
        else: 
            # Reconstruction mode
            return mu
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]