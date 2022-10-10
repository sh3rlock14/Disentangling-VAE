from .base import *
from .vanilla_vae import *
from .beta_vae import *


# Aliases
VAE = VanillaVAE

vae_models = {
    'VanillaVAE': VanillaVAE,
    'BetaVAE': BetaVAE,
    'BetaVAE_Burgess': BetaVAE_Burgess,
}