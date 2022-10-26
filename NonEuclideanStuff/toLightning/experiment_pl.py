from turtle import forward
import wandb
from torch import optim
from torch import stack, cat, mean, sum, exp, rand, zeros
from .types_ import *
#from utils import data_loader
import pytorch_lightning as pl
#from torchvision import transforms
#from torchvision.utils import make_grid
#from torch.utils.data import DataLoader
from .pointnet_vae_pl import PointNetVAE

from misc.utils import pairwise_dists
from utils.distances import calc_euclidean_dist_matrix, distance_GIH

NUM_ITERS = 0

### LOSSES ###

# reconstruction loss
# 1. Compute loss for global Euclidean distortions on data points
def euclid_recon_loss(model, batch, beta=1, latent_dim=256):
    # TODO: check if this is a viable practice instead of calling model.train()
    x = stack([d.pos for d in batch])

    recons = model(x)

    De_t = pairwise_dists(x)
    De_r = pairwise_dists(recons)

    distortion_loss = mean( (De_t - De_r)**2)

    kl_loss = -0.5 * mean(1 + model.z_var - exp(model.z_var) - model.z_mu**2)

    N = x.shape[1] # Input dimensionality
    M = model.z_mu.shape[1] # Dimensionality of latent space
    b_norm = (M/N) * beta

    loss = distortion_loss + b_norm * kl_loss

    return loss


# 2. 
def interpolation_loss(model, batch, config, geoloss=1e1, eucloss=1e1):
    #model.enable_bn(False)

    # TODO: check if these to(device) are actually necessary
    x = stack([d.pos for d in batch])
    orix = stack([d.oripos for d in batch])
    
    Dg = stack([d.Dg for d in batch])
    De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['latent_dim']]

    device = x.device

    a = rand(1, device=device)
    Dg_t = Dg[0] * a + (1-a) * Dg[1]
    De_t = De[0] * a + (1-a) * De[1]

    #latent_t = latent[0:1] * a + (1-a) * latent[1:2]
    latent_t = latent[0] * a + (1-a) * latent[1]


    rec = model.decoder(latent_t.unsqueeze(dim=0)) # make it a batch

    #loss1 = zeros(1, device=device)
    #loss2 = zeros(1, device=device)
    loss = 0.

    if geoloss > 0:
        T = batch[0].faces[None, ...]
        Dg_r, _, _, _, _, _ = distance_GIH(rec, T)
        #loss1 = geoloss * mean( (Dg_t - Dg_r.float())**2 )
        loss += geoloss * mean( (Dg_t - Dg_r.float())**2 )
    
    if eucloss > 0:
        local_mask = (Dg_t < config['local_th'] * Dg_t.max()).float()
        De_r = calc_euclidean_dist_matrix(rec)
        #loss2 = eucloss * sum( local_mask * ((De_t - De_r)/(De_t + 1e-3))**2 ) / sum(local_mask)
        loss += eucloss * sum( local_mask * ((De_t - De_r)/(De_t + 1e-3))**2 ) / sum(local_mask)
    
    #loss = loss1 + loss2
    return loss


# 3.
def ext_disentanglement_loss(model, batch, config, geoloss=1e0, eucloss=1e0):
    #model.enable_bn(False)

    #data = batch_loader.__iter__().next()
    T = batch[0].faces[None, ...]
    
    device = T.device

    x = stack([d.pos for d in batch])
    #orix = stack([d.oripos for d in batch])
    Dg = stack([d.Dg for d in batch])

    #De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['latent_dim']]

    a = rand(1, device=device)#.to(device)
    Dg_t = Dg[0]

    #loss1 = zeros(1).to(device)
    #loss2 = zeros(1).to(device)

    #localmask =  (Dg_t < config['exp_params']['local_th'] * Dg_t.max()).float()

    # Interpolate only the portion of (different subjects') latent vectors encoding the POSE =>
    #   => this way the geodesic distances SHOULD be preserved

    latent_t = cat([ latent[0:1, :config['pose_dim']] * a + (1-a) * latent[1:2, :config['pose_dim']], latent[0:1, config['pose_dim']:]], -1)
    rec = model.decoder(latent_t)

    Dg_r, _, _, _, _, _ = distance_GIH(rec, T)
    loss = geoloss * mean( ((Dg_t - Dg_r.float()))**2)

    return loss



# 4.
def int_disentanglement_loss(model, batch, config, geoloss=1e0, eucloss=1e0):
    #model.enable_bn(False)

    #data = batch_loader.__iter__().next()
    #T = data[0].faces[None, ...].to(device)

    x = stack([d.pos for d in batch])
    
    device = x.device

    orix = stack([d.oripos for d in batch])
    
    #Dg = stack([d.Dg for d in data]).to(device)
    De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['latent_dim']]

    a = rand(1, device=device)
    #Dg_t = Dg[0]

    #loss1 = zeros(1).to(device)
    #loss2 = zeros(1).to(device)

    #localmask = (Dg_t < config['exp_params']['local_th'] * Dg_t.max()).float()

    # Interpolating the portion of SAME identity's latent vector encoding DIFFERENT styles SHOULD not result in changes on the embedding
    latent_t = cat( [latent[0:1, :config['pose_dim']], latent[0:1, config['pose_dim']:] *  a + (1-a) * latent[1:2, config['pose_dim']:]], -1)
    rec = model.decoder(latent_t)

    De_r = calc_euclidean_dist_matrix(rec)

    loss = eucloss * mean( ( (De[0:1] - De_r)/(De[0:1] + 1e-3) )**2 ) # L2 loss on the relative euclidean distortion
    
    return loss




### EXPERIMENT ###




class DLAIExperiment(pl.LightningModule):

    def __init__(self,
                model: PointNetVAE,
                params: dict) -> None:
                
                super(DLAIExperiment, self).__init__()

                # FOR MANUAL OPTIMIZATION set to False
                self.automatic_optimization=True

                self.save_hyperparameters(ignore=['model'])

                self.model = model
                self.params = params
                self.curr_device = None
                self.hold_graph = False

                try:
                    self.hold_graph = self.params['retain_first_packpass']
                except:
                    pass
    

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    
    def training_step(self, batch_dict, batch_idx, optimizer_idx = 0):
        
        global NUM_ITERS

        batch_rec = batch_dict["rec"]
        loss_keys = ['train_loss', 'loss_rec', 'loss_interp', 'loss_int_dis', 'loss_ext_dis']
        #losses = []

        #loss1 = loss2 = loss3 = loss4 = zeros((1,)).float().to(self.params['curr_device'])
        loss1 = euclid_recon_loss(self.model, batch_rec, self.params['beta'], self.params['latent_dim'])

    
        if self.global_step % self.params['vbs'] == 0:
            NUM_ITERS +=1
    
    

        if NUM_ITERS >= self.params['opt_procedure_ths'][0] and NUM_ITERS <= self.params['opt_procedure_ths'][1]:
            
            batch_interp = batch_dict["interp"]
            batch_intra = batch_dict["intra"]

            loss2 = interpolation_loss(self.model, batch_interp, self.params, 0, 1e1)
            loss3 = int_disentanglement_loss(self.model, batch_intra, self.params, 0, 1e1)

            loss = loss1 + 1e-1*loss2 + 1e1*(loss3)

            loss_dict = dict(zip(loss_keys, [loss.item(), loss1.item(), loss2.item(), loss3.item()] ))
            self.log_dict(loss_dict, rank_zero_only=True)

            return loss
        
        elif NUM_ITERS > self.params['opt_procedure_ths'][1] and NUM_ITERS <= self.params['opt_procedure_ths'][2]:
            
            batch_interp = batch_dict["interp"]
            batch_intra = batch_dict["intra"]

            loss2 = interpolation_loss(self.model, batch_interp, self.params, 1e-2, 1e2)
            loss3 = int_disentanglement_loss(self.model, batch_intra, self.params, 1e-2, 1e1)
            loss4 = ext_disentanglement_loss(self.model, batch_interp, self.params, 1e1)

            loss = loss1 + 1e-1*loss2 + 1e1*(1e0*loss3 + 1e0*loss4)

            loss_dict = dict(zip(loss_keys, [loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()] ))
            self.log_dict(loss_dict, rank_zero_only=True)

            return loss

        elif NUM_ITERS > self.params['opt_procedure_ths'][2]:
            
            batch_interp = batch_dict["interp"]
            batch_intra = batch_dict["intra"]

            loss2 = interpolation_loss(self.model, batch_interp, self.params, 1e-3, 1e2)
            loss3 = int_disentanglement_loss(self.model, batch_intra, self.params, 1e-3, 1e0)
            loss4 = ext_disentanglement_loss(self.model, batch_interp, self.params, 1e0)

            loss = loss1 + 1e-1*loss2 + 1e1*(1e0*loss3 + 1e0*loss4)

            loss_dict = dict(zip(loss_keys, [loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()] ))
            self.log_dict(loss_dict, rank_zero_only=True)


            return loss
        
        # BY DEFAULT
        #losses.extend(loss1)

        loss_dict = dict(zip([loss_keys[1]], [loss1.item()] ))
        self.log_dict(loss_dict, rank_zero_only=True)


        return loss1

    
    
    def configure_optimizers(self):

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'])
        return optimizer

