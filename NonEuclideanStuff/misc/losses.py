from torch import stack, cat, mean, sum, exp, rand, zeros
from torch.nn.modules import loss
from misc.losses import *
from misc.utils import pairwise_dists
from utils.distances import calc_euclidean_dist_matrix, distance_GIH
# Define 3 losses:

# 1. Compute loss for global Euclidean distortions on data points
def euclid_recon_loss(model, batch_loader, device):
    # TODO: check if this is a viable practice instead of calling model.train()
    model.enable_bn(True)

    data = batch_loader.__iter__().next()

    x = stack([d.pos for d in data]).to(device)

    recons = model(x)

    De_t = pairwise_dists(x)
    De_r = pairwise_dists(recons)

    distortion_loss = mean( (De_t - De_r)**2)

    kl_loss = -0.5 * mean(1 + model.z_var - exp(model.z_var) - model.z_mu**2)

    loss = distortion_loss + kl_loss

    return loss


# 2. 
def interpolation_loss(model, batch_loader, device, config, geoloss=1e1, eucloss=1e1):
    model.enable_bn(False)

    data = batch_loader.__iter__().next()
    T = data[0].faces[None, ...].to(device)

    # TODO: check if these to(device) are actually necessary
    x = stack([d.pos for d in data]).to(device)
    orix = stack([d.oripos for d in data]).to(device)
    
    Dg = stack([d.Dg for d in data]).to(device)
    De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['model_params']['latent_dim']]

    a = rand(1).to(device)
    Dg_t = Dg[0] * a + (1-a) * Dg[1]
    De_t = De[0] * a + (1-a) * De[1]

    latent_t = latent[0:1] * a + (1-a) * latent[1:2]

    rec = model.decoder(latent_t)

    loss1 = zeros(1).to(device)
    loss2 = zeros(1).to(device)

    if geoloss > 0:
        Dg_r, grad, div, W, S, C = distance_GIH(rec, T)
        loss1 = geoloss * mean( (Dg_t - Dg_r.float())**2 )
    
    if eucloss > 0:
        local_mask = (Dg_t < config['exp_params']['local_th'] * Dg_t.max()).float()
        De_r = calc_euclidean_dist_matrix(rec)
        loss2 = eucloss * sum( local_mask * ((De_t - De_r)/(De_t + 1e-3))**2 ) / sum(local_mask)
    
    loss = loss1 + loss2
    return loss

# 3.
def ext_disentanglement_loss(model, batch_loader, device, config, geoloss=1e0, eucloss=1e0):
    model.enable_bn(False)

    data = batch_loader.__iter__().next()
    T = data[0].faces[None, ...].to(device)

    x = stack([d.pos for d in data]).to(device)
    orix = stack([d.oripos for d in data]).to(device)
    Dg = stack([d.Dg for d in data]).to(device)

    De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['model_params']['latent_dim']]

    a = rand(1).to(device)
    Dg_t = Dg[0]

    loss1 = zeros(1).to(device)
    loss2 = zeros(1).to(device)

    localmask =  (Dg_t < config['exp_params']['local_th'] * Dg_t.max()).float()

    # Interpolate only the portion of (different subjects') latent vectors encoding the POSE =>
    #   => this way the geodesic distances SHOULD be preserved

    latent_t = cat([ latent[0:1, :config['model_params']['pose_dim']] * a + (1-a) * latent[1:2, :config['model_params']['pose_dim']], latent[0:1, config['model_params']['pose_dim']:]], -1)
    rec = model.decoder(latent_t)

    Dg_r, grad, div, W, S, C = distance_GIH(rec, T)
    loss = geoloss * mean( ((Dg_t - Dg_r.float()))**2)

    return loss


def int_disentanglement_loss(model, batch_loader, device, config, geoloss=1e0, eucloss=1e0):
    model.enable_bn(False)

    data = batch_loader.__iter__().next()
    T = data[0].faces[None, ...].to(device)

    x = stack([d.pos for d in data]).to(device)
    orix = stack([d.oripos for d in data]).to(device)
    
    Dg = stack([d.Dg for d in data]).to(device)
    De = calc_euclidean_dist_matrix(orix)

    latent = model.encoder(x)[..., :config['model_params']['latent_dim']]

    a = rand(1).to(device)
    Dg_t = Dg[0]

    loss1 = zeros(1).to(device)
    loss2 = zeros(1).to(device)

    localmask = (Dg_t < config['exp_params']['local_th'] * Dg_t.max()).float()

    # Interpolating the portion of SAME identity's latent vector encoding DIFFERENT styles SHOULD not result in changes on the embedding
    latent_t = cat( [latent[0:1, :config['model_params']['pose_dim']], latent[0:1, config['model_params']['pose_dim']:] *  a + (1-a) * latent[1:2, config['model_params']['pose_dim']:]], -1)
    rec = model.decoder(latent_t)

    De_r = calc_euclidean_dist_matrix(rec)

    loss = eucloss * mean( ( (De[0:1] - De_r)/(De[0:1] + 1e-3) )**2 ) # L2 loss on the relative euclidean distortion
    
    return loss


