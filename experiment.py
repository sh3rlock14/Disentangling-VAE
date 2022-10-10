#import os
#import math
#import torch
import wandb
from torch import optim
from models import BaseVAE
from models.types_ import *
#from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                vae_model: BaseVAE,
                params: dict) -> None:
        super(VAEXperiment, self).__init__()

        #self.save_hyperparameters()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False 
        try:
            self.hold_graph = self.params['retain_first_packpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx = 0):

        imgs, labels = batch
        self.curr_device = imgs.device

        imgs_recon = self.forward(imgs, labels = labels)
        train_loss = self.model.loss_function(*imgs_recon,
                                              M_N = self.params['kld_weight'],
                                              optimizer_idx = optimizer_idx,
                                              batch_idx = batch_idx)
        
        # Log loss and metric
        #self.log('train_loss', train_loss['loss'])
        #self.log('train_recon_loss', train_loss['Reconstruction_Loss']) 
        #self.log('train_KLD', train_loss['KLD'])
        self.log_dict({f"train_{key}": val.item() for key, val in train_loss.items()}, sync_dist=False)

        return train_loss['loss']
    
    def validation_step(self, batch, batch_idx, optimizer_idx =  0):

        imgs, labels = batch
        self.curr_device = imgs.device

        imgs_recon = self.forward(imgs, labels = labels)
        val_loss = self.model.loss_function(*imgs_recon,
                                              M_N = 1.0,
                                              optimizer_idx = optimizer_idx,
                                              batch_idx = batch_idx)
        
        # Log loss and metric
        #self.log('val_loss', val_loss['loss'])
        #self.log('val_recon_loss', val_loss['Reconstruction_Loss']) 
        #self.log('val_KLD', val_loss['KLD'])
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=False)


    
    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        imgs_recon = self.model.generate(test_input, labels = test_label)
        
        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
        except Warning:
            pass

        
        # Log images
        wandb.log({"original" : wandb.Image(make_grid(test_input, nrow=12).data)}, commit=False)
        wandb.log({"recon" : wandb.Image(make_grid(imgs_recon, nrow=12).data)},    commit=False)
        
        try:
            wandb.log({"samples" : wandb.Image(make_grid(samples, nrow=12).data)})
        except:
            wandb.log()

        """
        self.logger.log_image(key="original", images = [make_grid(test_input, nrow=12).data], commit=False) # DA TESTARE
        self.logger.log_image(key="recon", images = [make_grid(imgs_recon, nrow=12).data], commit=False)
        
        try:
            self.logger.log_image(key="samples", images = [make_grid(samples, nrow=12).data])
        except Warning:
            pass

        """
        

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                            lr=self.params['LR'],
                            weight_decay = self.params['weight_decay'])
        optims.append(optimizer)

        # Check if more than 1 optimizer is required (Used for GAN)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
