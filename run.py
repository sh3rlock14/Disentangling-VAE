import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/debug.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)


if __name__ == '__main__':
    
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'],True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    #experiment = VAEXperiment(model, config['exp_params'])

    data = VAEDataset(**config['data_params'], pin_memory=config['trainer_params']['devices'] != 0)
    
    data.setup() # CALL IT MANUALLY JUST FOR DEBUGGING. THEN IT CAN BE REMOVED
    #trainer = Trainer(**config['trainer_params'])

    model.eval()
    with torch.no_grad():
        #images = torch.randn(1, 3, 64, 64)
        train_loader = data.train_dataloader()
        imgs_batch = next(iter(train_loader))[0] # Get a batch of images
        results = model(imgs_batch)
        print(results[0].shape, len(results))
    