import os
import argparse
import collections
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from pprint import pprint

# model import
from toLightning.pointnet_vae_pl import PointNetVAE

# data import
from toLightning.dataset_pl import VAEDataset

# training import
from toLightning.experiment_pl import DLAIExperiment
from pytorch_lightning import Trainer

# callbacks import
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def configToDict(cfg):

    cfgDict = {}

    for p in cfg.split("--"):
        k,v = p.split("=")
        cfgDict.update({k:v})

    return cfgDict

def deep_update(base_cfg, sweep_cfg):
    """
    Update a nested dict or similar mapping.
    Modify :source: in place.
    seen in: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """

    for k, v in sweep_cfg.items():
        if isinstance(v, collections.abc.Mapping) and v:
            returned = deep_update(base_cfg.get(k, {}), v)
            base_cfg[k] = returned
        else:
            base_cfg[k] = sweep_cfg[k]
    return base_cfg

def configToDict(cfg):

    cfgDict = {}

    for p in cfg.split("--"):
        try:
            k,v = p.split("=")
            cfgDict.update({k:eval(v)})
        except:
            pass
        

    return cfgDict



parser = argparse.ArgumentParser(description="DLAI NE GVAE Experiment")

# load default parameters
parser.add_argument('--base_config', '-c',
                        dest="base_config",
                        metavar='FILE',
                        help= 'path to the sweep config file',
                        default = 'NonEuclideanStuff/configs/debug.yaml')
    


args, unknown = parser.parse_known_args()


# Access the configs
with open(args.base_config, 'r') as file:
        try:
            base_config = yaml.safe_load(file)
            #print(base_config)
        except yaml.YAMLError as exc:
            print(exc)



base_config = deep_update(base_config, configToDict(unknown[0]))

"""
with open(args.sweep_config, 'r') as file:
        try:
            sweep_config = yaml.safe_load(file)
            print(sweep_config)
        except yaml.YAMLError as exc:
            print(exc)

"""
    
wandb.init(config=base_config)


# Update wandb config
wandb.config = deep_update(base_config, wandb.config)
#wandb.config.update(deep_update(base_config, wandb.config))



def main():

    #print(wandb.config)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = PointNetVAE(wandb.config['model_params'])

    # ------------------------
    # 2 DATA PIPELINE
    # ------------------------
    data = VAEDataset(**wandb.config['data_params'])

    # ------------------------
    # 3 WANDB LOGGER
    # ------------------------
    wandb_logger = WandbLogger(
        project = wandb.config['logging_params']['wandb_project'],
        save_dir = wandb.config['logging_params']['save_dir'],
        log_model = "all" # log while training
    )

    # ------------------------
    # 4 LIGHTNING EXPERIMENT
    # ------------------------
    experiment = DLAIExperiment(model, wandb.config['exp_params'])

    # ------------------------
    # 5 TRAINER
    # ------------------------
    
    # ckpt formatting
    model_ckpt_name = wandb.config['logging_params']['model_name']
    NUM_ITERS = wandb.config['exp_params']['num_iters'] * wandb.config['exp_params']['vbs']
    NUM_MODELS_TO_SAVE = 10   

    trainer = Trainer(
        log_every_n_steps=100,
        logger = wandb_logger,
        callbacks = [
            ModelCheckpoint(
                            save_top_k=2,
                            dirpath = os.path.join(wandb_logger.save_dir, "checkpoints"),
                            filename = model_ckpt_name + '-{step}-{loss_rec:.2f}', 
                            monitor = "loss_rec",
                            mode = "min",
                            every_n_train_steps = NUM_ITERS // NUM_MODELS_TO_SAVE,
                            save_on_train_epoch_end=True # cannot be False since there's no Validation
                        ),
                        LearningRateMonitor(logging_interval='epoch')
        ],
        max_steps= NUM_ITERS,
        num_sanity_val_steps=0,
        limit_val_batches=0,  # disable validation
        **wandb.config['trainer_params']
    )

    
    if wandb.config['training_params']['resume_train']:
        assert wandb.config['training_params']['ckpt_path'] is not None, "To resume training, you need to specify the ckpth path"
        print(f"======= Resuming Training {wandb.config['model_params']['name']} from {wandb.config['training_params']['ckpt_path']} =======")
    else:
        print(f"======= Training {wandb.config['model_params']['model_name']} from scratch =======")
    

    # START TRAINING
    trainer.fit(
        experiment,
        datamodule = data,
        ckpt_path = wandb.config['training_params']['ckpt_path']
    )



if __name__ == '__main__':

    print(f'Starting a run with {wandb.config}')
    main()