import os
import argparse
import collections
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb


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


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
            print(base_config)
        except yaml.YAMLError as exc:
            print(exc)

"""
with open(args.sweep_config, 'r') as file:
        try:
            sweep_config = yaml.safe_load(file)
            print(sweep_config)
        except yaml.YAMLError as exc:
            print(exc)

"""

if base_config['logging_params']['enable_wandb']:
    
    #wandb.login(key = base_config['logging_params']['wandb_key'] )
    wandb.init(
    #    entity = base_config['logging_params']['wandb_entity'],
    #    project= base_config['logging_params']['wandb_project'],
        config = base_config)
        
    wandb_logger = WandbLogger(
            project = base_config['logging_params']['wandb_project'],
            save_dir = base_config['logging_params']['save_dir'],
            log_model = "all" # log while training
        )


# Update wandb config
wandb.config = deep_update(base_config, wandb.config)


print(wandb.config)









    