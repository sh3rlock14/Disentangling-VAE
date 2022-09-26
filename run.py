if __name__ == '__main__':


    import os
    import yaml
    import wandb
    import argparse
    #import numpy as np
    from pathlib import Path
    from models import *
    from experiment import VAEXperiment
    #import torch.backends.cudnn as cudnn
    from pytorch_lightning import Trainer
    #from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.utilities.seed import seed_everything
    from pytorch_lightning.loggers import WandbLogger
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
    
    # wandb setting
    KEY = "b2f43af624f34e36163a25d9c7a60d3385d7d46a"
    user = 'mattiacapparella'
    project = "DLAI AA 2022 - Disentangling VAE"
    

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'],True)

    model = vae_models[config['model_params']['name']](**config['model_params'])
    data = VAEDataset(**config['data_params'], pin_memory=config['trainer_params']['devices'] != 0)
    #data.setup() # CALL IT MANUALLY JUST FOR DEBUGGING. THEN IT CAN BE REMOVED
    
    experiment = VAEXperiment(model, config['exp_params'])
    
    if config['logging_params']['enable_wandb']:
        wandb.login(key = KEY)
        wandb.init(entity=user, project = project)
        wandb_logger = WandbLogger(
            project = project,
            save_dir = config['logging_params']['save_dir'],
            log_model = True
        )

    trainer = Trainer(
        logger = wandb_logger,
        callbacks = [
        #    ModelCheckpoint(save_top_k=2,
        #                    dirpath = os.path.join(wandb_logger.save_dir, "checkpoints"),
        #                    monitor = "val_loss",
        #                    mode = "min",
        #                    save_on_train_epoch_end= True,         
        #    ),
            LearningRateMonitor(logging_interval='epoch')
            ],
        limit_train_batches= 100,
        limit_val_batches= 50,
        val_check_interval = 0.5,
        fast_dev_run = False,
        **config['trainer_params'])
    
    print(f"======= Training {config['model_params']['name']} =======")
    trainer.fit(experiment,
                datamodule = data)



    """
    model.eval()
    with torch.no_grad():
        #images = torch.randn(1, 3, 64, 64)
        train_loader = data.train_dataloader()
        imgs_batch = next(iter(train_loader))[0] # Get a batch of images
        results = model(imgs_batch)
        print(results[0].shape, len(results))
    """
    