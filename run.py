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
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
    from dataset import VAEDataset

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/bbvae.yaml')

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

    # Ckpt formatting
    model_ckpt_name = config['logging_params']['name']
    

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'],True)

    model = vae_models[config['model_params']['name']](**config['model_params'])


    """
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

    

    if config['training_params']['tune_lr']:
        trainer = Trainer()

        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(experiment,
                                        datamodule=data,
                                        min_lr= config['training_params']['lr_min'],
                                        max_lr= config['training_params']['lr_max'],)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        # TODO
    else:
        trainer = Trainer(
        logger = wandb_logger,
        callbacks = [
            EarlyStopping(
                monitor = 'val_loss',
                patience = config['training_params']['patience'],
                mode = 'min',
                check_finite = True,

            ),
            ModelCheckpoint(save_top_k=2,
                            dirpath = os.path.join(wandb_logger.save_dir, "checkpoints"),
                            filename =  model_ckpt_name + '-{epoch}-{val_loss:.2f}',
                            monitor = "val_loss",
                            mode = "min",
                            save_on_train_epoch_end= False, # If this is False, then the check runs at the end of the validation
                            every_n_epochs= config['training_params']['every_n_epochs'],      
            ),
            LearningRateMonitor(logging_interval='epoch')
            ],

        limit_train_batches= 0.5,
        limit_val_batches= 0.5,
        val_check_interval = 0.5,
        fast_dev_run = False,
        **config['trainer_params'])


        if config['training_params']['resume_train']:
            assert config['training_params']['ckpt_path'] is not None, "To resume training, you need to specify the ckpth path"
            
            print(f"======= Resuming Training {config['model_params']['name']} from {config['training_params']['ckpt_path']} =======")
        else:
            print(f"======= Training {config['model_params']['name']} from scratch =======")
        
        
        trainer.fit(experiment,
                    datamodule = data,
                    ckpt_path=config['training_params']['ckpt_path'])
    """
    



    """
    model.eval()
    with torch.no_grad():
        #images = torch.randn(1, 3, 64, 64)
        train_loader = data.train_dataloader()
        imgs_batch = next(iter(train_loader))[0] # Get a batch of images
        results = model(imgs_batch)
        print(results[0].shape, len(results))
    """
    