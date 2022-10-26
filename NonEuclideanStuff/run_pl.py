if __name__ == '__main__':

    import os
    import yaml
    import argparse
    #from pathlib import Path

    # seeder
    from pytorch_lightning.utilities.seed import seed_everything

    # data import
    from toLightning.dataset_pl import VAEDataset

    # model import
    from toLightning.pointnet_vae_pl import PointNetVAE

    # training import
    from toLightning.experiment_pl import DLAIExperiment
    from pytorch_lightning import Trainer


    # callbacks import
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help= 'path to the config file',
                        default = 'NonEuclideanStuff/configs/debug.yaml')
    
    args = parser.parse_args()
    
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    
    
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    # wandb setting
    KEY = "b2f43af624f34e36163a25d9c7a60d3385d7d46a"
    user = 'mattiacapparella'
    project = "DLAI AA 2022 - Disentangling VAE"

    if config['logging_params']['enable_wandb']:
        wandb.login(key=KEY)
        wandb.init(entity=user, project=project, config=config)
        wandb_logger = WandbLogger(
            project = project,
            save_dir = config['logging_params']['save_dir'],
            log_model = "all" # log while training
        )

    print(wandb.config)


    # ckpt formatting
    model_ckpt_name = config['logging_params']['model_name']

    

    model = PointNetVAE(config['model_params']) # config è un campo del modello
    data = VAEDataset(**config['data_params'])
    
    ### DEBUG ###
    #data.setup()
    #data.train_dataloader()

    ### END DEBUG ###
    
    experiment = DLAIExperiment(model, config['exp_params'])


    NUM_ITERS = config['exp_params']['num_iters'] * config['exp_params']['vbs'] 
    NUM_MODELS_TO_SAVE = 10


    trainer = Trainer(
                    log_every_n_steps= 100,
                    logger = wandb_logger,
                    callbacks = [
                        #EarlyStopping(
                        #    monitor = 'loss_rec',
                        #    patience = config['training_params']['patience'],
                        #    mode = 'min',
                        #    check_finite = True,
                        #    check_on_train_epoch_end = True

                        #),
                        ModelCheckpoint(
                            save_top_k=2,
                            dirpath = os.path.join(wandb_logger.save_dir, "checkpoints"),
                            filename = model_ckpt_name + '-{step}-{train_loss:.2f}', 
                            monitor = "loss_rec",
                            mode = "min",
                            every_n_train_steps = NUM_ITERS // NUM_MODELS_TO_SAVE,
                            save_on_train_epoch_end=True # cannot be False since there's no Validation
                        ),
                        LearningRateMonitor(logging_interval='epoch')
                    ],

                    max_steps= NUM_ITERS,
                    num_sanity_val_steps=0,
                    limit_val_batches=0, # disable validation
                    **config['trainer_params'])


    if config['training_params']['resume_train']:
            assert config['training_params']['ckpt_path'] is not None, "To resume training, you need to specify the ckpth path"
            
            print(f"======= Resuming Training {config['model_params']['name']} from {config['training_params']['ckpt_path']} =======")
    else:
        print(f"======= Training {config['model_params']['model_name']} from scratch =======")
        
    
    trainer.fit(experiment,
                datamodule= data,
                ckpt_path= config['training_params']['ckpt_path'])
    
    

    