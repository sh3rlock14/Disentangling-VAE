if __name__ == '__main__':
    #import os
    from tqdm import tqdm
    import yaml
    import scipy.sparse as sp
    import torch
    import argparse
    #import dill
    #import pickle
    #import torch.nn as nn
    #from random import shuffle
    
    from torch import save
    from torch.optim import Adam
    #import torch_geometric
    from torch_geometric.loader import DataLoader, DataListLoader


    from models.pointnet_vae import *
    
    from misc.losses import *
    from misc.utils import *
    from misc.samplers import *
    
    from utils.distances import *
    
    from datasets.faust_2500 import Faust2500Dataset


    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='NonEuclideanStuff/configs/debug.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)


    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset_train = Faust2500Dataset(root="D:\LaureaMagistrale\PrimoAnno\Secondo_Semestre\DeepLearning\datasets\FAUST-2500", train=True)

    # Create 3 loaders for the training set
    intra_loader = DataListLoader(dataset_train, batch_size=2, num_workers=0, sampler=IntraSampler(dataset_train))
    interp_loader = DataListLoader(dataset_train, batch_size=2, shuffle=True, num_workers=0)
    rec_loader = DataListLoader(dataset_train, batch_size=config['data_params']['train_batch_size'], shuffle=True, num_workers=0)

    NUM_ITERS = config['exp_params']['num_iters']
    VIRTUAL_BATCH_SIZE = config['exp_params']['vbs']

    # TODO: Implement the model
    model = PointNetVAE(config['model_params']).to(device)

    optimizer = Adam(model.parameters(), lr=config['exp_params']['LR'])

    total_loss = 0
    losses = []
    #t = time.time()


    # Prepare for checkpointing
    os.makedirs('logs/checkpoints/', exist_ok=True)

    print("Starting training...")

    with tqdm(total=NUM_ITERS, desc="Iter") as pbar:
        for i in range(NUM_ITERS+1):

            for vbs_ix in range(VIRTUAL_BATCH_SIZE):
                optimizer.zero_grad()
                loss1 = loss2 = loss3 = loss4 = torch.zeros((1,)).float().to(device)

                loss1 = euclid_recon_loss(model, rec_loader, device)

                if i > config['exp_params']['opt_procedure_ths'][0] and i <= config['exp_params']['opt_procedure_ths'][1]:
                    
                    loss2 = interpolation_loss(model, interp_loader, device, config, 0, 1e1)
                    loss3 = int_disentanglement_loss(model, intra_loader, device, config, 0, 1e1)

                elif i > config['exp_params']['opt_procedure_ths'][1] and i < config['exp_params']['opt_procedure_ths'][2]:
                
                    loss2 = interpolation_loss(model, interp_loader, device, config, 1e-2, 1e2)
                    loss3 = int_disentanglement_loss(model, intra_loader, device, config, 1e-2, 1e1)
                    loss4 = ext_disentanglement_loss(model, interp_loader, device, config, 1e1)

                elif i >= config['exp_params']['opt_procedure_ths'][2]:
                    
                    loss2 = interpolation_loss(model, interp_loader, device, config, 1e-3, 1e2)
                    loss3 = int_disentanglement_loss(model, intra_loader, device, config, 1e-3, 1e0)
                    loss4 = ext_disentanglement_loss(model, interp_loader, device, config, 1e0)
                
                
                loss = loss1 + 1e-1*loss2 + 1e1*(1e0*loss3 + 1e0*loss4)
                #loss = loss1
                loss.backward()
                optimizer.step()

                losses.append([loss.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item()])

            # Update the pbar
            pbar.update(1)

            # Log to terminal
            if i%10 == 0:
                avg_loss = np.mean(losses[-50:], 0)
                tqdm.write("Iter n° {}: Loss: {:.2e} ({:.2e}, {:.2e}, {:.2e}, {:.2e})".format(i, avg_loss[0], avg_loss[1], avg_loss[2], avg_loss[3], avg_loss[4]))

            # Save checkpoint
            if i%100 == 0:
                if i <= 3000:
                    save(model.state_dict(), 'NonEuclideanStuff/logs/checkpoints/{}_{}_iter={}_loss={:.2f}.dict'.format(config['model_params']['model_name'], config['data_params']['dataset'], i, loss.item()))
                elif i <= 4000:
                    save(model.state_dict(), 'NonEuclideanStuff/logs/checkpoints/{}_{}_iter={}_loss={:.2f}_EUCL.dict'.format(config['model_params']['model_name'], config['data_params']['dataset'], i, loss.item()))
                else:
                    save(model.state_dict(), 'NonEuclideanStuff/logs/checkpoints/{}_{}_iter={}_loss={:.2f}_GEOD.dict'.format(config['model_params']['model_name'], config['data_params']['dataset'], i, loss.item()))
        
    print("Training completed!")
