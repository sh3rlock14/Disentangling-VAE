import os
import yaml
import argparse
import wandb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
                

# Define args
parser = argparse.ArgumentParser(description='Generic runner for VAE models')

parser.add_argument('--sweep_config', '-sc',
                    dest="sweep_config",
                    metavar='FILE',
                    help= 'path to the sweep config file',
                    default = 'NonEuclideanStuff/sweep_configs/sweep_debug.yaml')


args = parser.parse_args()


# Define sweep config
# sweep config
with open(args.sweep_config, 'r') as file:
    try:
        sweep_config = yaml.safe_load(file)
        print(sweep_config)
    except yaml.YAMLError as exc:
        print(exc)

#wandb.login(key="b2f43af624f34e36163a25d9c7a60d3385d7d46a")



# Initialize the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project="DLAI AA 2022 - Disentangling VAE")

# 🐝 Step 4: Call to `wandb.agent` to start a sweep
wandb.agent(sweep_id, count=1)
