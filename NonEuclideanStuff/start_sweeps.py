import os
import yaml
import argparse
import wandb
import collections

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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




### SWEEP CONFIG VARIABLES ###
sweepConfigPath = "NonEuclideanStuff/sweep_configs/sweep_debug.yaml"
defaultConfigPath = "NonEuclideanStuff/configs/debug.yaml"
projectName = "DLAI AA 2022 - Disentangling VAE"
nTrials = 3





# 🐝 Step 2: Load sweep config
parser = argparse.ArgumentParser(description='Generic runner for VAE models')

parser.add_argument('--sweep_config', '-sc',
                    dest="sweep_config",
                    metavar='FILE',
                    help= 'path to the sweep config file',
                    default = sweepConfigPath)

"""
parser.add_argument('--default_config', '-c',
                    dest="default_config",
                    metavar='FILE',
                    help= 'path to the default config file',
                    default = defaultConfigPath)
"""

args = parser.parse_args()


with open(args.sweep_config, 'r') as file:
    try:
        sweep_config = yaml.safe_load(file)
        print("Sweep config loaded!")
        #print(sweep_config)
    except yaml.YAMLError as exc:
        print(exc)

"""
with open(args.default_config, 'r') as file:
    try:
        default_config = yaml.safe_load(file)
        print("Default config loaded!")
    except yaml.YAMLError as exc:
        print(exc)

sweep_config = deep_update(default_config, sweep_config)
"""

#wandb.login(key="b2f43af624f34e36163a25d9c7a60d3385d7d46a")

# 🐝 Step 3: Initialize sweep by passing in config
sweep_id = wandb.sweep(sweep=sweep_config, project = projectName)

# 🐝 Step 4: Call to `wandb.agent` to start a sweep
wandb.agent(sweep_id, count = nTrials)
