if __name__ == '__main__':


    import os
    import yaml
    import wandb
    import argparse
    #import numpy as np
    from pathlib import Path
    
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    import pytorch_lightning as pl

    print(pl.__version__)
