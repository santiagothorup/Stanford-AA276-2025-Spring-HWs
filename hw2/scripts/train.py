print('Do you want to use WandB for logging training progress (highly recommended)? (y/n): ')
use_wandb = input() == 'y'
if use_wandb:
    import wandb
    wandb.login()
    print('USING WANDB - THE URL TO ACCESS LOGS WILL BE PRINTED SHORTLY')

import sys
import subprocess

# subprocess.run([sys.executable, '../libraries/DeepReach_MPC/run_experiment.py'])
# TODO FOR COURSE STAFF: implement by EOD Saturday 26th