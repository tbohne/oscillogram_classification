import wandb
import os
import train
import argparse

from train import file_path
from configs import api_key, example_sweep_config

TUNING_METHOD = 'random'
N_RUNS_IN_SWEEP = 3

def main():

    def call_training_procedure(config_dict=None):
        return train.train_procedure(args.train_path, args.val_path, args.test_path, hyperparameter_config=config_dict)

    parser = argparse.ArgumentParser(description='Hyperparametertunig with Weights&biases sweep')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = api_key.wandb_api_key
    sweep_config = {'method': TUNING_METHOD, 'parameters': example_sweep_config.sweep_config}
    sweep_id = wandb.sweep(sweep_config, project="Oscillogram Classification CNN")
    wandb.agent(sweep_id, call_training_procedure, count=N_RUNS_IN_SWEEP)


if __name__ == '__main__':
    main()
