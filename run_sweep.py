import wandb
import os
import train
import yaml

from configs import api_key

TUNING_METHOD = 'random'
N_RUNS_IN_SWEEP = 3


def main():
    os.environ["WANDB_API_KEY"] = api_key.wandb_api_key
    try:
        with open(".\\configs\\example_sweep_config.yaml") as f:
            hyperparameter_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Could not find config file!")
        raise

    sweep_config = {'method': TUNING_METHOD, 'parameters': hyperparameter_config}
    sweep_id = wandb.sweep(sweep_config, project="Oscillogram Classification CNN")
    wandb.agent(sweep_id, train.train_procedure, count=N_RUNS_IN_SWEEP)


if __name__ == '__main__':
    main()
