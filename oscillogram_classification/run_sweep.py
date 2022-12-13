#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author Patricia Windler, Tim Bohne

import argparse
import os

import wandb

import train
from config import sweep_config, api_key
from train import file_path

TUNING_METHOD = 'random'
N_RUNS_IN_SWEEP = 3


def main():
    def call_training_procedure(config_dict=None):
        """
        Wrapper for the training procedure.

        :param config_dict: hyperparameter config
        """
        train.train_procedure(args.train_path, args.val_path, args.test_path, hyperparameter_config=config_dict)

    parser = argparse.ArgumentParser(description='hyper parameter tuning with "weights & biases sweep"')
    parser.add_argument('--train_path', type=file_path, required=True)
    parser.add_argument('--val_path', type=file_path, required=True)
    parser.add_argument('--test_path', type=file_path, required=True)
    args = parser.parse_args()

    os.environ["WANDB_API_KEY"] = api_key.wandb_api_key
    config = {'method': TUNING_METHOD, 'parameters': sweep_config.sweep_config}
    sweep_id = wandb.sweep(config, project="Oscillogram Classification")
    wandb.agent(sweep_id, call_training_procedure, count=N_RUNS_IN_SWEEP)


if __name__ == '__main__':
    main()
