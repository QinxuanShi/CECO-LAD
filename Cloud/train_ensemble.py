import os
import time
import argparse
import yaml
from itertools import product
from torch.backends import cudnn
from utils.utils import *
from solver_ensemble import Solver
import sys
import logging
import datetime


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    return solver


if __name__ == '__main__':
    # Set up logging to file
    log_filename = f'BAT_train_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./model_config/bat_config/ensemble_train_bgl_config.yaml', help='Path to config file')
    args, _ = parser.parse_known_args()

    yaml_config = load_config(args.config)

    # Hyperparameter sweep keys
    search_keys = ['num_epochs', 'k', 'e_layer_num', 'batch_size']
    search_space = [yaml_config[key] for key in search_keys]

    combinations = list(product(*search_space))

    # Base config (exclude list-based search parameters)
    base_config = {k: v for k, v in yaml_config.items() if k not in search_keys}

    for i, values in enumerate(combinations):
        config_dict = base_config.copy()
        for key, val in zip(search_keys, values):
            config_dict[key] = val

        config = argparse.Namespace(**config_dict)

        logging.info(f"\n=== Training model {i + 1}/{len(combinations)} ===")

        for k, v in vars(config).items():
            logging.info(f"{k}: {v}")
        logging.info('----------------------------------')

        start_train = time.time()

        main(config)

        end_train = time.time()
        logging.info(f"Training time for {i + 1}/{len(combinations)} model: {end_train - start_train:.2f} seconds")