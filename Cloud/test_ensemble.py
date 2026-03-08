import os
import argparse
import yaml
from itertools import product
from torch.backends import cudnn
from utils.utils import *
from solver_ensemble import Solver
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import logging
import datetime


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def performance(gt, pred):
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
    logging.info("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        accuracy, precision, recall, f_score))


def ensemble_method(method, ensemble_data):
    if method == 'majority':  # 1. majority voting
        return (ensemble_data.sum(axis=1) >= (ensemble_data.shape[1] // 2 + 1)).astype(int)
    elif method == 'at least one':  # 2. at least one voting
        return np.any(ensemble_data == 1, axis=1).astype(int)
    elif method == 'consensus':  # 3. consensus voting
        return np.all(ensemble_data == 1, axis=1).astype(int)
    return None


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    pred, gt = solver.singlemodelpred()
    performance(gt, pred)

    return pred, gt


if __name__ == '__main__':
    # Set up logging to file
    log_filename = f'BAT_test_sequence_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./model_config/bat_config/ensemble_test_bgl_config.yaml', help='Path to config file')
    parser.add_argument('--voting', type=str, default='all',
                        choices=['all', 'majority', 'at least one', 'consensus'],
                        help='Voting method for ensemble')
    args, _ = parser.parse_known_args()

    yaml_config = load_config(args.config)

    # Hyperparameter sweep keys
    search_keys = ['num_epochs', 'k', 'e_layer_num', 'batch_size']
    search_space = [yaml_config[key] for key in search_keys]

    combinations = list(product(*search_space))

    # Base config (exclude list-based search parameters)
    base_config = {k: v for k, v in yaml_config.items() if k not in search_keys}

    ground_truth = None
    prediction_results = []

    # BAT model testing by prediction in sequence
    for i, values in enumerate(combinations):
        config_dict = base_config.copy()
        for key, val in zip(search_keys, values):
            config_dict[key] = val

        config = argparse.Namespace(**config_dict)

        logging.info("\n=== Testing model %d/%d ===", i + 1, len(combinations))
        for k, v in vars(config).items():
            logging.info(f"{k}: {v}")
        logging.info('----------------------------------')

        # single model prediction
        pred, gt = main(config)

        # check the label
        if ground_truth is None:
            ground_truth = gt
        elif not np.array_equal(ground_truth, gt):
            raise ValueError("Ground truth not the same!")

        # put all prediction from different models together
        prediction_results.append(pred.reshape(-1, 1))

        # each step of ensemble results (ensemble process)
        ensemble_process = np.concatenate(prediction_results, axis=1)

        logging.info(f"===================={args.voting} voting for ensemble step {i + 1}======================")
        if i + 1 < 3 and (args.voting == 'majority' or args.voting == 'all'):
            logging.info("Majority voting needs at least three models!")
        else:
            if args.voting == 'all':
                for method in ['majority', 'at least one', 'consensus']:
                    ensemble_results_part = ensemble_method(method, ensemble_process)
                    performance(ground_truth, ensemble_results_part)
            else:
                ensemble_results_part = ensemble_method(args.voting, ensemble_process)
                performance(ground_truth, ensemble_results_part)

    # final results for ensemble
    prediction_results = np.concatenate(prediction_results, axis=1)

    logging.info(f"===================={args.voting} voting for ensemble all models======================")
    if args.voting == 'all':
        for method in ['majority', 'at least one', 'consensus']:
            ensemble_results = ensemble_method(method, prediction_results)
            performance(ground_truth, ensemble_results)
    else:
        ensemble_results = ensemble_method(args.voting, prediction_results)
        performance(ground_truth, ensemble_results)
