import os
import argparse
import logging
import datetime
from torch.backends import cudnn
from utils.utils import *

from solver_ensemble import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    # Set up logging to file
    log_filename = f'EM-AT_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--data_seq_len', type=int, default=10) 
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--e_layer_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/ensemble_bgl')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)

    config = parser.parse_args()

    fileparam = (
        f"e{config.num_epochs}_"
        f"k{config.k}_"
        f"l{config.e_layer_num}_"
        f"b{config.batch_size}"
    )
    model_name = f"{config.dataset}_{fileparam}_checkpoint.pth"
    
    # update model_save_path to a unique folder for this model
    config.model_save_path = os.path.join(
        config.model_save_path, model_name
    )

    logging.info(f"model_save_path: {config.model_save_path}")

    args = vars(config)
    logging.info('------------ Options -------------')
    for k, v in sorted(args.items()):
        logging.info("%s: %s", k, v)
    logging.info('-------------- End ----------------')
    main(config)
