import random
from itertools import product
from typing import Dict, Tuple, Any
import yaml


def get_random_state(config_path: str, num_epochs: int, k: int, e_layer_num: int, batch_size: int) -> int:
    """
    return a unique random state for the given parameter combination

    args:
        config_path (str): path to the YAML config file
        num_epochs (int): selected num_epochs
        k (int): selected k
        e_layer_num (int): selected encoder layer number
        batch_size (int): selected batch size

    returns:
        int: unique random state in range [10, 301]
    """

    # load yaml
    with open(config_path, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    # read parameter lists from yaml
    try:
        num_epochs_list = cfg["num_epochs"]
        k_list = cfg["k"]
        e_layer_num_list = cfg["e_layer_num"]
        batch_size_list = cfg["batch_size"]
    except KeyError as e:
        raise KeyError(f"Missing required key in config: {e}")

    # generate all combinations
    all_combinations = list(product(num_epochs_list, k_list, e_layer_num_list, batch_size_list))

    # generate unique random state values
    random.seed(42)
    unique_random_states = random.sample(range(10, 301), len(all_combinations))

    # mapping from combination to random state
    combo_to_random_state: Dict[Tuple[int, int, int, int], int] = {
        combo: state for combo, state in zip(all_combinations, unique_random_states)
    }

    key = (num_epochs, k, e_layer_num, batch_size)
    if key not in combo_to_random_state:
        raise ValueError(f"invalid combination: {key}")

    print(f"get combination {key}, the random state is {combo_to_random_state[key]}")
    return combo_to_random_state[key]
