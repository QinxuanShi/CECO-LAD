import os
import argparse
import datetime
from itertools import product

import yaml
import numpy as np
import torch
import torch.nn as nn

from model.EMAT import EMAT
from solver_ensemble import my_kl_loss


def load_selected_data(path, win_size, input_c):
    """Load pre-scaled selected data and segment into windows.

    Assumes `path` is a whitespace-separated text file where each line is one
    timestep vector of length `input_c`, i.e. the same format as
    `self.scaler.transform(test_data)` saved with `np.savetxt`.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        # Single timestep edge-case: ensure 2D
        data = data.reshape(1, -1)

    if data.shape[1] != input_c:
        raise ValueError(
            f"Selected data has {data.shape[1]} features, but input_c={input_c}. "
            "Please check win_size/input_c and the selected file."
        )

    n = data.shape[0]
    if n < win_size:
        raise ValueError(
            f"Selected data length {n} < win_size {win_size}. Cannot form a window."
        )

    # Use the same 'thre' mode stepping logic as data_loader: step == win_size
    num_windows = (n - win_size) // win_size + 1
    windows = np.stack(
        [data[i * win_size:(i + 1) * win_size] for i in range(num_windows)],
        axis=0,
    )  # [B, win_size, input_c]

    return windows


def build_model(win_size, input_c, output_c, e_layers, d_model=512, n_heads=8, d_ff=512,
                dropout=0.0, activation='gelu', device="cpu"):
    model = EMAT(
        win_size=win_size,
        enc_in=input_c,
        c_out=output_c,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=dropout,
        activation=activation,
        output_attention=True,
    )
    model.to(device)
    model.eval()
    return model


def compute_energy_for_selected(model, x, win_size, temperature=50):
    """Compute attention-based energy for selected windows, mirroring Solver.singlemodelpred."""
    device = next(model.parameters()).device
    x = x.to(device)

    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        output, series, prior, _ = model(x)
        loss = torch.mean(criterion(output, x), dim=-1)  # [B, L]

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(
                    series[u],
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, win_size)
                    ).detach(),
                ) * temperature
                prior_loss = my_kl_loss(
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, win_size)
                    ),
                    series[u].detach(),
                ) * temperature
            else:
                series_loss += my_kl_loss(
                    series[u],
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, win_size)
                    ).detach(),
                ) * temperature
                prior_loss += my_kl_loss(
                    (
                        prior[u]
                        / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1)
                        .repeat(1, 1, 1, win_size)
                    ),
                    series[u].detach(),
                ) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss              # [B, L]
        cri = cri.detach().cpu().numpy()
        energy = cri.reshape(-1)         # 1D energy for all selected timesteps

    return energy


def ensemble_method(method, ensemble_data):
    """Ensemble voting, same rules as Cloud/test_ensemble.py."""
    if method == 'majority':
        # 1. majority voting
        return (ensemble_data.sum(axis=1) >= (ensemble_data.shape[1] // 2 + 1)).astype(int)
    elif method == 'at least one':
        # 2. at least one voting
        return np.any(ensemble_data == 1, axis=1).astype(int)
    elif method == 'consensus':
        # 3. consensus voting
        return np.all(ensemble_data == 1, axis=1).astype(int)
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict on pre-scaled selected data subset using a BAT ensemble "
            "(all checkpoints from a BAT test config)."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to BAT test YAML config (e.g., ensemble_test_os_config.yaml).",
    )
    parser.add_argument(
        "--selected_data",
        type=str,
        required=True,
        help="Path to pre-scaled selected data (e.g., os_selected.txt).",
    )
    parser.add_argument(
        "--thresholds_yaml",
        type=str,
        required=True,
        help=(
            "Path to QBAT ensemble config with per-model thresholds, e.g. "
            "model_config/qbat_config/ensemble_config_os.yaml."
        ),
    )
    parser.add_argument(
        "--voting",
        type=str,
        default="majority",
        choices=["majority", "at least one", "consensus"],
        help="Ensemble voting method across all checkpoints.",
    )
    parser.add_argument(
        "--output_pred",
        type=str,
        default="cloud_selected_preds.txt",
        help="Where to save ensemble 0/1 predictions for selected data.",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load BAT test config (same structure as Cloud/model_config/bat_config/*)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    win_size = cfg["win_size"]
    input_c = cfg["input_c"]
    output_c = cfg["output_c"]
    dataset = cfg["dataset"]
    # model_save_path = cfg["model_save_path"]
    # default model save path relative to this repository; adjust if needed
    model_save_path = "checkpoints/ensemble_os"

    # Hyperparameter sweep keys (same as in test_ensemble.py)
    search_keys = ["num_epochs", "k", "e_layer_num", "batch_size"]
    search_space = [cfg[key] for key in search_keys]
    combinations = list(product(*search_space))

    print(f"Loaded BAT config from {args.config}")
    print(f"Dataset: {dataset}, model_save_path: {model_save_path}")
    print(f"Total checkpoints (ensemble size): {len(combinations)}")

    # Load per-model thresholds from QBAT ensemble config
    with open(args.thresholds_yaml, "r") as f_thr:
        thr_cfg = yaml.safe_load(f_thr)

    # Build mapping: model_name -> threshold
    model_thresholds = {}
    for m in thr_cfg.get("models", []):
        name = m.get("name")
        thr = m.get("threshold")
        if name is None or thr is None:
            continue
        model_thresholds[name] = float(thr)

    if not model_thresholds:
        raise RuntimeError("No model thresholds found in thresholds YAML.")

    print(f"Loaded {len(model_thresholds)} model thresholds from {args.thresholds_yaml}")

    # 1) Load and window the selected data (already scaled on edge)
    print(f"Loading selected data from {args.selected_data}")
    windows = load_selected_data(args.selected_data, win_size, input_c)
    print(f"Selected windows shape: {windows.shape}")

    x = torch.from_numpy(windows).float()

    # 2) For each checkpoint, load model, compute energy and 0/1 predictions
    all_model_preds = []

    for i, values in enumerate(combinations):
        num_epochs, k_val, e_layer_num, batch_size = values

        print(
            f"\n=== Predicting with model {i + 1}/{len(combinations)}: "
            f"e{num_epochs}_k{k_val}_l{e_layer_num}_b{batch_size} ==="
        )

        model = build_model(
            win_size=win_size,
            input_c=input_c,
            output_c=output_c,
            e_layers=e_layer_num,
            device=device,
        )

        fileparam = f"e{num_epochs}_k{k_val}_l{e_layer_num}_b{batch_size}"
        ckpt_name = f"{dataset}_{fileparam}_checkpoint.pth"
        ckpt_path = os.path.join(model_save_path, ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}, skipping this model.")
            continue

        # Lookup per-model threshold from QBAT config by model name
        model_name = f"{dataset}_{fileparam}"
        if model_name not in model_thresholds:
            print(
                f"Threshold for model {model_name} not found in {args.thresholds_yaml}, skipping this model."
            )
            continue
        thr_i = model_thresholds[model_name]
        print(f"Using threshold {thr_i} for model {model_name}")

        print(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Compute energy for selected subset
        energy = compute_energy_for_selected(model, x, win_size=win_size)
        print(f"Energy shape for this model: {energy.shape}")

        # Apply model-specific threshold to get 0/1 predictions for this model
        preds_i = (energy > thr_i).astype(int).reshape(-1, 1)
        all_model_preds.append(preds_i)

    if not all_model_preds:
        raise RuntimeError("No valid checkpoints were found; cannot compute ensemble predictions.")

    # 3) Stack model predictions and apply ensemble voting
    ensemble_matrix = np.concatenate(all_model_preds, axis=1)
    print(f"Ensemble prediction matrix shape: {ensemble_matrix.shape}")

    ensemble_preds = ensemble_method(args.voting, ensemble_matrix)
    print(f"Final ensemble predictions shape: {ensemble_preds.shape}")

    # 4) Save final ensemble predictions
    print(f"Saving ensemble predictions to {args.output_pred}")
    np.savetxt(args.output_pred, ensemble_preds, fmt="%d")


if __name__ == "__main__":
    main()
