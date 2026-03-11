import os
import argparse
import datetime
import logging

import numpy as np
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
from scipy.spatial.distance import euclidean, mahalanobis

from Cloud.test_ensemble import run_bat_ensemble


def ensemble_method(method, ensemble_data):
    """Simple voting over binary predictions (0/1)."""
    if method == 'majority':  # 1. majority voting
        return (ensemble_data.sum(axis=1) >= (ensemble_data.shape[1] // 2 + 1)).astype(int)
    elif method == 'at least one':  # 2. at least one voting
        return np.any(ensemble_data == 1, axis=1).astype(int)
    elif method == 'consensus':  # 3. consensus voting
        return np.all(ensemble_data == 1).astype(int)
    return None


def get_disagreement_indices(all_preds):
    """Indices where edge model predictions disagree."""
    disagreement_mask = np.any(all_preds != all_preds[:, [0]], axis=1)
    return np.where(disagreement_mask)[0].tolist()


def get_abnormal_indices(all_preds):
    """Indices predicted abnormal by at least one edge model."""
    abnormal_indices = set()
    for preds in all_preds.T:
        abnormal_indices.update(np.where(preds == 1)[0])
    return sorted(abnormal_indices)


def get_threshold_indices_by_distance(all_scores, thresholds, dist_type, tolerance=0.1):
    """Indices whose score vectors are far from the threshold point.

    Distance is measured either by Euclidean ("eu") or Mahalanobis ("ma").
    The top `tolerance` fraction with largest distance are selected.
    """
    threshold_point = np.array(thresholds)
    samples = all_scores

    if dist_type == 'ma':
        cov_matrix = np.cov(samples, rowvar=False)
        inv_covmat = np.linalg.inv(cov_matrix)

    all_distances = []
    for i, sample in enumerate(samples):
        if dist_type == 'eu':
            distance = euclidean(sample, threshold_point)
        elif dist_type == 'ma':
            distance = mahalanobis(sample, threshold_point, inv_covmat)
        else:
            raise ValueError("Invalid distance type. Use 'eu' or 'ma'.")
        all_distances.append((i, distance))

    # sort by distance descending and select top fraction
    all_distances.sort(key=lambda x: x[1], reverse=True)
    num_select = int(len(all_distances) * tolerance)
    selected_indices = [idx for idx, _ in all_distances[:num_select]]
    return sorted(selected_indices)


def performance(gt, pred, prefix=""):
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(gt, pred, average='binary')
    logging.info("%sAccuracy : %.6f, Precision : %.6f, Recall : %.6f, F-score : %.6f", prefix, accuracy, precision, recall, f_score)
    return f_score


def load_edge_outputs(config_path):
    """Load edge-side Q-BAT scores/predictions defined by a qbat_config YAML.

    The YAML format is the same as Cloud/model_config/qbat_config/ensemble_config_*.yaml.
    """
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    filename = cfg['data']['filename']
    label_path = cfg['data']['label']
    output_dir = cfg['data']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Loading labels from %s", label_path)
    labels = np.loadtxt(label_path, dtype=int).reshape(-1)

    model_names = [m['name'] for m in cfg['models']]

    all_scores = []
    all_preds = []
    thresholds = []

    for model_cfg in cfg['models']:
        name = model_cfg['name']
        thresholds.append(model_cfg['threshold'])

        scores_file = os.path.join(output_dir, f"{name}_scores.txt")
        preds_file = os.path.join(output_dir, f"{name}_predictions.txt")

        logging.info("Loading edge outputs for %s", name)
        scores = np.loadtxt(scores_file, dtype=float).reshape(-1)
        preds = np.loadtxt(preds_file, dtype=int).reshape(-1)

        all_scores.append(scores)
        all_preds.append(preds)

    all_scores = np.array(all_scores).T
    all_preds = np.array(all_preds).T

    return filename, labels, all_scores, all_preds, thresholds, model_names


def select_edge_models(all_preds, labels, model_names, chosen_models=None):
    """Select subset of edge models, optionally by explicit names.

    Returns chosen_preds, chosen_scores indices, and logging per-model F1.
    """
    f1_scores = [f1_score(labels, all_preds[:, i]) for i in range(all_preds.shape[1])]
    sorted_indices = np.argsort(f1_scores)[::-1]

    all_preds_sorted = all_preds[:, sorted_indices]
    model_names_sorted = [model_names[i] for i in sorted_indices]
    f1_sorted = [f1_scores[i] for i in sorted_indices]

    for name, f1 in zip(model_names_sorted, f1_sorted):
        logging.info("Edge model %s: F1 = %.4f", name, f1)

    if chosen_models is None:
        # Default: take best three edge models
        chosen_models = model_names_sorted[:3]

    chosen_indices = [model_names_sorted.index(name) for name in chosen_models]
    chosen_preds = all_preds_sorted[:, chosen_indices]

    return chosen_preds, chosen_indices, model_names_sorted, f1_sorted


def run_hybrid(edge_cfg, cloud_bat_cfg, decision_choice='ma', tolerance=0.1, chosen_models=None):
    """Run full edge-cloud hybrid pipeline for a single dataset.

    Parameters
    ----------
    edge_cfg : str
        Path to qbat_config YAML (edge outputs + model thresholds).
    cloud_bat_cfg : str
        Path to BAT test YAML for the corresponding dataset.
    decision_choice : {'voting', 'all-abnormal', 'eu', 'ma'}
        Routing policy to decide which samples are sent to cloud.
    tolerance : float
        For 'eu' or 'ma' routing, fraction of points with largest distance to send.
    chosen_models : list[str] or None
        Optional fixed list of edge model names to use (e.g. ['e3_k4_l3_b32', ...]).
    """
    filename, labels, all_scores, all_preds, thresholds, model_names = load_edge_outputs(edge_cfg)

    # Edge model selection
    chosen_preds, chosen_indices, model_names_sorted, _ = select_edge_models(
        all_preds, labels, model_names, chosen_models=chosen_models
    )
    chosen_scores = all_scores[:, chosen_indices]
    chosen_thresholds = [thresholds[i] for i in chosen_indices]

    # Edge ensemble metrics
    logging.info("\n=========== Edge ensemble (majority) ===========")
    ensemble_edge_mj = ensemble_method('majority', chosen_preds)
    performance(labels, ensemble_edge_mj, prefix="Edge majority: ")

    # Routing indices
    if decision_choice == 'voting':
        selected_indices = get_disagreement_indices(chosen_preds)
    elif decision_choice == 'all-abnormal':
        selected_indices = get_abnormal_indices(chosen_preds)
    else:
        selected_indices = get_threshold_indices_by_distance(
            chosen_scores, chosen_thresholds, decision_choice, tolerance=tolerance
        )

    logging.info("Selected %d indices to transmit to cloud (policy=%s)", len(selected_indices), decision_choice)

    # Cloud BAT ensemble
    logging.info("Running cloud BAT ensemble using config: %s", cloud_bat_cfg)
    cloud_preds, cloud_gt = run_bat_ensemble(cloud_bat_cfg, voting_method='majority', log_intermediate=False)

    if cloud_gt.shape[0] != labels.shape[0]:
        logging.warning(
            "Label length mismatch: edge labels=%d, cloud gt=%d. Assuming same ordering and slicing.",
            labels.shape[0], cloud_gt.shape[0],
        )

    logging.info("\n=========== Cloud BAT (majority) ===========")
    performance(labels, cloud_preds, prefix="Cloud majority: ")

    # Hybrid: edge majority + cloud override on selected indices
    hybrid_preds = ensemble_edge_mj.copy()
    hybrid_preds[selected_indices] = cloud_preds[selected_indices]

    logging.info("\n=========== Hybrid edge-cloud result ===========")
    performance(labels, hybrid_preds, prefix="Hybrid: ")

    return {
        'labels': labels,
        'edge_majority': ensemble_edge_mj,
        'cloud_majority': cloud_preds,
        'hybrid': hybrid_preds,
        'selected_indices': np.array(selected_indices, dtype=int),
        'edge_cfg': edge_cfg,
        'cloud_cfg': cloud_bat_cfg,
        'decision_choice': decision_choice,
        'tolerance': tolerance,
    }


if __name__ == "__main__":
    log_filename = f"Hybrid_edge_cloud_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(),
        ],
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_cfg",
        type=str,
        default="Cloud/model_config/qbat_config/ensemble_config_bgl.yaml",
        help="Path to qbat_config YAML for edge outputs (BGL/HDFS/OS).",
    )
    parser.add_argument(
        "--cloud_cfg",
        type=str,
        default="Cloud/model_config/bat_config/ensemble_test_bgl_config.yaml",
        help="Path to BAT test YAML for cloud models.",
    )
    parser.add_argument(
        "--decision_choice",
        type=str,
        default="ma",
        choices=["voting", "all-abnormal", "eu", "ma"],
        help="Routing policy: disagreement, all abnormal, or distance-based.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="Fraction of samples to route when using 'eu' or 'ma' policies.",
    )
    args = parser.parse_args()

    run_hybrid(
        edge_cfg=args.edge_cfg,
        cloud_bat_cfg=args.cloud_cfg,
        decision_choice=args.decision_choice,
        tolerance=args.tolerance,
    )
