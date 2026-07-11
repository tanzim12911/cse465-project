import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_features(filepath="./data/features/features.pt"):
    print(f"Loading features from {filepath}...")
    try:
        return torch.load(filepath, weights_only=False)
    except Exception as e:
        print(f"Error loading features: {e}")
        return None


def train_and_evaluate_probe(X, y, layer_name, task_name):
    """Trains a linear probe on X and returns accuracy."""
    # Filter out None placeholders
    valid = [(x, label) for x, label in zip(X, y) if x is not None]
    if len(valid) < 4:
        print(f"  [{task_name}] {layer_name}: not enough valid samples ({len(valid)}), skipping.")
        return 0.0

    X_clean = np.array([x for x, _ in valid])
    y_clean = np.array([label for _, label in valid])

    # Must be 2D
    if X_clean.ndim == 1:
        X_clean = X_clean.reshape(-1, 1)

    # Drop NaN rows
    nan_mask = np.isnan(X_clean).any(axis=1)
    X_clean  = X_clean[~nan_mask]
    y_clean  = y_clean[~nan_mask]

    if len(X_clean) < 4:
        print(f"  [{task_name}] {layer_name}: too many NaNs, skipping.")
        return 0.0

    if len(np.unique(y_clean)) < 2:
        print(f"  [{task_name}] {layer_name}: only one class present, skipping.")
        return 0.0

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
        )
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  [{task_name}] {layer_name:15s} probe accuracy: {acc*100:.1f}%")
        return acc
    except Exception as e:
        print(f"  [{task_name}] {layer_name} error: {e}")
        return 0.0


def main(features_path="./data/features/features.pt"):
    features = load_features(features_path)
    if features is None:
        return {}

    tasks  = np.array(features["task"])
    labels = np.array(features["labels"])

    unique_tasks = np.unique(tasks)
    print(f"\n--- Three-Stage Probing Results ({len(unique_tasks)} tasks) ---")

    probe_results = {}
    for task in unique_tasks:
        mask        = (tasks == task)
        task_labels = labels[mask]
        print(f"\nTask: {task}  ({mask.sum()} samples)")

        task_probes = {}
        for layer in ["vision_encoder", "llm_mid", "llm_out"]:
            raw = features.get(layer, [])
            if len(raw) == 0:
                continue
            layer_feats = np.array(raw, dtype=object)[mask].tolist()
            acc = train_and_evaluate_probe(layer_feats, task_labels, layer, task)
            task_probes[layer] = acc

        probe_results[task] = task_probes

    return probe_results


if __name__ == "__main__":
    main()
