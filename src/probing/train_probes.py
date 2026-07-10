import os
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_features(filepath="./data/features/features.pt"):
    print(f"Loading features from {filepath}...")
    try:
        features = torch.load(filepath, weights_only=False)
        return features
    except Exception as e:
        print(f"Error loading features: {e}")
        return None

def train_and_evaluate_probe(X, y, layer_name, task_name):
    """
    Trains a lightweight linear classifier on the features and returns accuracy.
    """
    if len(X) == 0:
        return 0.0
        
    # Some classes might only have 1 sample in a small subset, handle gracefully
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"[{task_name}] {layer_name} Probe Accuracy: {acc*100:.2f}%")
        return acc
    except Exception as e:
        print(f"[{task_name}] {layer_name} Probe Error: {e}")
        return 0.0

def main():
    features = load_features()
    if features is None:
        return
        
    tasks = np.array(features['task'])
    labels = np.array(features['labels'])
    
    unique_tasks = np.unique(tasks)
    
    print("\n--- Three-Stage Probing Results ---")
    
    for task in unique_tasks:
        print(f"\nEvaluating probes for task: {task}")
        mask = (tasks == task)
        task_labels = labels[mask]
        
        # We need more than one class to train a classifier
        if len(np.unique(task_labels)) < 2:
            print(f"Skipping {task} - not enough distinct labels in subset.")
            continue
            
        for layer in ['vision_encoder', 'llm_mid', 'llm_out']:
            if len(features[layer]) > 0:
                layer_features = np.array(features[layer])[mask]
                train_and_evaluate_probe(layer_features, task_labels, layer, task)

if __name__ == "__main__":
    main()
