"""
This version integrates all advanced features: pre-computation, data-driven
concepts, BCE loss, class balancing, CV, ROC/PR analysis, misclassification analysis,
and generates comprehensive documentation including performance graphs, data path logs,
and a maximally detailed Training Bill of Materials (TBOM).
"""
import os
import time
import json
import hashlib
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import clip
import numpy as np
import pandas as pd
import matplotlib
import platform
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from PIL import ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, roc_curve, precision_recall_curve, auc)
from sklearn.model_selection import train_test_split, StratifiedKFold
import sklearn

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set random seeds for reproducibility
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

#  Configuration & Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_EMBEDDING_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# User input for dataset paths
def get_user_input():
    """Get dataset paths from user input"""
    print("=== Mushroom Classifier Configuration ===")
    csv_path = input("Enter path to mushroom CSV dataset (or press Enter for default): ").strip()
    #if not csv_path:
        #csv_path = "/Users/mac2019/Documents/GitHub/DBOM_HiWi/data_sets/mushroom_data/mushrooms.csv"

    image_path = input("Enter path to mushroom image dataset (or press Enter for default): ").strip()
    #if not image_path:
        #image_path = "/Users/mac2019/Documents/GitHub/DBOM_HiWi/data_sets/mushroom_dataset"

    return csv_path, image_path

CSV_PATH, IMAGE_PATH = get_user_input()
OUTPUT_DIR = "./outputs_final_documentation"

# Enhanced hyperparameters for stability
BATCH_SIZE = 64
NUM_EPOCHS_PER_FOLD = 25  # Increased for better convergence
LEARNING_RATE = 5e-5  # Reduced for stability
WEIGHT_DECAY = 1e-3  # Increased for better regularization
CLIP_MODEL_NAME = "ViT-L/14"
CLIP_EMBEDDING_DIM = 768
N_SPLITS = 5
GRAD_CLIP_NORM = 1.0  # Gradient clipping for stability

# 2. Helper Functions & Model Definition 

def generate_concepts_from_csv(csv_path):
    """Generates a list of descriptive text concepts from the mushrooms.csv file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    attribute_map = {
        'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'},
        'cap-surface': {'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'},
        'cap-color': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'bruises': {'t': 'has bruises', 'f': 'has no bruises'},
        'odor': {'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul', 'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'},
        'gill-attachment': {'a': 'attached', 'd': 'descending', 'f': 'free', 'n': 'notched'},
        'gill-spacing': {'c': 'close', 'w': 'crowded', 'd': 'distant'},
        'gill-size': {'b': 'broad', 'n': 'narrow'},
        'gill-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray', 'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'stalk-shape': {'e': 'enlarging', 't': 'tapering'},
        'stalk-root': {'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', '?': 'missing'},
        'stalk-surface-above-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
        'stalk-surface-below-ring': {'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'},
        'stalk-color-above-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'stalk-color-below-ring': {'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange', 'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'},
        'veil-type': {'p': 'partial', 'u': 'universal'},
        'veil-color': {'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'},
        'ring-number': {'n': 'none', 'o': 'one', 't': 'two'},
        'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large', 'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'},
        'spore-print-color': {'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green', 'o': 'orange', 'p': 'purple', 'u': 'white', 'w': 'yellow', 'y': 'missing'},
        'population': {'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered', 'v': 'several', 'y': 'solitary'},
        'habitat': {'g': 'on grasses', 'l': 'on leaves', 'm': 'in meadows', 'p': 'on paths', 'u': 'in urban areas', 'w': 'in waste areas', 'd': 'in woods'}
    }
    concepts = []
    for col in [c for c in df.columns if c != 'class']:
        if col in attribute_map:
            for value_code in df[col].unique():
                prompt = f"a photo of a mushroom where the {col.replace('-', ' ')} is {attribute_map[col].get(value_code, 'unknown')}"
                concepts.append(prompt)
    return concepts

def precompute_features(dataset, clip_model, text_embeddings, device, batch_size):
    """Computes hybrid features for the entire dataset once."""
    all_hybrid_features, all_labels = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            img_feats = clip_model.encode_image(imgs).float()
            img_feats_norm = img_feats / img_feats.norm(dim=1, keepdim=True)
            concept_scores = img_feats_norm @ text_embeddings.T
            combined_features = torch.cat([img_feats, concept_scores], dim=1)
            all_hybrid_features.append(combined_features.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_hybrid_features), torch.cat(all_labels)

# Enhanced MLP with batch normalization for stability
class HybridMLP(nn.Module):
    def __init__(self, in_dim, h1=1024, h2=512, h3=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h3, 1)
        )
    def forward(self, x):
        return self.net(x)

def find_misclassifications(true_labels, pred_labels, pred_probs, dataset_samples, indices, max_examples=5):
    """Find highest confidence misclassification examples"""
    misclassifications = []
    for i, (true_label, pred_label, prob) in enumerate(zip(true_labels, pred_labels, pred_probs)):
        if true_label != pred_label:
            confidence = max(prob, 1-prob)  # Confidence in the prediction
            sample_idx = indices[i]
            file_path = dataset_samples[sample_idx][0] if dataset_samples else "Unknown"
            misclassifications.append({
                'index': int(sample_idx),
                'file_path': file_path,
                'true_label': int(true_label),
                'predicted_label': int(pred_label),
                'confidence': float(confidence)
            })

    # Sort by confidence and return top examples
    misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
    return misclassifications[:max_examples]

def compute_roc_pr_metrics(labels, logits):
    """Compute ROC and PR AUC metrics"""
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    labels_np = np.array(labels)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels_np, probabilities)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels_np, probabilities)
    pr_auc = auc(recall, precision)

    return roc_auc, pr_auc, (fpr.tolist(), tpr.tolist()), (precision.tolist(), recall.tolist())

def get_software_versions():
    """Get comprehensive software version information"""
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "clip": "openai/clip (latest)",
        "pillow": "Latest available",
        "platform": platform.platform()
    }
    return versions

def main():
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate script hash
    try:
        with open(sys.argv[0], 'rb') as f:
            script_content = f.read()
        script_hash = hashlib.sha256(script_content).hexdigest()
    except:
        script_hash = "Unable to compute"

    concepts = generate_concepts_from_csv(CSV_PATH)
    num_concepts = len(concepts)
    print(f"Generated {num_concepts} unique concepts.")

    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()

    with torch.no_grad():
        tokens = clip.tokenize(concepts).to(DEVICE)
        text_emb = clip_model.encode_text(tokens).float()
        text_emb /= text_emb.norm(dim=1, keepdim=True)

    full_dataset = datasets.ImageFolder(root=IMAGE_PATH, transform=preprocess)
    class_names = full_dataset.classes

    print("\n Pre-computing features for all images")
    all_features, all_labels = precompute_features(full_dataset, clip_model, text_emb, DEVICE, BATCH_SIZE)
    print(" Feature pre-computation complete.")

    indices = np.arange(len(all_features))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=all_labels.numpy(), random_state=RANDOM_STATE)
    train_val_features, test_features = all_features[train_val_idx], all_features[test_idx]
    train_val_labels, test_labels = all_labels[train_val_idx], all_labels[test_idx]

    kfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    all_fold_epoch_metrics = []
    fold_final_metrics = []
    all_fold_logits = []
    all_fold_labels = []

    print(f"\n--- Starting {N_SPLITS}-Fold Cross-Validation ---")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_val_features, train_val_labels)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")

        train_ds = TensorDataset(train_val_features[train_ids], train_val_labels[train_ids])
        val_ds = TensorDataset(train_val_features[val_ids], train_val_labels[val_ids])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # Save datapaths for this fold
        fold_datapath_dir = os.path.join(OUTPUT_DIR, "datapaths")
        os.makedirs(fold_datapath_dir, exist_ok=True)
        train_indices_this_fold = train_val_idx[train_ids]
        val_indices_this_fold = train_val_idx[val_ids]

        train_paths = [full_dataset.samples[i][0] for i in train_indices_this_fold]
        val_paths = [full_dataset.samples[i][0] for i in val_indices_this_fold]

        fold_paths_df = pd.DataFrame([
            *[{'path': p, 'split': 'train', 'index': i} for p, i in zip(train_paths, train_indices_this_fold)],
            *[{'path': p, 'split': 'validation', 'index': i} for p, i in zip(val_paths, val_indices_this_fold)]
        ])
        fold_paths_df.to_csv(os.path.join(fold_datapath_dir, f'fold_{fold+1}_paths.csv'), index=False)

        neg, pos = np.bincount(train_val_labels[train_ids].numpy())
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(DEVICE)
        print(f"Calculated Class Weights for balancing: {pos_weight.item():.4f}")

        input_dim = all_features.shape[1]
        model = HybridMLP(input_dim).to(DEVICE)

        # Enhanced optimizer with learning rate scheduling
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=3, verbose=False)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        fold_epoch_history = []
        fold_val_logits, fold_val_labels = [], []

        for epoch in range(1, NUM_EPOCHS_PER_FOLD + 1):
            model.train()
            train_losses, tr_preds, tr_labels = [], [], []
            for features, labels in train_loader:
                features, labels_float = features.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
                logits = model(features)
                loss = criterion(logits, labels_float)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # Gradient clipping
                optimizer.step()
                train_losses.append(loss.item())
                tr_preds.extend((logits.squeeze() > 0).long().cpu().tolist())
                tr_labels.extend(labels.cpu().tolist())

            model.eval()
            val_losses, va_preds, va_labels, va_logits = [], [], [], []
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels_float = features.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
                    logits = model(features)
                    val_losses.append(criterion(logits, labels_float).item())
                    va_preds.extend((logits.squeeze() > 0).long().cpu().tolist())
                    va_labels.extend(labels.cpu().tolist())
                    va_logits.extend(logits.squeeze().cpu().tolist())

            tr_loss, tr_acc = np.mean(train_losses), accuracy_score(tr_labels, tr_preds)
            va_loss, va_acc = np.mean(val_losses), accuracy_score(va_labels, va_preds)

            scheduler.step(va_loss)  # Learning rate scheduling

            fold_epoch_history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})
            print(f"  Epoch {epoch:02d} | Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        all_fold_epoch_metrics.append(fold_epoch_history)
        fold_val_logits = va_logits
        fold_val_labels = va_labels
        all_fold_logits.append(fold_val_logits)
        all_fold_labels.append(fold_val_labels)

        # Compute advanced metrics for this fold
        val_cm = confusion_matrix(va_labels, va_preds)
        val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]

        # ROC and PR metrics
        roc_auc, pr_auc, roc_curve_data, pr_curve_data = compute_roc_pr_metrics(va_labels, fold_val_logits)

        fold_final_metrics.append({
            'fold': fold + 1,
            'final_val_acc': va_acc,
            'confusion_matrix': val_cm.tolist(),
            'confusion_matrix_normalized': val_cm_normalized.tolist(),
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'roc_curve': roc_curve_data,
            'pr_curve': pr_curve_data
        })

        fold_metrics_dir = os.path.join(OUTPUT_DIR, "fold_metrics")
        os.makedirs(fold_metrics_dir, exist_ok=True)
        pd.DataFrame(fold_epoch_history).to_csv(os.path.join(fold_metrics_dir, f'fold_{fold+1}_metrics.csv'), index=False)

    # --- Final Model Training ---
    print("\n--- Training Final Model on all non-test data ---")
    final_train_loader = DataLoader(TensorDataset(train_val_features, train_val_labels), batch_size=BATCH_SIZE, shuffle=True)
    final_model = HybridMLP(input_dim).to(DEVICE)
    final_optimizer = optim.AdamW(final_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, factor=0.7, patience=3, verbose=False)

    final_neg, final_pos = np.bincount(train_val_labels.numpy())
    final_pos_weight = torch.tensor(final_neg / final_pos, dtype=torch.float32).to(DEVICE)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=final_pos_weight)
    print(f"Calculated Class Weights for balancing: {final_pos_weight.item():.4f}")

    for epoch in range(1, NUM_EPOCHS_PER_FOLD + 1):
        final_model.train()
        epoch_losses = []
        for features, labels in final_train_loader:
            features, labels_float = features.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
            logits = final_model(features)
            loss = final_criterion(logits, labels_float)
            final_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), GRAD_CLIP_NORM)
            final_optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        final_scheduler.step(avg_loss)
        print(f"Final Model Epoch {epoch:02d}/{NUM_EPOCHS_PER_FOLD} | Avg Train Loss: {avg_loss:.4f}")

    model_file = os.path.join(OUTPUT_DIR, "final_model.pt")
    torch.save(final_model.state_dict(), model_file)

    # --- Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Unseen Test Set ---")
    final_model.eval()
    test_loader = DataLoader(TensorDataset(test_features, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    te_preds, te_labels, te_logits = [], [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            logits = final_model(features)
            te_preds.extend((logits.squeeze() > 0).long().cpu().tolist())
            te_labels.extend(labels.cpu().tolist())
            te_logits.extend(logits.squeeze().cpu().tolist())

    print("\nTest Accuracy:", accuracy_score(te_labels, te_preds))

    test_cm = confusion_matrix(te_labels, te_preds)
    print("Test Confusion Matrix:")
    print(test_cm)

    # --- Advanced Performance Analysis ---

    # Aggregate CV metrics
    cv_accuracies = [f['final_val_acc'] for f in fold_final_metrics]
    cv_roc_aucs = [f['roc_auc'] for f in fold_final_metrics]
    cv_pr_aucs = [f['pr_auc'] for f in fold_final_metrics]

    aggregate_cm = np.sum([np.array(f['confusion_matrix']) for f in fold_final_metrics], axis=0)
    aggregate_cm_normalized = aggregate_cm.astype('float') / aggregate_cm.sum(axis=1)[:, np.newaxis]

    # Test set advanced metrics
    test_roc_auc, test_pr_auc, test_roc_curve, test_pr_curve = compute_roc_pr_metrics(te_labels, te_logits)
    test_cm_normalized = test_cm.astype('float') / test_cm.sum(axis=1)[:, np.newaxis]

    # Misclassification analysis
    te_probs = torch.sigmoid(torch.tensor(te_logits)).numpy()
    misclassifications = find_misclassifications(te_labels, te_preds, te_probs, full_dataset.samples, test_idx, max_examples=10)

    # --- Model Interpretation ---
    try:
        final_weights = final_model.net[-1].weight.cpu().detach().numpy().flatten()
        concept_weights = final_weights[-num_concepts:]

        if len(concept_weights) == len(concepts):
            concept_weights_df = pd.DataFrame({'concept': concepts, 'weight': concept_weights})
            poisonous_concepts = concept_weights_df[concept_weights_df['weight'] > 0].sort_values('weight', ascending=False)
            edible_concepts = concept_weights_df[concept_weights_df['weight'] < 0].sort_values('weight', ascending=True)
    except Exception as e:
        print(f"Could not perform interpretation due to an error: {e}")
        poisonous_concepts = pd.DataFrame()
        edible_concepts = pd.DataFrame()

    # --- Performance Graphs ---
    plt.figure(figsize=(16, 12))
    max_len = max(len(hist) for hist in all_fold_epoch_metrics)
    epochs = np.arange(1, max_len + 1)

    avg_train_loss = np.nanmean([pd.DataFrame(hist).set_index('epoch')['train_loss'].reindex(epochs).values for hist in all_fold_epoch_metrics], axis=0)
    avg_val_loss = np.nanmean([pd.DataFrame(hist).set_index('epoch')['val_loss'].reindex(epochs).values for hist in all_fold_epoch_metrics], axis=0)
    avg_train_acc = np.nanmean([pd.DataFrame(hist).set_index('epoch')['train_acc'].reindex(epochs).values for hist in all_fold_epoch_metrics], axis=0)
    avg_val_acc = np.nanmean([pd.DataFrame(hist).set_index('epoch')['val_acc'].reindex(epochs).values for hist in all_fold_epoch_metrics], axis=0)

    # Loss subplot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, avg_train_loss, label='Avg Train Loss', marker='o', linewidth=2)
    plt.plot(epochs, avg_val_loss, label='Avg Val Loss', marker='s', linewidth=2)
    plt.title('Average Loss vs. Epochs across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy subplot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, avg_train_acc, label='Avg Train Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, avg_val_acc, label='Avg Val Accuracy', marker='s', linewidth=2)
    plt.title('Average Accuracy vs. Epochs across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ROC Curve subplot
    plt.subplot(2, 2, 3)
    fpr, tpr = test_roc_curve
    plt.plot(fpr, tpr, linewidth=2, label=f'Test ROC (AUC = {test_roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PR Curve subplot
    plt.subplot(2, 2, 4)
    precision, recall = test_pr_curve
    plt.plot(recall, precision, linewidth=2, label=f'Test PR (AUC = {test_pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_DIR, 'performance_graphs.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"\nPerformance graphs saved to {graph_path}")

    # --- Comprehensive TBOM Generation ---

    def get_class_distribution(indices, all_labels, class_names):
        dist = {}
        subset_labels = all_labels[indices].numpy()
        for i, class_name in enumerate(class_names):
            dist[class_name] = int((subset_labels == i).sum())
        return dist

    train_val_dist = get_class_distribution(train_val_idx, all_labels, class_names)
    test_dist = get_class_distribution(test_idx, all_labels, class_names)
    overall_dist = {k: train_val_dist.get(k, 0) + test_dist.get(k, 0) for k in class_names}

    final_report = classification_report(te_labels, te_preds, target_names=class_names, digits=4, output_dict=True)

    training_time = time.time() - start_time

    tbom = {
        "tbom_version": "3.0-advanced",
        "generation_details": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generating_script": os.path.basename(sys.argv[0]) if not hasattr(sys, 'ps1') else "interactive_notebook",
            "script_sha256_hash": script_hash,
            "training_time_seconds": training_time,
            "random_seed": RANDOM_STATE
        },
        "data_summary": {
            "image_dataset_path": os.path.abspath(IMAGE_PATH),
            "concept_source_csv": os.path.abspath(CSV_PATH),
            "total_samples": len(full_dataset),
            "class_information": {"count": len(class_names), "names": class_names, "distribution": overall_dist},
            "concept_details": {"generation_method": "Programmatic from CSV", "count": num_concepts, "concepts_list": concepts},
            "data_splits": {
                "train_validation_set": {"size": len(train_val_idx), "distribution": train_val_dist, "indices": train_val_idx.tolist()},
                "test_set": {"size": len(test_idx), "distribution": test_dist, "indices": test_idx.tolist()}
            }
        },
        "model_architecture": {
            "model_name": "Enhanced Hybrid Concept-Based MLP Classifier",
            "description": f"A concept-based model using {CLIP_MODEL_NAME} backbone with batch normalization and enhanced regularization.",
            "components": [
                {"name": "Backbone", "type": f"Frozen CLIP Image Encoder ({CLIP_MODEL_NAME})"},
                {"name": "Concept_Bank", "type": f"{num_concepts} Data-Driven Text Concepts"},
                {"name": "Feature_Vector", "type": "Concatenated Raw Features and Concept Scores", "dimension": input_dim},
                {"name": "Classifier_Head", "type": "4-Layer MLP with BatchNorm", "details": "Linear(in -> 1024) -> BatchNorm -> ReLU -> Dropout -> Linear(1024 -> 512) -> BatchNorm -> ReLU -> Dropout -> Linear(512 -> 256) -> BatchNorm -> ReLU -> Dropout -> Linear(256 -> 1)"}
            ]
        },
        "training_methodology": {
            "approach": f"{N_SPLITS}-Fold Stratified Cross-Validation",
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "epochs_per_fold": NUM_EPOCHS_PER_FOLD,
                "optimizer": "AdamW",
                "loss_function": "BCEWithLogitsLoss",
                "gradient_clipping": GRAD_CLIP_NORM,
                "learning_rate_scheduler": "ReduceLROnPlateau"
            },
            "stability_enhancements": [
                "Batch Normalization",
                "Gradient Clipping",
                "Learning Rate Scheduling",
                "Enhanced Regularization"
            ],
            "class_balancing": "Enabled via pos_weight in loss function.",
            "final_model_training": f"A final model was trained on the complete train+validation set for {NUM_EPOCHS_PER_FOLD} epochs."
        },
        "performance_metrics": {
            "cross_validation_results": {
                "mean_accuracy": float(np.mean(cv_accuracies)),
                "std_accuracy": float(np.std(cv_accuracies)),
                "mean_roc_auc": float(np.mean(cv_roc_aucs)),
                "std_roc_auc": float(np.std(cv_roc_aucs)),
                "mean_pr_auc": float(np.mean(cv_pr_aucs)),
                "std_pr_auc": float(np.std(cv_pr_aucs)),
                "aggregate_confusion_matrix": aggregate_cm.tolist(),
                "aggregate_confusion_matrix_normalized": aggregate_cm_normalized.tolist(),
                "per_fold_details": fold_final_metrics
            },
            "final_test_results": {
                "accuracy": final_report['accuracy'],
                "roc_auc": test_roc_auc,
                "pr_auc": test_pr_auc,
                "macro_avg": final_report['macro avg'],
                "weighted_avg": final_report['weighted avg'],
                "per_class_metrics": {class_names[0]: final_report[class_names[0]], class_names[1]: final_report[class_names[1]]},
                "confusion_matrix": test_cm.tolist(),
                "confusion_matrix_normalized": test_cm_normalized.tolist(),
                "roc_curve": test_roc_curve,
                "pr_curve": test_pr_curve
            },
            "misclassification_analysis": {
                "high_confidence_errors": misclassifications[:5],
                "total_misclassifications_analyzed": len(misclassifications)
            }
        },
        "model_interpretation": {
            "poisonous_indicators": poisonous_concepts.to_dict('records') if not poisonous_concepts.empty else [],
            "edible_indicators": edible_concepts.to_dict('records') if not edible_concepts.empty else []
        },
        "output_artifacts": {
            "model_path": os.path.abspath(model_file),
            "tbom_path": os.path.abspath(os.path.join(OUTPUT_DIR, "TBOM.json")),
            "performance_graphs": os.path.abspath(graph_path),
            "per_fold_metrics_logs": os.path.abspath(os.path.join(OUTPUT_DIR, "fold_metrics")),
            "per_fold_datapath_logs": os.path.abspath(os.path.join(OUTPUT_DIR, "datapaths"))
        },
        "environment_and_dependencies": {
            "device": DEVICE,
            "software_versions": get_software_versions()
        }
    }

    # Generate TBOM signature
    tbom_string = json.dumps(tbom, sort_keys=True).encode("utf-8")
    tbom["signature"] = hashlib.sha256(tbom_string).hexdigest()

    # Save TBOM to file (don't print to terminal)
    tbom_file = os.path.join(OUTPUT_DIR, "TBOM.json")
    with open(tbom_file, "w") as f:
        json.dump(tbom, f, indent=4)
    print(f"TBOM successfully generated and saved to {tbom_file}")

if __name__ == "__main__":
    main()
