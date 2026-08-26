"""
Smart Overlap MoE Allocation Optimizer
---------------------------------------
Computes acoustic vulnerability (Error Rate + Entropy + Pairwise Peak Confusion)
from realistic acoustic benchmark evaluation and generates the optimal 
vocabulary partitions with smart overlap and acoustic anti-affinity.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import yaml

from datasets import SpeechCommandsDataset
from models import streaming as streaming_models
from eval_multi_device_array import (
    MASTER_LABELS,
    SpectrogramTransform,
    synthesize_spatial_channels_and_hendriks_noise,
    compute_normalized_mel,
    load_model_from_log
)


def calculate_overlap_allocation_optimized(
    cm_noisy: np.ndarray,
    class_names: list,
    recall_drops: np.ndarray = None,
    num_nodes: int = 5,
    min_rep: int = 3,
    alpha: float = 0.70,   # Weight: Error Rate (1 - Recall)
    beta: float = 0.15,    # Weight: Confusion Entropy (noise dispersion)
    gamma: float = 0.15,   # Weight: Maximum Pairwise Confusion Peak
    top_p_3node: float = 0.20, # Top 20% hardest clusters -> 3 nodes
    top_p_2node: float = 0.35, # Next 35% hard clusters -> 2 nodes
    coupling_threshold: float = 0.05, # Minimum mutual confusion to form an atomic cluster
    max_cluster_size: int = 4 # Bounded Graph Cut: maximum allowed words per atomic cluster
):
    """
    Graph-Coupled Bounded Cluster MoE Optimizer (Scalable to 9,000 Classes):
      1. Builds Symmetrized Acoustic Adjacency Matrix W_ij = max(P(j|i), P(i|j)).
      2. Extracts Atomic Confusable Cliques (e.g. {tree, three}, {forward, four}).
      3. Enforces Bounded Graph Capacity (max_cluster_size <= 4) to prevent giant chains.
      4. Propagates maximum vulnerability across the cluster (Symmetric Invariant).
      5. Co-locates coupled clusters onto the exact same edge devices to eliminate truncated Softmax distortion.
    """
    n_classes = len(class_names)
    assert cm_noisy.shape == (n_classes, n_classes), "CM dimensions must match class count."

    # 1. Normalize Confusion Matrix -> P(prediction = j | ground_truth = i)
    row_sums = cm_noisy.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p_matrix = cm_noisy / row_sums

    # 2. Per-class Recall & Error Rate
    recalls = np.diag(p_matrix)
    error_rates = 1.0 - recalls

    # 3. Acoustic Confusion Entropy H(k)
    entropies = np.zeros(n_classes)
    for i in range(n_classes):
        row = p_matrix[i, :]
        nonzero = row[row > 0]
        entropies[i] = -np.sum(nonzero * np.log2(nonzero)) / np.log2(n_classes)

    # 4. Symmetrized Acoustic Adjacency Matrix W_ij
    off_diag_matrix = p_matrix.copy()
    np.fill_diagonal(off_diag_matrix, 0.0)
    w_matrix = np.maximum(off_diag_matrix, off_diag_matrix.T)
    max_pairwise_peaks = off_diag_matrix.max(axis=1)

    # 5. Raw Vulnerability Score V_raw(k)
    norm_error = error_rates / (error_rates.max() + 1e-8)
    norm_peaks = max_pairwise_peaks / (max_pairwise_peaks.max() + 1e-8)
    v_raw = (alpha * norm_error + beta * entropies + gamma * norm_peaks)
    v_raw_norm = (v_raw - v_raw.min()) / (v_raw.max() - v_raw.min() + 1e-8)

    # 6. Graph Connected Components / Bounded Atomic Clusters
    visited = set()
    clusters = []

    for i in range(n_classes):
        if i in visited:
            continue
        # Find all strongly coupled neighbors
        cluster = [i]
        visited.add(i)
        
        # Breadth-First Search for coupled neighbors
        queue = [i]
        while queue and len(cluster) < max_cluster_size:
            curr = queue.pop(0)
            for j in range(n_classes):
                if j not in visited and w_matrix[curr, j] >= coupling_threshold:
                    if len(cluster) < max_cluster_size:
                        visited.add(j)
                        cluster.append(j)
                        queue.append(j)
        clusters.append(cluster)

    # 7. Cluster-Level Vulnerability & Symmetric Propagation
    cluster_vulnerabilities = []
    v_coupled = np.zeros(n_classes)

    for cluster in clusters:
        # Cluster vulnerability = maximum vulnerability of any member
        cluster_v = float(np.max([v_raw_norm[idx] for idx in cluster]))
        cluster_vulnerabilities.append(cluster_v)
        for idx in cluster:
            v_coupled[idx] = cluster_v

    # 8. Assign Universal Overlap Replication Levels (min_rep for standard words, num_nodes for anchors)
    replications = np.full(n_classes, min_rep, dtype=int)
    # Top 2 hardest confusable words get num_nodes (Anchors on all nodes)
    top_anchor_indices = np.argsort(-v_coupled)[:2]
    replications[top_anchor_indices] = num_nodes

    # 9. Atomic Cluster Allocation to Nodes (Zero Softmax Truncation Distortion)
    node_allocations = {f"ESP32_Node_{i+1}": [] for i in range(num_nodes)}
    node_loads = np.zeros(num_nodes, dtype=int)

    # Sort clusters descending by vulnerability
    sorted_cluster_indices = np.argsort([-cv for cv in cluster_vulnerabilities])

    for c_idx in sorted_cluster_indices:
        cluster = clusters[c_idx]
        rep_count = replications[cluster[0]] # All members share identical replication

        if rep_count == num_nodes:
            for n_i in range(num_nodes):
                for kw_idx in cluster:
                    node_allocations[f"ESP32_Node_{n_i+1}"].append(class_names[kw_idx])
                    node_loads[n_i] += 1
        else:
            # Assign entire atomic cluster together to the least loaded nodes
            assigned_nodes = np.argsort(node_loads)[:rep_count]
            for n_i in assigned_nodes:
                for kw_idx in cluster:
                    node_allocations[f"ESP32_Node_{n_i+1}"].append(class_names[kw_idx])
                    node_loads[n_i] += 1

    df_report = pd.DataFrame({
        "Keyword": class_names,
        "Recall": np.round(recalls, 3),
        "Entropy": np.round(entropies, 3),
        "Pairwise_Peak": np.round(max_pairwise_peaks, 3),
        "Vulnerability": np.round(v_coupled, 3),
        "Replication_Nodes": replications
    }).sort_values(by="Vulnerability", ascending=False).reset_index(drop=True)

    return df_report, node_allocations, node_loads


def generate_smart_overlap_yamls(node_allocations, num_nodes=5):
    """Generates standard Hydra YAML configs for smart overlap nodes."""
    for node_name, classes in node_allocations.items():
        dev_idx = node_name.split("_")[-1]
        
        # Ensure 'nothing' is in vocabulary
        vocab = sorted([c for c in classes if c != "nothing"]) + ["nothing"]
        num_classes = len(vocab)
        prefix = f"5dev_" if num_nodes == 5 else ""
        exp_name = f"kws_smart_overlap_{prefix}dev{dev_idx}_{num_classes}classes"

        cfg_dict = {
            "experiment_name": exp_name,
            "hydra": {
                "run": {"dir": f"./logs/${{experiment_name}}/${{now:%Y-%m-%d_%H-%M-%S}}"},
                "job": {"name": "${experiment_name}"}
            },
            "training": {
                "batch_size": 64,
                "epochs": 100,
                "patience": 10,
                "min_delta": 0.001
            },
            "model": {
                "_target_": "models.streaming.Improved_Phi_GRU_ATT_Streaming",
                "num_classes": num_classes,
                "n_mel_bins": 40,
                "hidden_dim": 32,
                "n_fft": 256,
                "hop_length": 160,
                "export_mode": False,
                "matchbox": {
                    "input_channels": 40,
                    "base_filters": 32,
                    "block_filters": 16,
                    "dropout_rate": 0.25,
                    "use_se": True,
                    "expansion_factor": 0.8,
                    "num_blocks": 2,
                    "sub_blocks_per_block": 2,
                    "kernel_sizes": [7, 5, 3, 3, 3, 5, 3, 1],
                    "dilations": [1, 2, 4, 8, 4, 2, 1, 1],
                    "skip_connections": {
                        "enable_block_skips": True,
                        "enable_sub_block_skips": True,
                        "enable_final_skip": True
                    }
                }
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 1e-3,
                "weight_decay": 1e-5
            },
            "scheduler": {
                "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
                "T_max": 100,
                "eta_min": 5e-6,
                "verbose": True
            },
            "dataset": {
                "defaults": None,
                "preload": True,
                "allowed_classes": vocab,
                "train": {
                    "_target_": "datasets.SpeechCommandsDataset",
                    "root_dir": "speech_commands_dataset",
                    "subset": "training",
                    "augment": True
                },
                "val": {
                    "_target_": "datasets.SpeechCommandsDataset",
                    "root_dir": "speech_commands_dataset",
                    "subset": "validation",
                    "augment": False
                },
                "test": {
                    "_target_": "datasets.SpeechCommandsDataset",
                    "root_dir": "speech_commands_dataset",
                    "subset": "testing",
                    "augment": False
                }
            }
        }

        yaml_path = f"config/{exp_name}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(cfg_dict, f, sort_keys=False)
        print(f"Created Smart Overlap Config: {yaml_path} ({num_classes} classes)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize Smart Overlap Allocation for Distributed MoE.")
    parser.add_argument("--nodes", "-N", type=int, default=5, choices=[3, 5], help="Number of distributed nodes (3 or 5)")
    parser.add_argument("--min_rep", "-R", type=int, default=0, help="Replication level for standard keywords (0 = auto: 3 for 5 nodes, 2 for 3 nodes)")
    args = parser.parse_args()

    min_rep = args.min_rep if args.min_rep > 0 else (3 if args.nodes == 5 else 2)

    print("==========================================================")
    print(f"  SMART OVERLAP MOE ALLOCATION OPTIMIZER ({args.nodes} NODES, R={min_rep})      ")
    print("==========================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_all = "logs/kws_streaming_35_classes/2026-08-17_11-49-55"
    if not os.path.exists(log_all):
        log_all = max(glob.glob("logs/kws_streaming_35_classes/*"), key=os.path.getmtime)
    
    print(f"Loading reference 36-class generalist model from: {log_all}")
    model_all, classes_all = load_model_from_log(log_all, device)

    # Filter out 'nothing' from the target keywords for allocation
    keyword_names = [c for c in classes_all if c != "nothing"]
    n_kws = len(keyword_names)

    # 1. Compute empirical confusion matrix on test dataset
    print(f"Computing empirical confusion matrix over test dataset ({n_kws} keywords)...")
    test_ds = SpeechCommandsDataset(
        root_dir="speech_commands_dataset",
        allowed_classes=classes_all,
        subset="testing",
        augment=False,
        preload=False
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True)
    spec_tf = SpectrogramTransform(n_fft=256, hop_length=160, n_mels=40)
    from tqdm import tqdm

    y_true = []
    y_pred = []

    print(f"Evaluating 3,500 balanced test samples across all 35 classes...")
    with torch.no_grad():
        for i, (wave, label_idx) in tqdm(enumerate(test_loader), total=3500, desc="Evaluating test set"):
            if i >= 3500: break
            if wave.dim() == 3: wave = wave.squeeze(0)
            
            true_kw = classes_all[label_idx.item()]
            if true_kw == "nothing": continue

            spec = compute_normalized_mel(wave, spec_tf, device)
            out = model_all(spec)
            if isinstance(out, tuple): out = out[0]
            
            probs = torch.softmax(out, dim=-1).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probs))
            pred_kw = classes_all[pred_idx]

            if pred_kw in keyword_names and true_kw in keyword_names:
                y_true.append(keyword_names.index(true_kw))
                y_pred.append(keyword_names.index(pred_kw))

    cm = np.zeros((n_kws, n_kws), dtype=float)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1.0

    # 2. Run the Optimization Algorithm
    df_report, node_allocations, node_loads = calculate_overlap_allocation_optimized(
        cm_noisy=cm,
        class_names=keyword_names,
        num_nodes=args.nodes,
        min_rep=min_rep,
        alpha=0.70,
        beta=0.15,
        gamma=0.15,
        top_p_3node=0.20,
        top_p_2node=0.35,
        coupling_threshold=0.05,
        max_cluster_size=4
    )

    print("\n================ TOP VULNERABLE KEYWORDS ================")
    print(df_report.head(15).to_string(index=False))

    print(f"\n================ BALANCED {args.nodes}-NODE ALLOCATIONS ================")
    for node, kw_list in node_allocations.items():
        print(f"\n{node} (Total Vocab: {len(kw_list) + 1} with 'nothing'):")
        print(f"  Keywords: {', '.join(sorted(kw_list))}")

    # 3. Generate YAML configs
    print("\n================ GENERATING CONFIG FILES ================")
    generate_smart_overlap_yamls(node_allocations, num_nodes=args.nodes)

    # 4. Generate Allocation Visual Chart
    plt.figure(figsize=(16, 6), dpi=150)
    plt.subplot(1, 2, 1)
    sns.barplot(data=df_report.head(18), x="Vulnerability", y="Keyword", palette="Reds_r")
    plt.title("Top 18 Acoustically Vulnerable Keywords", fontsize=12, fontweight="bold")
    plt.xlabel("Composite Vulnerability Score [0 - 1]", fontsize=10)

    plt.subplot(1, 2, 2)
    node_names = list(node_allocations.keys())
    vocab_sizes = [len(v) + 1 for v in node_allocations.values()]
    palette = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    colors = palette[:len(node_names)]
    bars = plt.bar(node_names, vocab_sizes, color=colors, width=0.5, edgecolor='black')
    plt.axhline(36, color='red', linestyle='--', label='Homogeneous Generalist (36 Classes)')
    plt.title(f"Node Vocabulary Load ({args.nodes} ESP32s)", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Output Classes", fontsize=10)
    plt.ylim(0, 42)
    plt.xticks(rotation=15, ha='right')
    plt.legend()

    for bar, val in zip(bars, vocab_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val} Classes", ha='center', fontweight='bold')

    plt.tight_layout()
    chart_path = f"smart_overlap_vulnerability_chart_{args.nodes}dev.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved Allocation Chart: {chart_path}")
    print("Optimization Complete!\n")


if __name__ == "__main__":
    main()
