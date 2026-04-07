import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan
from sklearn.datasets import make_blobs, make_moons, make_circles

# === ΡΥΘΜΙΣΕΙΣ ===
output_dir = r"D:\kmeans\output_images"
os.makedirs(output_dir, exist_ok=True)

# === ΟΡΙΣΜΟΣ REAL DATASETS ===
real_datasets_paths = [
    r"D:\kmeans\datasets\dataset1.txt",
    r"D:\kmeans\datasets\dataset2.txt",
    r"D:\kmeans\datasets\dataset3.txt",
    r"D:\kmeans\datasets\dataset4.csv"
]

# === ΟΡΙΣΜΟΣ SYNTHETIC DATASETS ===
synthetic_datasets = {
    "Blobs_4_centers": make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0),
    "Blobs_8_centers": make_blobs(n_samples=500, centers=8, random_state=42, cluster_std=0.8),
    "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
    "Circles": make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
}

# === ΦΟΡΤΩΣΗ REAL DATASETS ===
def load_real_datasets(file_paths):
    dataset_dict = {}
    for path in file_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path, delimiter="\t")
            if df.shape[1] == 1:
                df = pd.read_csv(path, delimiter=",")
            print(f"[INFO] {name}: shape = {df.shape}, columns = {df.columns.tolist()}")
            df = df.select_dtypes(include=["number"])
            if df.shape[1] == 0:
                print(f"[ΠΡΟΕΙΔΟΠΟΙΗΣΗ] Το '{name}' δεν έχει αριθμητικές στήλες. Παραλείπεται.")
                continue
            dataset_dict[name] = df.to_numpy()
        except Exception as e:
            print(f"[ΣΦΑΛΜΑ] Σφάλμα στο αρχείο '{name}': {e}")
    return dataset_dict

# === HDBSCAN ΑΝΑΛΥΣΗ ===
def analyze_hdbscan(X, dataset_name, min_cluster_sizes):
    hdbscan_results = {}
    best_score = -1
    best_setting = None

    total_cols = len(min_cluster_sizes)
    fig = plt.figure(figsize=(total_cols * 4, 8))
    plt.suptitle(f"{dataset_name} - HDBSCAN Results", fontsize=16)

    for i, min_cluster_size in enumerate(min_cluster_sizes):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(X)

        # Πρώτη σειρά: clustering scatter plot
        ax1 = fig.add_subplot(2, total_cols, i + 1)
        ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=10)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        outlier_percent = 100 * n_noise / len(labels)
        silhouette = silhouette_score(X, labels) if n_clusters > 1 else -1

        key = f"min_cluster_size={min_cluster_size}"
        hdbscan_results[key] = {
            "Clusters": n_clusters,
            "Silhouette": silhouette,
            "Outliers %": outlier_percent
        }

        if silhouette > best_score:
            best_score = silhouette
            best_setting = key

        ax1.set_title(f"min={min_cluster_size}\nSil: {silhouette:.2f}, Outl: {outlier_percent:.1f}%")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.grid(True)

        # Δεύτερη σειρά: condensed tree plot
        ax2 = fig.add_subplot(2, total_cols, total_cols + i + 1)
        plt.sca(ax2)
        clusterer.condensed_tree_.plot(select_clusters=True)
        ax2.set_title("Condensed Tree")


        ax2.set_title("Condensed Tree")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, f"{dataset_name}_HDBSCAN_full.png"), dpi=150)
    plt.close()

    print(f"[OK] {dataset_name} → Best: {best_setting} (Silhouette: {best_score:.3f})")
    return hdbscan_results

# === ΠΑΡΑΜΕΤΡΟΙ ===
min_cluster_sizes = [5, 10, 20]
hdbscan_all_results = {}

# === ΕΚΤΕΛΕΣΗ REAL DATASETS ===
all_real_datasets = load_real_datasets(real_datasets_paths)

for dataset_name, data in all_real_datasets.items():
    data_scaled = StandardScaler().fit_transform(data)
    hdbscan_all_results[dataset_name] = analyze_hdbscan(data_scaled, dataset_name, min_cluster_sizes)

# === ΕΚΤΕΛΕΣΗ SYNTHETIC DATASETS ===
for name, (X, _) in synthetic_datasets.items():
    dataset_name = name.replace(" ", "_")
    X_scaled = StandardScaler().fit_transform(X)
    hdbscan_all_results[dataset_name] = analyze_hdbscan(X_scaled, dataset_name, min_cluster_sizes)

# === ΑΠΟΘΗΚΕΥΣΗ ΑΠΟΤΕΛΕΣΜΑΤΩΝ ===
with open(os.path.join(output_dir, "hdbscan_results.txt"), "w") as f:
    f.write("=== HDBSCAN RESULTS ===\n\n")
    for dataset, results in hdbscan_all_results.items():
        f.write(f"--- {dataset} ---\n")
        for setting, metrics in results.items():
            f.write(f"{setting}:\n")
            for metric, val in metrics.items():
                f.write(f"  {metric}: {val:.4f}\n")
        f.write("\n")

print(f"[FINISHED] Combined plots saved in: {output_dir}")
