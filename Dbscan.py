import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs, make_moons, make_circles

# Δημιουργία φακέλου για αποθήκευση εικόνων
output_dir = r"D:\kmeans\output_images"
os.makedirs(output_dir, exist_ok=True)

def save_figure(name):
    filename = os.path.join(output_dir, f"{name}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

# DBSCAN Ανάλυση

def analyze_dbscan(X, dataset_name, eps_values, min_samples_values):
    dbscan_results = {}

    n_rows = len(min_samples_values)
    n_cols = len(eps_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    fig.suptitle(f"{dataset_name} - DBSCAN Parameter Grid", fontsize=16)

    for i, min_samples in enumerate(min_samples_values):
        for j, eps in enumerate(eps_values):
            ax = axes[i, j]
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            outlier_percent = 100 * n_noise / len(labels)

            if n_clusters > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1

            key = f"eps={eps}_min={min_samples}"
            dbscan_results[key] = {
                "Clusters": n_clusters,
                "Silhouette": silhouette,
                "Outliers %": outlier_percent
            }

            scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='Spectral', s=10)
            ax.set_title(f"eps={eps}, min={min_samples}\nSil: {silhouette:.2f}, Outl: {outlier_percent:.1f}%")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(f"{dataset_name}_DBSCAN_grid")
    plt.close()

    return dbscan_results

# Φόρτωση datasets
real_datasets = [
    r"D:\kmeans\datasets\FinalDataset.csv"
    #r"D:\kmeans\datasets\dataset1.txt",
    #r"D:\kmeans\datasets\dataset2.txt",
    #r"D:\kmeans\datasets\dataset3.txt",
    #r"D:\kmeans\datasets\dataset4.csv"
]

#synthetic_datasets = {
 #   "Blobs_4_centers": make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0),
  #  "Blobs_8_centers": make_blobs(n_samples=500, centers=8, random_state=42, cluster_std=0.8),
   # "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
    #"Circles": make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
#}

# Παράμετροι
eps_values = [0.1, 0.2, 0.3, 0.5, 0.7]
min_samples_values = [3, 5, 10]

dbscan_all_results = {}

# Ανάλυση πραγματικών datasets
for dataset_path in real_datasets:
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    df = pd.read_csv(dataset_path, delimiter="\t" if "\t" in open(dataset_path).read() else ",")
    df = df.select_dtypes(include=[np.number])
    data = StandardScaler().fit_transform(df.to_numpy())

    dbscan_all_results[dataset_name] = analyze_dbscan(data, dataset_name, eps_values, min_samples_values)

# Ανάλυση synthetic datasets
#for name, (X, _) in synthetic_datasets.items():
 #   dataset_name = name.replace(" ", "_").replace("(", "").replace(")", "")
  #  X_scaled = StandardScaler().fit_transform(X)
   # dbscan_all_results[dataset_name] = analyze_dbscan(X_scaled, dataset_name, eps_values, min_samples_values)

# Αποθήκευση αποτελεσμάτων
with open(os.path.join(output_dir, "dbscan_results.txt"), "w") as f:
    f.write("=== DBSCAN RESULTS ===\n\n")
    for dataset, results in dbscan_all_results.items():
        f.write(f"--- {dataset} ---\n")
        for setting, metrics in results.items():
            f.write(f"{setting}:\n")
            for metric, val in metrics.items():
                f.write(f"  {metric}: {val:.4f}\n")
        f.write("\n")

print(f"Ολοκληρώθηκε η ανάλυση DBSCAN! Αποτελέσματα και εικόνες αποθηκεύτηκαν στον φάκελο: {output_dir}")
