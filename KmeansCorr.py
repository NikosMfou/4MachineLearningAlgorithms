import os
os.environ["OMP_NUM_THREADS"] = "1"  # Για να αποφύγουμε τις προειδοποιήσεις

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.cm as cm

# Δημιουργία φακέλου για αποθήκευση εικόνων
output_dir = r"D:\kmeans\output_images"
os.makedirs(output_dir, exist_ok=True)

# Λίστα με τα dataset
datasets = [
    r"D:\kmeans\datasets\FinalDataset.csv"
    #r"D:\kmeans\datasets\dataset1.txt",
   #r"D:\kmeans\datasets\dataset2.txt",
    #r"D:\kmeans\datasets\dataset3.txt",
    #r"D:\kmeans\datasets\dataset4.csv"
]

# Synthetic datasets
#synthetic_datasets = {
 #   "Blobs_4_centers": make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0),
  #  "Blobs_8_centers": make_blobs(n_samples=500, centers=8, random_state=42, cluster_std=0.8),
   # "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
    #"Circles": make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
#}

results = {}

def save_figure(name):
    """Αποθήκευση σχήματος"""
    filename = os.path.join(output_dir, f"{name}.png")
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def plot_silhouette(data, labels, k, dataset_name):
    """Σχήμα silhouette"""
    plt.figure(figsize=(8, 6))
    silhouette_vals = silhouette_samples(data, labels)
    y_lower = 10
    
    for i in range(k):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()
        y_upper = y_lower + len(cluster_vals)
        
        if len(cluster_vals) > 0:
            color = cm.nipy_spectral(float(i) / k)
            plt.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            plt.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i+1))
        y_lower = y_upper + 10
    
    if len(silhouette_vals) > 0:
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(x=silhouette_avg, color="red", linestyle="--")
        plt.title(f"Silhouette Plot for k={k}\nAvg Score: {silhouette_avg:.3f}")
    
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster Label")
    plt.yticks([])
    save_figure(f"{dataset_name}_silhouette_k{k}")

def plot_clusters(X, labels, centers, title, dataset_name):
    """Σχήμα clusters"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k', s=50)
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, linewidths=3, color='red', label='Centroids')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    clean_title = title.split('\n')[0].replace(' ', '_').replace(':', '').replace('-', '')
    save_figure(f"{dataset_name}_{clean_title}")

def plot_inertia(kmeans, k, data, dataset_name):
    """Σχήμα inertia"""
    plt.figure(figsize=(8, 6))
    cluster_inertia = []
    for i in range(k):
        cluster_points = data[kmeans.labels_ == i]
        distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
        cluster_inertia.append(np.sum(distances**2))
    
    bars = plt.bar([f"Cluster {i+1}" for i in range(k)], cluster_inertia, color='skyblue')
    plt.xlabel("Clusters")
    plt.ylabel("Inertia Value")
    plt.title(f"Inertia Plot for k={k}")
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
    
    save_figure(f"{dataset_name}_inertia_k{k}")
    return cluster_inertia

def analyze_dataset(X, y, dataset_name, k_values=range(2, 11)):
    """Ανάλυση dataset για όλα τα k"""
    dataset_results = {}
    
    # Σύγκριση όλων των k σε ένα σχήμα
    plt.figure(figsize=(15, 8))
    for i, k in enumerate(k_values, 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        silhouette = silhouette_score(X, labels)
        dataset_results[k] = {
            "Inertia": kmeans.inertia_,
            "Silhouette": silhouette
        }
        
        if k == len(np.unique(y)):
            dataset_results[k].update({
                "ARI": adjusted_rand_score(y, labels),
                "NMI": normalized_mutual_info_score(y, labels)
            })
        
        plt.subplot(2, 5, i)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=30)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=100, color='red')
        plt.title(f'k={k}\nSilhouette: {silhouette:.2f}')
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle(f'K-Means on {dataset_name}', y=1.02)
    plt.tight_layout()
    save_figure(f"{dataset_name}_all_k_comparison")
    
    # Ατομική ανάλυση για κάθε k
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        title = f"{dataset_name} (k={k})"
        if k == len(np.unique(y)):
            title += f"\nARI: {dataset_results[k].get('ARI', 'N/A'):.2f}, NMI: {dataset_results[k].get('NMI', 'N/A'):.2f}"
        else:
            title += f"\nSilhouette: {dataset_results[k]['Silhouette']:.2f}"
        
        plot_clusters(X, labels, kmeans.cluster_centers_, title, dataset_name)
        plot_silhouette(X, labels, k, dataset_name)
        plot_inertia(kmeans, k, X, dataset_name)
    
    # Γραφήματα μετρικών
    plot_metrics(k_values, dataset_results, dataset_name)
    
    return dataset_results

def plot_metrics(k_values, results, dataset_name):
    """Γραφήματα μετρικών αξιολόγησης"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_values, [results[k]["Inertia"] for k in k_values], 'bo-')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    
    ax2.plot(k_values, [results[k]["Silhouette"] for k in k_values], 'go-')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Scores')
    
    plt.suptitle(f'Evaluation Metrics for {dataset_name}', y=1.02)
    plt.tight_layout()
    save_figure(f"{dataset_name}_metrics")

# Ανάλυση πραγματικών datasets
for dataset_path in datasets:
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_csv(dataset_path, delimiter="\t" if "\t" in open(dataset_path).read() else ",")
    
    df = df.select_dtypes(include=[np.number])
    data = StandardScaler().fit_transform(df.to_numpy())

    results[dataset_name] = analyze_dataset(data, np.zeros(len(data)), dataset_name)

# Ανάλυση synthetic datasets
#for name, (X, y) in synthetic_datasets.items():
 #   dataset_name = name.replace(" ", "_").replace("(", "").replace(")", "")
  #  results[dataset_name] = analyze_dataset(X, y, dataset_name)

# Αποθήκευση αποτελεσμάτων
with open(os.path.join(output_dir, "results.txt"), "w") as f:
    f.write("=== FINAL RESULTS ===\n\n")
    for name, metrics in results.items():
        f.write(f"--- {name} ---\n")
        if isinstance(metrics, dict) and 'Inertia' in metrics and isinstance(metrics['Inertia'], list):
            # Real dataset results
            f.write(f"Best k: {metrics['Best k']}\n")
            f.write("Inertia:\n")
            for k, val in zip(range(2, 11), metrics['Inertia']):
                f.write(f"  k={k}: {val:.4f}\n")
            f.write("Silhouette Scores:\n")
            for k, val in zip(range(2, 11), metrics['Silhouette Scores']):
                f.write(f"  k={k}: {val:.4f}\n")
        elif isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
            # Synthetic dataset results
            for k, k_metrics in metrics.items():
                f.write(f"k={k}:\n")
                for metric, value in k_metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        else:
            # Other cases
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
        f.write("\n")

print(f"Ολοκληρώθηκε η ανάλυση! Τα αποτελέσματα και τα γραφήματα αποθηκεύτηκαν στον φάκελο: {output_dir}")