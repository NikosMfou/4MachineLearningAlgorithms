import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_samples
from sklearn.datasets import make_blobs, make_moons, make_circles
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

# ===== CONFIGURATION =====
# Input paths
dataset_paths = [
    r"D:\\kmeans\\datasets\\dataset1.txt",
    r"D:\\kmeans\\datasets\\dataset2.txt",
    r"D:\\kmeans\\datasets\\dataset3.txt",
    r"D:\\kmeans\\datasets\\dataset4.csv"
]

# Synthetic datasets
synthetic_datasets = {
    "Blobs_4_centers": make_blobs(n_samples=500, centers=4, random_state=42, cluster_std=1.0),
    "Blobs_8_centers": make_blobs(n_samples=500, centers=8, random_state=42, cluster_std=0.8),
    "Moons": make_moons(n_samples=500, noise=0.05, random_state=42),
    "Circles": make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42),
}

# Output directories
output_dir_hca = r"D:\\hca\\output_images"
output_dir_comparison = r"D:\\hca\\comparison_results"
os.makedirs(output_dir_hca, exist_ok=True)
os.makedirs(output_dir_comparison, exist_ok=True)

# Analysis parameters
k_values = range(2, 11)
linkage_methods = ['single', 'complete', 'average', 'ward', 'centroid']

# ===== UTILITY FUNCTIONS =====
def save_figure(name, output_dir):
    """Save figure to specified output directory"""
    plt.savefig(os.path.join(output_dir, f"{name}.png"), bbox_inches='tight', dpi=150)
    plt.close()

def plot_silhouette(data, labels, k, dataset_name, method, output_dir):
    """Plot silhouette diagram"""
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1:
        print(f"Skipping silhouette plot for {dataset_name} with {method} - only 1 cluster found")
        return

    plt.figure(figsize=(8, 6))
    silhouette_vals = silhouette_samples(data, labels)
    y_lower = 10

    for i in range(k):
        cluster_vals = silhouette_vals[labels == i]
        cluster_vals.sort()
        y_upper = y_lower + len(cluster_vals)
        color = cm.nipy_spectral(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i+1))
        y_lower = y_upper + 10

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.title(f"Silhouette for {method} (k={k})\nAvg: {silhouette_avg:.3f}")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster Label")
    plt.yticks([])
    save_figure(f"{dataset_name}_silhouette_{method}_k{k}", output_dir)

def plot_clusters(X, labels, title, dataset_name, method, output_dir):
    """Plot cluster visualization"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k', s=50)
    plt.title(title)
    plt.grid(True)
    save_figure(f"{dataset_name}_clusters_{method}_k{len(np.unique(labels))}", output_dir)

def plot_dendrogram(X, method, dataset_name, output_dir):
    """Plot dendrogram for hierarchical clustering"""
    Z = linkage(X, method=method)
    plt.figure(figsize=(15, 6))
    dendrogram(Z)
    plt.title(f"Dendrogram - {dataset_name} ({method})")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    save_figure(f"{dataset_name}_dendrogram_{method}", output_dir)

def compute_metrics(X, labels, true_labels=None):
    """Compute clustering metrics"""
    metrics = {}

    # Silhouette score (only if more than 1 cluster)
    if len(np.unique(labels)) > 1:
        metrics['Silhouette'] = silhouette_score(X, labels)
    else:
        metrics['Silhouette'] = -1

    # External metrics if true labels are provided
    if true_labels is not None and len(np.unique(true_labels)) > 1:
        metrics['ARI'] = adjusted_rand_score(true_labels, labels)
        metrics['NMI'] = normalized_mutual_info_score(true_labels, labels)

    return metrics

def run_kmeans(X, k, random_state=42):
    """Run K-Means clustering"""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    return {
        'labels': labels,
        'inertia': kmeans.inertia_,
        'centers': kmeans.cluster_centers_
    }

def analyze_dbscan(X):
    """Analyze dataset with DBSCAN to find outliers and optimal eps"""
    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, -1], axis=0)
    eps = distances[int(0.95 * len(distances))]  # 95th percentile

    dbscan = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = dbscan.labels_

    return {
        'eps': eps,
        'outliers': np.sum(labels == -1),
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
    }

def find_optimal_k(X, max_k=10, linkage='ward'):
    """Find optimal K using silhouette score"""
    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        try:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"Skipping k={k} for linkage={linkage} due to error: {e}")

    return best_k

def generate_summary_table(all_datasets, all_results, comparison_results):
    """Generate comprehensive summary table for all datasets"""
    summary_data = []

    for dataset_name in all_datasets.keys():
        X = all_datasets[dataset_name][0] if isinstance(all_datasets[dataset_name], tuple) else all_datasets[dataset_name]
        X_scaled = StandardScaler().fit_transform(X)

        # DBSCAN analysis
        dbscan = analyze_dbscan(X_scaled)

        # Find optimal k for K-Means
        optimal_k_kmeans = find_optimal_k(X_scaled)

        # Find best HCA method and optimal k
        best_method = None
        best_avg_silhouette = -1
        optimal_k_hca = None

        if dataset_name in all_results:  # For real datasets
            for method in all_results[dataset_name].keys():
                avg_silhouette = np.mean([all_results[dataset_name][method][k].get('Silhouette', -1) 
                                       for k in all_results[dataset_name][method]])

                if avg_silhouette > best_avg_silhouette:
                    best_avg_silhouette = avg_silhouette
                    best_method = method
        else:  # For synthetic datasets not in all_results
            best_method = 'ward'
            optimal_k_hca = find_optimal_k(X_scaled, linkage=best_method)
            best_avg_silhouette = silhouette_score(X_scaled, AgglomerativeClustering(n_clusters=optimal_k_hca, linkage=best_method).fit_predict(X_scaled))

        # K-Means metrics
        kmeans = KMeans(n_clusters=optimal_k_kmeans, random_state=42).fit(X_scaled)
        kmeans_silhouette = silhouette_score(X_scaled, kmeans.labels_)

        # DBSCAN metrics for K-Means
        dbscan_kmeans = analyze_dbscan(X_scaled)

        summary_data.append({
            'Dataset': dataset_name,
            'Optimal K (K-Means)': optimal_k_kmeans,
            'Optimal K (HCA)': optimal_k_hca,
            'Best HCA Method': best_method,
            'Silhouette (HCA)': round(best_avg_silhouette, 3),
            'Silhouette (K-Means)': round(kmeans_silhouette, 3),
            'Outliers (DBSCAN)': dbscan['outliers'],
            'DBSCAN eps': round(dbscan['eps'], 3),
            'DBSCAN Clusters': dbscan['n_clusters'],
            'K-Means Inertia': round(kmeans.inertia_, 2),
            'DBSCAN Outliers (K-Means)': dbscan_kmeans['outliers'],
            'DBSCAN eps (K-Means)': round(dbscan_kmeans['eps'], 3),
            'DBSCAN Clusters (K-Means)': dbscan_kmeans['n_clusters'],
        })

    return pd.DataFrame(summary_data)

# ===== MAIN ANALYSIS =====
def main():
    all_datasets = {}
    all_results = {}
    comparison_results = {}

    # Load real datasets
    for path in dataset_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, delimiter="\t" if "\t" in open(path).read() else ",")
        df = df.select_dtypes(include=[np.number])
        all_datasets[name] = df.values

    # Add synthetic datasets
    for name, (X, y) in synthetic_datasets.items():
        all_datasets[name] = X

    # Perform analysis and generate summary table
    summary_table = generate_summary_table(all_datasets, all_results, comparison_results)

    # Save and display results
    summary_table.to_csv(os.path.join(output_dir_comparison, "summary_table.csv"), index=False)
    print("\nSummary table saved to:", os.path.join(output_dir_comparison, "summary_table.csv"))
    print("\nMarkdown version:\n")
    print(summary_table.to_markdown(index=False))

if __name__ == "__main__":
    main()
