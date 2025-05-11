#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki Tag Clustering

This script loads tag embeddings, performs HDBSCAN clustering,
and creates visualizations of the discovered clusters.
"""

# %% [REGION 1] Imports and setup
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import umap
import hdbscan
from datetime import datetime

# Set plot style
sns.set_context("paper", font_scale=1.5)

# %% [REGION 2] Load embeddings
def load_embeddings(embeddings_path):
    """Load embeddings from NPZ file"""
    print(f"Loading embeddings from {embeddings_path}")
    
    # Check if file exists
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    # Load the embeddings
    data = np.load(embeddings_path, allow_pickle=True)
    
    # Extract components
    embeddings = data['embeddings']
    texts = data['texts']
    tag_ids = data['tag_ids'] if 'tag_ids' in data else None
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    return embeddings, texts, tag_ids

# %% [REGION 3] Load tag counts from CSV
def load_tag_counts(csv_path):
    """Load tag counts from the original CSV file"""
    print(f"Loading tag counts from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Verify 'count' column exists
    if 'count' not in df.columns:
        raise ValueError("CSV file doesn't contain the 'count' column")
    
    # Create a mapping from tag_id to count
    tag_count_map = dict(zip(df['id'], df['count']))
    
    print(f"Loaded counts for {len(tag_count_map)} tags")
    
    return tag_count_map, df

# %% [REGION 3] Dimensionality reduction
def reduce_dimensions(embeddings, n_components=2, n_neighbors=15, min_dist=0.1):
    """Reduce dimensionality for visualization"""
    print(f"Reducing dimensions with UMAP (n_components={n_components})")
    
    # First reduce with PCA to speed up UMAP
    print("Applying initial PCA...")
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings)
        print(f"PCA reduced dimensions from {embeddings.shape[1]} to {embeddings_pca.shape[1]}")
    else:
        embeddings_pca = embeddings
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )
    
    embeddings_2d = reducer.fit_transform(embeddings_pca)
    print(f"UMAP reduced dimensions to {embeddings_2d.shape[1]}")
    
    return embeddings_2d

# %% [REGION 4] HDBSCAN clustering
def perform_clustering(embeddings_2d, min_cluster_size=2, min_samples=None):
    """Perform HDBSCAN clustering on reduced embeddings"""
    print(f"Performing HDBSCAN clustering (min_cluster_size={min_cluster_size})")
    
    # Initialize HDBSCAN with adjusted parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    # Fit the model
    cluster_labels = clusterer.fit_predict(embeddings_2d)
    
    # Count clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    return cluster_labels, n_clusters, n_noise

# %% [REGION 5] Visualization
def create_visualizations(embeddings_2d, cluster_labels, texts, output_dir, min_cluster_size):
    """Create and save visualizations of clusters"""
    print("Creating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a scatter plot of clusters
    plt.figure(figsize=(20, 16))
    
    # Create a palette with distinct colors + black for noise
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    colors = sns.color_palette("husl", n_colors=n_clusters)
    palette = {i: colors[i] for i in range(n_clusters)}
    palette[-1] = (0.1, 0.1, 0.1)  # Black for noise points
    
    # Plot the points
    scatter = sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels,
        palette=palette,
        s=100,
        alpha=0.7,
        legend="full"
    )
    
    # Title and labels
    plt.title(f'HDBSCAN Clustering (min_cluster_size={min_cluster_size})', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure
    scatter_path = os.path.join(output_dir, "clusters_scatter.png")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    
    # Create a second plot with text labels for a sample of points
    plt.figure(figsize=(24, 20))
    
    # Plot all points
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels,
        palette=palette,
        s=70,
        alpha=0.4,
        legend=None
    )
    
    # Add text labels for a sample of points from each cluster
    clusters = set(cluster_labels)
    for cluster in clusters:
        # Get indices of points in this cluster
        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        
        # Sample a limited number of points for labeling
        max_labels = 5 if cluster != -1 else 3  # Fewer labels for noise
        sample_size = min(max_labels, len(indices))
        sample_indices = np.random.choice(indices, size=sample_size, replace=False)
        
        # Add text labels
        for idx in sample_indices:
            plt.text(
                embeddings_2d[idx, 0], 
                embeddings_2d[idx, 1],
                texts[idx],
                fontsize=10 if cluster != -1 else 8,
                alpha=0.8 if cluster != -1 else 0.5
            )
    
    # Title and labels
    plt.title(f'Tag Cluster Labels (min_cluster_size={min_cluster_size})', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Save the figure
    labels_path = os.path.join(output_dir, "clusters_labeled.png")
    plt.tight_layout()
    plt.savefig(labels_path, dpi=300)
    plt.close()
    
    # Create a heatmap showing cluster density (2D histogram)
    plt.figure(figsize=(18, 16))
    
    # Create a 2D histogram
    h, xedges, yedges = np.histogram2d(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        bins=50
    )
    
    # Create a heatmap
    plt.imshow(
        h.T,
        origin='lower',
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='viridis'
    )
    
    # Add a colorbar
    plt.colorbar(label='Point Density')
    
    # Title and labels
    plt.title('Tag Embedding Density Map', fontsize=18)
    plt.xlabel('UMAP Dimension 1', fontsize=14)
    plt.ylabel('UMAP Dimension 2', fontsize=14)
    
    # Save the figure
    heatmap_path = os.path.join(output_dir, "density_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")
    return [scatter_path, labels_path, heatmap_path]

# %% [REGION 6] Save cluster data
def save_cluster_data(embeddings_2d, cluster_labels, texts, tag_ids, tag_counts, output_dir):
    """Save cluster data to CSV file"""
    print("Saving cluster data to CSV...")
    
    # Create DataFrame with cluster information
    data = {
        'tag_id': tag_ids if tag_ids is not None else range(len(texts)),
        'text': texts,
        'cluster': cluster_labels,
        'count': [tag_counts.get(tag_id, 0) for tag_id in tag_ids] if tag_ids is not None else [0] * len(texts),
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    }
    
    df = pd.DataFrame(data)
    
    # Count tags per cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    print("\nTags per cluster:")
    print(cluster_counts)
    
    # Calculate sum of tag counts per cluster
    cluster_tag_counts = df.groupby('cluster')['count'].sum().sort_index()
    print("\nSum of tag counts per cluster:")
    print(cluster_tag_counts)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "cluster_data.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Generate summary with cluster details
    summary_path = os.path.join(output_dir, "cluster_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Total tags: {len(df)}\n")
        f.write(f"Number of clusters: {len(cluster_counts) - (1 if -1 in cluster_counts.index else 0)}\n")
        f.write(f"Noise points: {cluster_counts.get(-1, 0)}\n\n")
        
        f.write("Tags per cluster:\n")
        for cluster, count in cluster_counts.items():
            if cluster == -1:
                f.write(f"  Noise: {count} tags\n")
            else:
                f.write(f"  Cluster {cluster}: {count} tags\n")
        
        f.write("\nSum of tag counts per cluster:\n")
        for cluster, count in cluster_tag_counts.items():
            if cluster == -1:
                f.write(f"  Noise: {count} total usages\n")
            else:
                f.write(f"  Cluster {cluster}: {count} total usages\n")
        
        f.write("\n\nSample tags for each cluster:\n")
        for cluster in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster]
            cluster_tags = cluster_df['text'].values
            sample_indices = np.random.choice(range(len(cluster_df)), size=min(5, len(cluster_df)), replace=False)
            sample_rows = cluster_df.iloc[sample_indices]
            
            if cluster == -1:
                f.write(f"\nNoise (-1) sample tags:\n")
            else:
                f.write(f"\nCluster {cluster} sample tags:\n")
            
            for _, row in sample_rows.iterrows():
                f.write(f"  - {row['text']} (count: {row['count']})\n")
    
    print(f"Cluster data saved to {csv_path}")
    print(f"Cluster summary saved to {summary_path}")
    
    return csv_path, summary_path

# %% [REGION 7] Main execution
def main():
   
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_path = os.path.join(current_dir, "../../embeddings/all_tags_translated_text_paraphrase-multilingual-mpnet-base-v2_embeddings.npz")
    csv_path = os.path.join(current_dir, "../mappings/all_tags_counts_translated.csv")
    output_dir = os.path.join(current_dir, "figures")
    
    # Parameters
    min_cluster_size = 10
    
    # Load embeddings
    embeddings, texts, tag_ids = load_embeddings(embeddings_path)
    
    # Load tag counts
    tag_count_map, _ = load_tag_counts(csv_path)
    
    # Reduce dimensions visualization
    embeddings_2d = reduce_dimensions(embeddings)
    
    # Perform clustering
    cluster_labels, n_clusters, n_noise = perform_clustering(embeddings_2d, min_cluster_size=min_cluster_size)
    
    # Create visualizations
    vis_paths = create_visualizations(embeddings_2d, cluster_labels, texts, output_dir, min_cluster_size)
    
    # Save cluster data with tag counts
    data_paths = save_cluster_data(embeddings_2d, cluster_labels, texts, tag_ids, tag_count_map, output_dir)
    
    # Print summary to console
    print(f"\nClustering completed successfully!")
    print(f"Found {n_clusters} clusters and {n_noise} noise points")
    
    # Calculate and print aggregated count sums for each cluster
    df = pd.DataFrame({
        'tag_id': tag_ids,
        'cluster': cluster_labels,
        'count': [tag_count_map.get(tag_id, 0) for tag_id in tag_ids]
    })
    
    # Calculate sum of counts by cluster
    cluster_sums = df.groupby('cluster')['count'].sum().sort_index()
    cluster_tag_counts = df['cluster'].value_counts().sort_index()
    
    print("\nAggregated results for each cluster:")
    print("Cluster ID | # of Tags | Total Usages")
    print("-" * 40)
    total_count = 0
    total_tags = 0
    for cluster in cluster_sums.index:
        total_count += cluster_sums[cluster]
        total_tags += cluster_tag_counts[cluster]
        cluster_name = f"Cluster {cluster:2d}" if cluster != -1 else "Noise (-1)"
        print(f"{cluster_name:10} | {cluster_tag_counts[cluster]:9d} | {cluster_sums[cluster]:12d}")
    
    print("-" * 40)
    print(f"TOTAL       | {total_tags:9d} | {total_count:12d}")
    print(f"\nAll outputs saved to {output_dir}")

# %% [REGION 8] Execute if run directly
if __name__ == "__main__":
    main()
