#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki SuperTag Mapping File Generator

This script applies the human-reviewed SuperTag mappings from cluster_supertags.xlsx
to the tag data in supertags_tag_data.csv, and generates mapping files in the format
expected by the BlindWiki application.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set plot style
sns.set_context("paper", font_scale=1.5)

# Add parent directory to path to access utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def load_reviewed_supertags(excel_path):
    """Load human-reviewed SuperTag mappings from Excel file"""
    print(f"Loading reviewed SuperTag mappings from {excel_path}")
    
    # Check if file exists
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    
    # Read Excel file
    df = pd.read_excel(excel_path)
    
    # Verify required columns exist
    required_cols = ['cluster', 'supertag_reviewed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Excel file: {missing_cols}")
    
    # Create mapping from cluster ID to reviewed SuperTag
    cluster_to_supertag = {}
    for _, row in df.iterrows():
        cluster_id = row['cluster']
        supertag = row['supertag_reviewed']
        if pd.notna(supertag) and str(supertag).strip():
            cluster_to_supertag[cluster_id] = str(supertag).strip()
    
    print(f"Loaded {len(cluster_to_supertag)} reviewed SuperTag mappings")
    return cluster_to_supertag


def apply_mapping_directly(df, cluster_to_supertag):
    """Apply the mapping from the Excel file directly to the dataframe in memory
    
    Args:
        df: DataFrame containing the tag data with cluster information
        cluster_to_supertag: Mapping of cluster IDs to SuperTag names
        
    Returns:
        DataFrame with SuperTag information applied
    """
    print("Applying reviewed SuperTag mapping directly to dataframe...")
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Ensure we have a cluster column
    if 'cluster' not in result_df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column")
    
    # Map cluster IDs to SuperTags
    result_df['supertag'] = result_df['cluster'].map(cluster_to_supertag)
    
    # If no mapping was found, keep the original SuperTag if it exists
    if 'supertag' in df.columns:
        # First save the original supertags
        result_df['original_supertag'] = df['supertag']
        # Then fill missing values in the new supertag column
        result_df['supertag'] = result_df['supertag'].fillna(result_df['original_supertag'])
    
    # For any remaining unmapped clusters, use a default naming scheme
    unmapped_clusters = result_df['supertag'].isna()
    if unmapped_clusters.any():
        unmapped_count = unmapped_clusters.sum()
        print(f"Warning: {unmapped_count} tags in {len(result_df[unmapped_clusters]['cluster'].unique())} clusters have no SuperTag mapping")
        
        # Apply a default naming scheme for unmapped clusters
        for cluster_id in result_df.loc[unmapped_clusters, 'cluster'].unique():
            mask = (result_df['cluster'] == cluster_id) & unmapped_clusters
            result_df.loc[mask, 'supertag'] = f"Cluster_{cluster_id}"
    
    # Create tag to SuperTag mapping for output files
    tag_to_supertag = {}
    for _, row in result_df.iterrows():
        if pd.notna(row['tag']) and pd.notna(row['supertag']):
            tag_to_supertag[str(row['tag']).strip()] = str(row['supertag']).strip()
    
    print(f"Applied SuperTag mapping to {len(result_df)} tags ({len(tag_to_supertag)} unique tag-SuperTag pairs)")
    
    return result_df, tag_to_supertag


def generate_output_files(tag_to_supertag, df, output_folder):
    """Generate output files in the format expected by the application"""
    print(f"Generating output files in {output_folder}")
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Generate JSON mapping file
    json_path = os.path.join(output_folder, "supertag_mapping_dict.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(tag_to_supertag, f, ensure_ascii=False, indent=2)
    print(f"Generated mapping JSON file: {json_path}")
    
    # 2. Generate CSV with counts per SuperTag
    counts = Counter()
    for tag, supertag in tag_to_supertag.items():
        # Find tag in dataframe and get its count
        tag_rows = df[df['tag'] == tag]
        if not tag_rows.empty:
            tag_count = tag_rows['count'].sum()
            counts[supertag] += tag_count
    
    # Create DataFrame from counts
    counts_df = pd.DataFrame({
        'supertag': list(counts.keys()),
        'count': list(counts.values())
    })
    
    # Sort by count descending
    counts_df = counts_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    # Save to CSV
    csv_path = os.path.join(output_folder, "supertag_counts.csv")
    counts_df.to_csv(csv_path, index=False)
    print(f"Generated counts CSV file: {csv_path}")
    
    # 3. Generate enhanced tag data with reviewed SuperTags
    enhanced_csv_path = os.path.join(output_folder, "supertags_enhanced_data.csv")
    
    # Save to CSV
    output_df = df.copy()
    output_df = output_df.rename(columns={'supertag_reviewed': 'supertag'})
    output_df.to_csv(enhanced_csv_path, index=False)
    print(f"Generated enhanced tag data CSV file: {enhanced_csv_path}")
    
    return json_path, csv_path, enhanced_csv_path


def create_visualizations(df, output_folder, timestamp):
    """Create visualizations showing clusters with their SuperTag names"""
    print("Creating visualizations with SuperTag names...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if dataframe has the necessary columns
    required_cols = ['x', 'y', 'cluster', 'supertag']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns for visualization: {[col for col in required_cols if col not in df.columns]}")
        return []
    
    # Extract data needed for visualization
    cluster_labels = df['cluster'].values
    embeddings_2d = df[['x', 'y']].values
    supertags = df['supertag'].values

    # Create a scatter plot of clusters with SuperTag names
    plt.figure(figsize=(24, 20))
    
    # Create a palette with distinct colors + black for noise
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    colors = sns.color_palette("husl", n_colors=n_clusters)
    palette = {i: colors[i] for i in range(n_clusters)}
    palette[-1] = (0.1, 0.1, 0.1)  # Black for noise points
    
    # Plot the points
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels,
        palette=palette,
        s=80,
        alpha=0.7,
        legend=None
    )
    
    # Create a mapping of cluster to SuperTag name
    cluster_to_supertag = {}
    for i, row in df.groupby('cluster').first().iterrows():
        cluster_to_supertag[i] = row['supertag']
    
    # Add SuperTag labels at the centroid of each cluster
    for cluster_id, supertag in cluster_to_supertag.items():
        # Skip noise cluster
        if cluster_id == -1:
            continue
        
        # Get coordinates of points in this cluster
        cluster_points = embeddings_2d[cluster_labels == cluster_id]
        if len(cluster_points) > 0:
            # Calculate centroid of the cluster
            centroid_x = np.mean(cluster_points[:, 0])
            centroid_y = np.mean(cluster_points[:, 1])
            
            # Add text label
            plt.text(
                centroid_x, centroid_y,
                supertag,
                fontsize=14,
                weight='bold',
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
            )
    
    # Title and labels
    plt.title('Tag Clusters with SuperTag Names', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16)
    plt.ylabel('UMAP Dimension 2', fontsize=16)
    
    # Save the figure
    supertagged_map_path = os.path.join(output_folder, "supertag_clusters.png")
    plt.tight_layout()
    plt.savefig(supertagged_map_path, dpi=300)
    plt.close()
    
    # Create a second figure with tag samples
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
    
    # Add text labels with the SuperTag name and sample of tags
    for cluster_id, supertag in cluster_to_supertag.items():
        # Skip noise cluster
        if cluster_id == -1:
            continue
        
        # Get points in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_points = embeddings_2d[cluster_mask]
        if len(cluster_points) > 0:
            # Calculate centroid of the cluster
            centroid_x = np.mean(cluster_points[:, 0])
            centroid_y = np.mean(cluster_points[:, 1])
            
            # Get a sample of tags from this cluster
            cluster_tags = df.loc[cluster_mask, 'tag'].values
            sample_size = min(3, len(cluster_tags))
            sample_tags = np.random.choice(cluster_tags, size=sample_size, replace=False)
            
            # Create label text with SuperTag and sample tags
            label_text = f"{supertag}:\n" + "\n".join(sample_tags)
            
            # Add text label
            plt.text(
                centroid_x, centroid_y,
                label_text,
                fontsize=10,
                ha='center',
                va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5')
            )
    
    # Title and labels
    plt.title('SuperTags with Example Tags', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16)
    plt.ylabel('UMAP Dimension 2', fontsize=16)
    
    # Save the figure
    tag_samples_path = os.path.join(output_folder, "supertag_samples.png")
    plt.tight_layout()
    plt.savefig(tag_samples_path, dpi=300)
    plt.close()
    
    # Create a figure with original tag names on data points
    plt.figure(figsize=(24, 20))
    
    # Plot all points first
    sns.scatterplot(
        x=embeddings_2d[:, 0],
        y=embeddings_2d[:, 1],
        hue=cluster_labels,
        palette=palette,
        s=70,
        alpha=0.4,
        legend=None
    )
    
    # Use a density-based approach to avoid label cluttering
    # 1. Compute pairwise distances between points
    
    # Determine the number of neighbors to consider for density estimation
    n_neighbors = min(20, len(embeddings_2d) - 1)
    
    # Construct nearest neighbors model
    neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    neighbors.fit(embeddings_2d)
    
    # Find distances to neighbors for each point
    distances, _ = neighbors.kneighbors(embeddings_2d)
    
    # Calculate a density score (average distance to neighbors)
    # Higher value means more space around the point (less dense)
    density_scores = np.mean(distances, axis=1)
    
    # Normalize scores to 0-1 range
    density_scores = (density_scores - np.min(density_scores)) / (np.max(density_scores) - np.min(density_scores))
    
    # Determine selection probability based on density
    # Higher density (lower score) means lower probability of selection
    selection_probs = density_scores ** 2  # Square to emphasize the effect
    
    # Apply a minimum threshold to avoid too few labels
    min_prob = 0.1
    selection_probs = np.maximum(selection_probs, min_prob)
    
    # Apply probability selection
    selected_mask = np.random.random(len(df)) < selection_probs
    
    # Ensure at least one tag from each cluster is selected
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        if not any(selected_mask & cluster_mask) and any(cluster_mask):
            # If no point selected from this cluster, select the most isolated one
            cluster_indices = np.where(cluster_mask)[0]
            most_isolated_idx = cluster_indices[np.argmax(density_scores[cluster_mask])]
            selected_mask[most_isolated_idx] = True
    
    # Get the selected points
    sample_df = df[selected_mask]
    print(f"Selected {len(sample_df)} out of {len(df)} tags for labeling ({100*len(sample_df)/len(df):.1f}%)")
    
    # Add text labels directly on the sampled points
    for idx, row in sample_df.iterrows():
        # Skip long tag names (exceeding 25 characters)
        if len(str(row['tag'])) > 25:
            continue
        
        plt.text(
            row['x'], row['y'],
            row['tag'],
            fontsize=12,
            ha='center',
            va='center',
            alpha=0.9,
            bbox=dict(facecolor='white', alpha=0.4, edgecolor=None, boxstyle='round,pad=0.1')
        )
    
    # Title and labels
    plt.title('Original Tag Names (Density-Based Sampling)', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16)
    plt.ylabel('UMAP Dimension 2', fontsize=16)
    
    # Save the figure
    tag_names_path = os.path.join(output_folder, "original_tag_names.png")
    plt.tight_layout()
    plt.savefig(tag_names_path, dpi=300)
    plt.close()
    
    # Create a detailed visualization for top SuperTags
    plt.figure(figsize=(20, 16))
    
    # Get most common SuperTags by count
    supertag_counts = df.groupby('supertag')['count'].sum().sort_values(ascending=False)
    top_supertags = supertag_counts.head(20).index.tolist()
    
    # Create a new color palette for SuperTags
    supertag_palette = dict(zip(top_supertags, sns.color_palette("tab20", n_colors=len(top_supertags))))
    
    # Create a new dataframe containing only tags with top SuperTags
    top_mask = df['supertag'].isin(top_supertags)
    
    # Plot points from top SuperTags with their colors
    for supertag in top_supertags:
        supertag_mask = df['supertag'] == supertag
        plt.scatter(
            df.loc[supertag_mask, 'x'],
            df.loc[supertag_mask, 'y'],
            s=80, 
            alpha=0.7,
            color=supertag_palette[supertag],
            label=f"{supertag} ({supertag_counts[supertag]})"
        )
    
    # Plot other points in gray
    other_mask = ~top_mask
    if any(other_mask):
        plt.scatter(
            df.loc[other_mask, 'x'],
            df.loc[other_mask, 'y'],
            s=50,
            alpha=0.2,
            color='gray',
            label=f"Other ({other_mask.sum()} tags)"
        )
    
    # Title and labels
    plt.title('Top 20 SuperTags by Usage Count', fontsize=20)
    plt.xlabel('UMAP Dimension 1', fontsize=16)
    plt.ylabel('UMAP Dimension 2', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the figure
    top_supertags_path = os.path.join(output_folder, "top_supertags.png")
    plt.tight_layout()
    plt.savefig(top_supertags_path, dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_folder}")
    return [supertagged_map_path, tag_samples_path, tag_names_path, top_supertags_path]


def load_original_cluster_data(cluster_data_path):
    """Load the original cluster data"""
    print(f"Loading original cluster data from {cluster_data_path}")
    
    # Check if file exists
    if not os.path.exists(cluster_data_path):
        print(f"Warning: Original cluster data file not found: {cluster_data_path}")
        return None
    
    # Read CSV file
    try:
        df = pd.read_csv(cluster_data_path)
        print(f"Loaded {len(df)} tags from original cluster data")
        return df
    except Exception as e:
        print(f"Error loading original cluster data: {e}")
        return None


def main():
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set input and output paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "5-cluster_supertags_review_file.xlsx")
    tag_data_path = os.path.join(script_dir, "data", "supertags_tag_data.csv")
    output_folder = os.path.join(script_dir, "..", "mappings")
    figures_folder = os.path.join(script_dir, "figures")
    cluster_data_path = os.path.join(script_dir, "figures/cluster_data.csv")
    
    # Command line arguments can override default paths
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    if len(sys.argv) > 2:
        tag_data_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_folder = sys.argv[3]
    if len(sys.argv) > 4:
        figures_folder = sys.argv[4]
    
    try:
        # Load reviewed SuperTag mappings
        cluster_to_supertag = load_reviewed_supertags(excel_path)
        
        # Load the data
        print(f"Loading tag data from {tag_data_path}")
        if not os.path.exists(tag_data_path):
            # Try to load from cluster data directly if tag data doesn't exist
            if os.path.exists(cluster_data_path):
                print(f"Tag data not found, loading from cluster data instead: {cluster_data_path}")
                df = pd.read_csv(cluster_data_path)
                if 'text' in df.columns and 'tag' not in df.columns:
                    df = df.rename(columns={'text': 'tag'})
            else:
                raise FileNotFoundError(f"Neither tag data ({tag_data_path}) nor cluster data ({cluster_data_path}) file found")
        else:
            df = pd.read_csv(tag_data_path)
        
        # Apply mapping directly to dataframe
        df_mapped, tag_to_supertag = apply_mapping_directly(df, cluster_to_supertag)
        
        # Generate output files
        json_path, csv_path, enhanced_csv_path = generate_output_files(tag_to_supertag, df_mapped, output_folder)
        
        # Create figures - using the mapped dataframe directly
        if 'x' in df_mapped.columns and 'y' in df_mapped.columns:
            # We already have coordinates in the dataframe
            fig_paths = create_visualizations(df_mapped, figures_folder, timestamp)
            if fig_paths:
                print(f"Generated {len(fig_paths)} visualization figures")
        else:
            # We need to get coordinates from the original cluster data
            orig_df = load_original_cluster_data(cluster_data_path)
            if orig_df is not None and 'x' in orig_df.columns and 'y' in orig_df.columns:
                # Merge the coordinates into our mapped dataframe
                if 'text' in orig_df.columns and 'tag' not in orig_df.columns:
                    orig_df = orig_df.rename(columns={'text': 'tag'})
                
                # Use the coordinates from orig_df but keep the SuperTag mappings from df_mapped
                vis_df = orig_df.copy()
                vis_df['supertag'] = vis_df['cluster'].map({row['cluster']: row['supertag'] for _, row in df_mapped.iterrows()})
                
                # Create the visualizations
                fig_paths = create_visualizations(vis_df, figures_folder, timestamp)
                if fig_paths:
                    print(f"Generated {len(fig_paths)} visualization figures")
            else:
                print("Warning: Could not create visualizations because coordinate data is missing")
        
        print("\nSummary:")
        print(f"- Applied {len(cluster_to_supertag)} reviewed SuperTag mappings")
        print(f"- Created mapping for {len(tag_to_supertag)} tags")
        print(f"- Generated {len(set(tag_to_supertag.values()))} SuperTag categories")
        print(f"\nOutput files:")
        print(f"- Mapping JSON: {json_path}")
        print(f"- Counts CSV: {csv_path}")
        print(f"- Enhanced tag data: {enhanced_csv_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
