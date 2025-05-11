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


def apply_mapping_to_tags(tag_data_path, cluster_to_supertag):
    """Apply the reviewed SuperTag mappings to the tag data"""
    print(f"Applying mappings to tag data from {tag_data_path}")
    
    # Check if file exists
    if not os.path.exists(tag_data_path):
        raise FileNotFoundError(f"Tag data file not found: {tag_data_path}")
    
    # Read tag data CSV
    df = pd.read_csv(tag_data_path)
    
    # Verify required columns exist
    required_cols = ['tag', 'count', 'cluster']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in tag data file: {missing_cols}")
    
    # Create mapping from tag to SuperTag
    tag_to_supertag = {}
    tags_not_mapped = []
    
    # Add reviewed supertag column
    df['supertag_reviewed'] = df['cluster'].map(cluster_to_supertag)
    
    # Fill missing values with original supertag
    if 'supertag' in df.columns:
        df['supertag_reviewed'] = df['supertag_reviewed'].fillna(df['supertag'])
    
    # Create tag to SuperTag mapping
    for _, row in df.iterrows():
        tag = row['tag']
        if pd.isna(tag) or not str(tag).strip():
            continue
            
        supertag = row['supertag_reviewed']
        if pd.isna(supertag) or not str(supertag).strip():
            tags_not_mapped.append(tag)
            continue
            
        tag_to_supertag[str(tag).strip()] = str(supertag).strip()
    
    print(f"Created mapping for {len(tag_to_supertag)} tags")
    if tags_not_mapped:
        print(f"Warning: {len(tags_not_mapped)} tags couldn't be mapped")
    
    return tag_to_supertag, df


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
        plt.text(
            row['x'], row['y'],
            row['tag'],
            fontsize=10,
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
    excel_path = os.path.join(script_dir, "cluster_supertags.xlsx")
    tag_data_path = os.path.join(script_dir, "supertags_tag_data.csv")
    output_folder = os.path.join(script_dir, "..", "mappings")
    figures_folder = os.path.join(script_dir, "figures")
    cluster_data_path = os.path.join(script_dir, "figures_20250511_204225/cluster_data_20250511_204225.csv")
    
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
        cluster_to_supertag = load_reviewed_suertags(excel_path)
        
        # Apply mappings to tag data
        tag_to_supertag, df = apply_mapping_to_tags(tag_data_path, cluster_to_supertag)
        
        # Generate output files
        json_path, csv_path, enhanced_csv_path = generate_output_files(tag_to_supertag, df, output_folder)
        
        # Create figures
        # First check if we have the necessary data for visualization
        if 'x' not in df.columns or 'y' not in df.columns:
            # Try to load the original cluster data which should have coordinates
            orig_df = load_original_cluster_data(cluster_data_path)
            if orig_df is not None and 'x' in orig_df.columns and 'y' in orig_df.columns:
                # Use original data for coordinates but add SuperTag information
                vis_df = orig_df.copy()
                if 'text' in vis_df.columns and 'tag' not in vis_df.columns:
                    vis_df = vis_df.rename(columns={'text': 'tag'})
                
                # Add supertag column to visualization dataframe
                vis_df['supertag'] = vis_df['cluster'].map(cluster_to_supertag)
                
                # Create the visualizations
                fig_paths = create_visualizations(vis_df, figures_folder, timestamp)
                if fig_paths:
                    print(f"Generated {len(fig_paths)} visualization figures")
            else:
                print("Warning: Could not create visualizations because coordinate data is missing")
        else:
            # Use the processed dataframe with SuperTags for visualization
            fig_paths = create_visualizations(df, figures_folder, timestamp)
            if fig_paths:
                print(f"Generated {len(fig_paths)} visualization figures")
        
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
        sys.exit(1)


if __name__ == "__main__":
    main()
