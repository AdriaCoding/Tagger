#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki Cluster SuperTag Generator

This script analyzes clustering results, derives appropriate SuperTag names
for each cluster, and generates a tag mapping file.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import scipy.sparse as sp
from datetime import datetime

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The parameter.*")

# Add parent directory to path to access utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Global log file for detailed derivation logs
log_file = None

# %% [REGION 1] Load and prepare data
def load_cluster_data(csv_path):
    """Load cluster data from CSV file"""
    print(f"Loading cluster data from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Verify required columns exist
    required_cols = ['cluster', 'text', 'count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Group by cluster
    clusters = df.groupby('cluster')
    
    print(f"Loaded {len(df)} tags across {len(clusters)} clusters")
    return df, clusters

def preprocess_cluster_data(df, min_count=5):
    """Set as clusters of one element all the elements of the noise cluster with counts > min_count.
       Then remove the noise cluster from the dataframe.
    """
    print("Preprocessing cluster data...")
    
    # Make a copy to avoid modifying the original during iteration
    df_processed = df.copy()
    
    # Identify the noise cluster (cluster = -1)
    noise_cluster = df_processed[df_processed['cluster'] == -1]
    
    # Find significant tags in noise cluster (count > min_count)
    significant_noise = noise_cluster[noise_cluster['count'] > min_count]
    
    if len(significant_noise) > 0:
        print(f"Found {len(significant_noise)} significant tags in noise cluster with count > {min_count}")
        
        # Determine the next available cluster IDs
        max_cluster_id = df_processed['cluster'].max()
        next_cluster_id = max_cluster_id + 1
        
        # Assign new cluster IDs to significant noise elements
        for idx, row in significant_noise.iterrows():
            # Update the cluster ID in the processed dataframe
            df_processed.loc[idx, 'cluster'] = next_cluster_id
            print(f"  • Moved tag '{row['text']}' (count: {row['count']}) to new cluster {next_cluster_id}")
            next_cluster_id += 1
    
    # Remove remaining noise points
    remaining_noise = df_processed[df_processed['cluster'] == -1]
    noise_count_sum = remaining_noise['count'].sum()
    print(f"Remaining noise points: {len(remaining_noise)} tags with total count: {noise_count_sum}")
    
    # Remove all remaining noise tags from the dataframe
    df_processed = df_processed[df_processed['cluster'] != -1]
    print(f"Removed all remaining noise points from further processing")
    
    # Return the processed dataframe
    return df_processed

# %% [REGION 2] Text processing utilities
def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading required NLTK resources...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')

def preprocess_text(text):
    """Preprocess text for analysis"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_stopwords():
    """Get stopwords for multiple languages"""
    stop_words = set()
    for lang in ['english', 'spanish', 'italian', 'german', 'french', 'portuguese']:
        try:
            stop_words.update(stopwords.words(lang))
        except:
            print(f"Warning: Could not load stopwords for language: {lang}")
    
    # Add custom stopwords
    custom_stopwords = {'de', 'la', 'el', 'en', 'con', 'del', 'al', 'lo', 'para', 'por', 'das', 'der', 'die', 'le', 'du'}
    stop_words.update(custom_stopwords)
    
    return list(stop_words)  # Convert to list for compatibility with sklearn

# %% [REGION 3] SuperTag derivation methods

def get_semantic_centroid(texts, model=None, top_similar=3, weights=None):
    """Find the texts closest to the semantic centroid of the cluster
    
    Args:
        texts: List of texts to analyze
        model: SentenceTransformer model to use
        top_similar: Number of similar texts to return
        weights: Optional list of weights (counts) for each text
    """
    if not model:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Filter empty texts (and corresponding weights if provided)
    valid_indices = [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip()]
    valid_texts = [texts[i] for i in valid_indices]
    
    if weights is not None:
        valid_weights = [weights[i] for i in valid_indices]
        # Normalize weights
        total_weight = sum(valid_weights)
        if total_weight > 0:
            valid_weights = [w/total_weight for w in valid_weights]
    else:
        valid_weights = None
    
    if not valid_texts:
        return []
    
    try:
        # Compute embeddings
        embeddings = model.encode(valid_texts)
        
        # Compute weighted centroid if weights are provided
        if valid_weights:
            # Weighted average of embeddings
            centroid = np.average(embeddings, axis=0, weights=valid_weights)
        else:
            # Simple average if no weights
            centroid = np.mean(embeddings, axis=0)
        
        # Compute similarity to centroid
        similarities = cosine_similarity([centroid], embeddings)[0]
        
        # Get top similar texts
        top_indices = similarities.argsort()[-top_similar:][::-1]
        
        return [valid_texts[i] for i in top_indices]
    except Exception as e:
        print(f"Warning: Semantic centroid calculation failed: {e}")
        return []

def initialize_log_file(output_dir):
    """Initialize the log file for derivation logs"""
    global log_file
    log_path = os.path.join(output_dir, "NamederivationLogs.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    print(f"Detailed derivation logs will be written to: {log_path}")
    return log_path

def log_to_file(message):
    """Write a message to the log file"""
    global log_file
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()  # Ensure logs are written immediately

def derive_supertag(cluster_df, model=None, cluster_id=None):
    """Derive a SuperTag name for a cluster"""
    texts = cluster_df['text'].tolist()
    counts = cluster_df['count'].tolist()
    
    # For small clusters, just use the most frequent tag
    if len(texts) < 3:
        max_idx = np.argmax(counts)
        supertag = texts[max_idx].title()
        print(f"Small cluster {cluster_id}: Using most frequent tag as SuperTag '{supertag}'")
        log_to_file(f"\nCluster {cluster_id} (small cluster):")
        log_to_file(f"Used most frequent tag as SuperTag: '{supertag}'")
        return supertag
    
    # Log cluster information
    log_to_file(f"\n{'=' * 60}")
    log_to_file(f"CLUSTER {cluster_id} DERIVATION DETAILS")
    log_to_file(f"{'=' * 60}")
    log_to_file(f"Number of tags: {len(texts)}")
    log_to_file(f"Total tag count: {sum(counts)}")
       
    # Get tags closest to semantic centroid
    if not model:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    central_tags = get_semantic_centroid(texts, model, top_similar=5, weights=counts)
    
    # Combine evidence
    candidate_terms = []
    if central_tags:
        # Extract first word of each central tag
        for tag in central_tags:
            first_word = tag.split()[0] if tag else ""
            if first_word and len(first_word) > 2 and first_word.lower() not in get_stopwords():
                candidate_terms.append(first_word)
    
    # Handle case where we have no good candidates
    if not candidate_terms:
        supertag = f"Cluster_{cluster_id}"
        print(f"No good candidates for cluster {cluster_id}: Using '{supertag}'")
        log_to_file("No good candidate terms found. Using cluster ID as SuperTag.")
        return supertag
    
    # Count occurrences to find most representative term
    best_term = candidate_terms[0]
    
    # Capitalize and clean
    supertag = best_term.title().strip()
        
    # Log derivation details to file
    log_to_file(f"\nDERIVATION METHOD RESULTS:")
    log_to_file(f"{'-' * 40}")
    log_to_file(f"Semantic centroid tags:    {', '.join(central_tags)}")
    log_to_file(f"Derived SuperTag:          {supertag}")
    log_to_file(f"\nTAGS IN CLUSTER {cluster_id}:")
    log_to_file(f"{'-' * 40}")
    log_to_file(f"{'#':3s} | {'Count':5s} | Tag")
    log_to_file(f"{'-' * 40}")
    for i, (text, count) in enumerate(zip(texts, counts)):
        log_to_file(f"{i+1:3d} | {count:5d} | {text}")
    log_to_file(f"{'-' * 40}")
    
    # Only print brief info to terminal
    print(f"Derived SuperTag '{supertag}' for cluster {cluster_id} ({len(texts)} tags)")
    
    # Add cluster ID for uniqueness
    return supertag

# %% [REGION 4] Generate mapping file
def generate_mapping(df, output_dir):
    """Generate a mapping file from each tag to its SuperTag"""
    print("Generating tag mapping...")
    
    # Initialize log file in the same directory as the output path
    initialize_log_file(output_dir)
    
    # Initialize sentence transformer model
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Process each cluster
    clusters = df.groupby('cluster')
    mapping = {}
    cluster_supertags = {}
    
    # First pass - derive SuperTags for each cluster
    for cluster_id, cluster_df in clusters:
        supertag = derive_supertag(cluster_df, model, cluster_id)
        cluster_supertags[cluster_id] = supertag
        total_count = cluster_df['count'].sum()
        print(f"Cluster {cluster_id}: SuperTag = '{supertag}' ({len(cluster_df)} tags, total count: {total_count})")
    
    # Check for duplicate SuperTags and disambiguate
    supertag_counts = Counter(cluster_supertags.values())
    for cluster_id, supertag in list(cluster_supertags.items()):
        if supertag_counts[supertag] > 1:
            # Disambiguate by adding cluster ID
            if cluster_id != -1:  # Keep "Other" as is
                cluster_supertags[cluster_id] = f"{supertag}_{cluster_id}"
                print(f"Disambiguated duplicate SuperTag '{supertag}' to '{cluster_supertags[cluster_id]}'")
    
    # Second pass - create the actual mapping
    for cluster_id, cluster_df in clusters:
        supertag = cluster_supertags[cluster_id]
        
        for _, row in cluster_df.iterrows():
            tag = row['text']
            if pd.isna(tag) or not tag.strip():
                continue  # Skip empty tags
                
            mapping[tag] = supertag
        
    print(f"Generated mapping with {len(mapping)} tags -> {len(set(mapping.values()))} SuperTags")
    
    # Create aggregated DataFrame at cluster level
    cluster_data = []
    for cluster_id, supertag in cluster_supertags.items():
        cluster_data.append({
            'cluster': cluster_id,
            'supertag': supertag,
            'supertag_reviewed': supertag  # Initialize as copy of supertag
        })
    
    cluster_df = pd.DataFrame(cluster_data)
    
    # Set path for the Excel file in the onthology_builder folder
    excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "5-cluster_supertags_review_file.xlsx")
    
    # Check if the file already exists
    if os.path.exists(excel_path):
        print(f"Existing cluster Excel file found at {excel_path}")
        try:
            # Load existing Excel file
            existing_df = pd.read_excel(excel_path)
            
            # Create a mapping of cluster_id to supertag_reviewed from existing file
            existing_reviews = {}
            for _, row in existing_df.iterrows():
                if 'cluster' in existing_df.columns and 'supertag_reviewed' in existing_df.columns:
                    existing_reviews[row['cluster']] = row['supertag_reviewed']
            
            # Update current DataFrame with existing reviewed supertags
            for i, row in cluster_df.iterrows():
                if row['cluster'] in existing_reviews:
                    cluster_df.at[i, 'supertag_reviewed'] = existing_reviews[row['cluster']]
            
            print(f"Preserved {len(existing_reviews)} existing reviewed SuperTags")
        except Exception as e:
            print(f"Warning: Could not read existing Excel file: {e}")
            print("Creating new Excel file with default values")
    
    # Save the data
    cluster_df.to_excel(excel_path, index=False)
    print(f"Aggregated cluster data saved to {excel_path}")
    
    # Close log file
    if log_file:
        log_file.close()
    
    return mapping, cluster_supertags

# %% [REGION 5] Main execution
def main():
    # Check for command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "data", "cluster_data.csv")
    
    # Set output path
    output_dir = os.path.join(current_dir, "data")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load cluster data
    df, clusters = load_cluster_data(csv_path)

    # preprocess the cluster data
    df = preprocess_cluster_data(df)
    
    # Generate mapping
    mapping, cluster_supertags = generate_mapping(df, output_dir)
    
    # Rename columns and reorganize dataframe
    df = df.rename(columns={'tag_id': 'id', 'text': 'tag'})
    
    # Add supertag column after count
    df['supertag'] = df['cluster'].map(cluster_supertags)
    
    # Reorder columns again to place supertag after count
    df = df[['id', 'tag', 'count', 'cluster', 'supertag', 'x', 'y']]
    
    # Save the enhanced dataframe to a CSV file
    csv_output_path = os.path.join(output_dir, "supertags_tag_data.csv")
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
    df.to_csv(csv_output_path, index=False)
    print(f"Enhanced tag data saved to {csv_output_path}")
    
    # Print examples for some SuperTags
    print("\nExample mappings for selected SuperTags:")
    print("-" * 50)
    example_tags = {}
    
    # Get up to 5 examples for each SuperTag
    for tag, supertag in mapping.items():
        if supertag not in example_tags:
            example_tags[supertag] = []
        if len(example_tags[supertag]) < 5:
            example_tags[supertag].append(tag)
    
    # Print examples for top 10 SuperTags by frequency
    top_supertags = [st for st, _ in Counter(mapping.values()).most_common(10)]
    for supertag in top_supertags:
        print(f"{supertag}:")
        for tag in example_tags[supertag]:
            print(f"  - {tag}")
        print()

# %% [REGION 6] Execute if run directly
if __name__ == "__main__":
    main()
