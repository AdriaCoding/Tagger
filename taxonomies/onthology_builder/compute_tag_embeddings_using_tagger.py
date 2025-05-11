#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki Tag Embeddings Generator using TextEmbeddingTagger

This script uses the existing TextEmbeddingTagger class to compute
embeddings for translated tags and saves them in NPZ format.
"""

import os
import sys
import numpy as np
import pandas as pd
import importlib.util
from sentence_transformers import SentenceTransformer

# We'll use the SentenceTransformer directly rather than through TextEmbeddingTagger
# to avoid the relative import issues

# %% [REGION 1] Load and process translated tags
def load_translated_tags(csv_path):
    """Load translated tags from CSV file"""
    print(f"Loading translated tags from {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, encoding='utf-8')
    
    # Verify 'name_eng' column exists
    if 'name_eng' not in df.columns:
        raise ValueError("CSV file doesn't contain the 'name_eng' column")
    
    # Filter out empty translations
    df = df[df['name_eng'].notna() & (df['name_eng'].str.strip() != "")]
    
    print(f"Loaded {len(df)} valid translated tags")
    return df

# %% [REGION 2] Compute embeddings using SentenceTransformer directly
def compute_tag_embeddings(df, model_name="paraphrase-multilingual-mpnet-base-v2"):
    """Compute embeddings for translated tag names using SentenceTransformer"""
    print(f"Loading embedding model: {model_name}")
    
    # Load the model directly
    model = SentenceTransformer(model_name)
    
    # Extract tag texts and IDs
    texts = df['name_eng'].tolist()
    tag_ids = df['id'].tolist() if 'id' in df.columns else None
    
    print(f"Computing embeddings for {len(texts)} tags...")
    
    # Compute embeddings in batch - more efficient
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"Computed embeddings with shape: {embeddings.shape}")
    
    return embeddings, texts, tag_ids

# %% [REGION 3] Save embeddings
def save_embeddings(embeddings, texts, output_path, tag_ids=None, model_name="paraphrase-multilingual-mpnet-base-v2"):
    """Save embeddings to NPZ file"""
    print(f"Saving embeddings to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save embeddings and metadata
    np.savez(
        output_path,
        embeddings=embeddings,
        texts=texts,
        tag_ids=tag_ids if tag_ids is not None else [],
        model_name=model_name
    )
    
    print(f"Embeddings saved to {output_path}")
    return output_path

# %% [REGION 4] Main execution
def main():
    # Define paths relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Update path to point to the mappings folder where the CSV file is located
    csv_path = os.path.join(current_dir, "../mappings/all_tags_counts_translated.csv")
    output_path = os.path.join(current_dir, "../../embeddings/all_tags_translated_text_paraphrase-multilingual-mpnet-base-v2_embeddings.npz")
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    
    # Load translated tags
    tags_df = load_translated_tags(csv_path)
    
    # Display first few rows
    print("\nFirst 5 rows of the data:")
    print(tags_df[['id', 'name', 'name_eng', 'count']].head())
    
    # Compute embeddings using SentenceTransformer directly
    embeddings, texts, tag_ids = compute_tag_embeddings(tags_df, model_name)
    
    # Save embeddings
    save_path = save_embeddings(embeddings, texts, output_path, tag_ids, model_name)
    
    print(f"\nProcess completed successfully!")
    print(f"Embeddings saved to: {save_path}")

# %% [REGION 5] Execute if run directly
if __name__ == "__main__":
    main() 