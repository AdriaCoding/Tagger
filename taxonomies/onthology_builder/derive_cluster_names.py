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

# Suppress common warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The parameter.*")

# Add parent directory to path to access utilities
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

def get_top_tfidf_terms(texts, n=5):
    """Extract top TF-IDF terms from a collection of texts"""
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Get stopwords
    stop_words = get_stopwords()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        min_df=1, 
        max_df=0.7,
        stop_words=stop_words,  # Now passing a list instead of a set
        ngram_range=(1, 2)  # Include both unigrams and bigrams
    )
    
    # Handle empty texts
    if not any(processed_texts):
        return []
    
    try:
        # Generate TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        
        # Sum TF-IDF scores across documents to find most important terms
        feature_names = vectorizer.get_feature_names_out()
        tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()
        
        # Get top terms
        top_indices = tfidf_sum.argsort()[-n:][::-1]
        top_terms = [feature_names[i] for i in top_indices if tfidf_sum[i] > 0]
        
        return top_terms
    except ValueError as e:
        print(f"Warning: TF-IDF extraction failed: {e}")
        return []

def get_most_frequent_words(texts, n=5):
    """Get most frequent words across all texts"""
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Get stopwords
    stop_words = set(get_stopwords())  # Convert back to set for faster lookups
    
    # Count words
    word_counter = Counter()
    for text in processed_texts:
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        word_counter.update(words)
    
    # Return top n words
    return [word for word, count in word_counter.most_common(n)]

def get_semantic_centroid(texts, model=None, top_similar=3):
    """Find the texts closest to the semantic centroid of the cluster"""
    if not model:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    # Filter empty texts
    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not valid_texts:
        return []
    
    try:
        # Compute embeddings
        embeddings = model.encode(valid_texts)
        
        # Compute centroid
        centroid = np.mean(embeddings, axis=0)
        
        # Compute similarity to centroid
        similarities = cosine_similarity([centroid], embeddings)[0]
        
        # Get top similar texts
        top_indices = similarities.argsort()[-top_similar:][::-1]
        
        return [valid_texts[i] for i in top_indices]
    except Exception as e:
        print(f"Warning: Semantic centroid calculation failed: {e}")
        return []

def derive_supertag(cluster_df, model=None, cluster_id=None):
    """Derive a SuperTag name for a cluster"""
    texts = cluster_df['text'].tolist()
    counts = cluster_df['count'].tolist()
    
    # For small clusters, just use the most frequent tag
    if len(texts) < 3:
        max_idx = np.argmax(counts)
        return texts[max_idx].title()
    
    # Get top TF-IDF terms
    top_tfidf = get_top_tfidf_terms(texts, n=3)
    
    # Get most frequent words
    top_freq = get_most_frequent_words(texts, n=3)
    
    # Get tags closest to semantic centroid
    if not model:
        model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    central_tags = get_semantic_centroid(texts, model, top_similar=2)
    
    # Combine evidence
    candidate_terms = []
    candidate_terms.extend(top_tfidf)
    candidate_terms.extend(top_freq)
    if central_tags:
        # Extract first word of each central tag
        for tag in central_tags:
            first_word = tag.split()[0] if tag else ""
            if first_word and len(first_word) > 2 and first_word.lower() not in get_stopwords():
                candidate_terms.append(first_word)
    
    # Handle case where we have no good candidates
    if not candidate_terms:
        return f"Cluster_{cluster_id}"
    
    # Count occurrences to find most representative term
    term_counter = Counter(candidate_terms)
    best_term = term_counter.most_common(1)[0][0]
    
    # Capitalize and clean
    supertag = best_term.title().strip()
    
    # Handle short or common terms
    if len(supertag) < 4 or supertag.lower() in get_stopwords():
        if len(term_counter) > 1:
            second_best = term_counter.most_common(2)[1][0]
            supertag = second_best.title().strip()
        # If still not good, use most frequent tag
        if len(supertag) < 4 or supertag.lower() in get_stopwords():
            supertag = texts[np.argmax(counts)].split()[0].title()
    
    # Add cluster ID for uniqueness
    return supertag

# %% [REGION 4] Generate mapping file
def generate_mapping(df, output_path):
    """Generate a mapping file from each tag to its SuperTag"""
    print("Generating tag mapping...")
    
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
    
    # Save mapping to JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    print(f"Generated mapping with {len(mapping)} tags -> {len(set(mapping.values()))} SuperTags")
    print(f"Mapping saved to {output_path}")
    
    return mapping, cluster_supertags

# %% [REGION 5] Main execution
def main():
    # Check for command line arguments
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, "figures_20250511_204225/cluster_data_20250511_204225.csv")
    
    # Set output path
    output_dir = os.path.join(current_dir, "../mappings")
    output_path = os.path.join(output_dir, "supertag_mapping.json")
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load cluster data
    df, clusters = load_cluster_data(csv_path)

    # preprocess the cluster data
    df = preprocess_cluster_data(df)
    
    # Generate mapping
    mapping, cluster_supertags = generate_mapping(df, output_path)
    
    # Rename columns and reorganize dataframe
    df = df.rename(columns={'tag_id': 'id', 'text': 'tag'})
    
    # Add supertag column after count
    df['supertag'] = df['cluster'].map(cluster_supertags)
    
    # Reorder columns again to place supertag after count
    df = df[['id', 'tag', 'count', 'cluster', 'supertag', 'x', 'y']]
    
    # Save the enhanced dataframe to a CSV file
    csv_output_path = os.path.join(output_dir, "enhanced_tag_data.csv")
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
