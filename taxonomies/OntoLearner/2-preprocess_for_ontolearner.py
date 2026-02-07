#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki Tags Preprocessing for OntoLearner

This script loads the translated tags CSV, cleans and normalizes the data,
and exports it in formats suitable for OntoLearner taxonomy discovery.
"""

import pandas as pd
import json
import os
from pathlib import Path

# %% [REGION 1] Configuration

# Paths
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "../mappings/all_tags_counts_translated.csv"
OUTPUT_DIR = SCRIPT_DIR / "data/ontolearner_input"

# Parameters
MIN_COUNT = 2 # Minimum occurrence count for tags
NORMALIZE_TAGS = True  # Convert to lowercase and replace spaces


# %% [REGION 2] Load and validate data

def load_tags_csv(csv_path):
    """Load tags from CSV file"""
    print(f"Loading tags from: {csv_path}")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} total rows")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df


# %% [REGION 3] Data cleaning and filtering

def clean_and_filter_tags(df, min_count=2):
    """Clean and filter tags based on criteria"""
    print("\n" + "="*60)
    print("CLEANING AND FILTERING")
    print("="*60)
    
    original_count = len(df)
    
    # Remove rows with empty or missing English names
    df = df[df['name_eng'].notna()].copy()
    df = df[df['name_eng'].str.strip() != '']
    print(f"After removing empty translations: {len(df)} tags ({original_count - len(df)} removed)")
    
    # Apply minimum count threshold
    count_before = len(df)
    df = df[df['count'] >= min_count].copy()
    print(f"After applying min_count={min_count}: {len(df)} tags ({count_before - len(df)} removed)")
    
    # Handle missing original names (use English as fallback)
    missing_original = df['name'].isna() | (df['name'].str.strip() == '')
    if missing_original.any():
        df.loc[missing_original, 'name'] = df.loc[missing_original, 'name_eng']
        print(f"Fixed {missing_original.sum()} missing original names using English translation")
    
    return df


def normalize_tag_names(df):
    """Normalize English tag names for OntoLearner"""
    print("\n" + "="*60)
    print("NORMALIZING TAG NAMES")
    print("="*60)
    
    df = df.copy()
    
    # Create normalized version
    df['name_eng_normalized'] = df['name_eng'].str.lower().str.strip()
    
    # Replace multiple spaces and special patterns
    df['name_eng_normalized'] = df['name_eng_normalized'].str.replace(r'\s+', '_', regex=True)
    
    # Remove leading/trailing underscores
    df['name_eng_normalized'] = df['name_eng_normalized'].str.strip('_')
    
    # Show some examples
    print("\nNormalization examples:")
    sample = df[['name', 'name_eng', 'name_eng_normalized']].head(10)
    print(sample.to_string(index=False))
    
    # Check for duplicates after normalization
    duplicates = df['name_eng_normalized'].duplicated()
    if duplicates.any():
        print(f"\nWarning: {duplicates.sum()} duplicate normalized names found")
        print("Duplicate examples:")
        dup_examples = df[duplicates][['name_eng', 'name_eng_normalized', 'count']].head(5)
        print(dup_examples.to_string(index=False))
        
        # Keep the one with higher count
        df = df.sort_values('count', ascending=False)
        df = df.drop_duplicates(subset=['name_eng_normalized'], keep='first')
        print(f"After removing duplicates: {len(df)} tags")
    
    return df


# %% [REGION 4] Export functions

def export_tags_simple(df, output_path):
    """Export simple list of normalized tags"""
    tags_list = sorted(df['name_eng_normalized'].tolist())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tags_list, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(tags_list)} tags to {output_path}")
    return tags_list


def export_tags_with_counts(df, output_path):
    """Export tags with metadata (counts, original names)"""
    tags_data = []
    
    for _, row in df.iterrows():
        tags_data.append({
            'tag_normalized': row['name_eng_normalized'],
            'tag_english': row['name_eng'],
            'tag_original': row['name'],
            'count': int(row['count']),
            'tag_id': int(row['id']) if pd.notna(row['id']) else None
        })
    
    # Sort by count (descending)
    tags_data.sort(key=lambda x: x['count'], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tags_data, f, indent=2, ensure_ascii=False)
    
    print(f"Exported {len(tags_data)} tags with metadata to {output_path}")
    return tags_data


def export_preprocessing_stats(df, original_count, output_path):
    """Export statistics about preprocessing"""
    stats = {
        'original_tag_count': original_count,
        'filtered_tag_count': len(df),
        'removed_count': original_count - len(df),
        'min_count_threshold': MIN_COUNT,
        'count_distribution': {
            'min': int(df['count'].min()),
            'max': int(df['count'].max()),
            'mean': float(df['count'].mean()),
            'median': float(df['count'].median())
        },
        'top_10_tags': [
            {
                'tag': row['name_eng_normalized'],
                'count': int(row['count'])
            }
            for _, row in df.nlargest(10, 'count').iterrows()
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Exported preprocessing statistics to {output_path}")
    return stats


# %% [REGION 5] Main execution

def main():
    print("="*60)
    print("BLINDWIKI TAGS PREPROCESSING FOR ONTOLEARNER")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    df = load_tags_csv(CSV_PATH)
    original_count = len(df)
    
    # Clean and filter
    df = clean_and_filter_tags(df, min_count=MIN_COUNT)
    
    # Normalize
    df = normalize_tag_names(df)
    
    # Export files
    print("\n" + "="*60)
    print("EXPORTING DATA")
    print("="*60)
    
    tags_simple_path = OUTPUT_DIR / "tags_simple.json"
    tags_counts_path = OUTPUT_DIR / "tags_with_counts.json"
    stats_path = OUTPUT_DIR / "preprocessing_stats.json"
    
    tags_list = export_tags_simple(df, tags_simple_path)
    tags_data = export_tags_with_counts(df, tags_counts_path)
    stats = export_preprocessing_stats(df, original_count, stats_path)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input CSV: {CSV_PATH}")
    print(f"Original tags: {original_count}")
    print(f"Filtered tags (min_count>={MIN_COUNT}): {len(df)}")
    print(f"Removal rate: {(1 - len(df)/original_count)*100:.1f}%")
    print(f"\nTop 5 most frequent tags:")
    for i, item in enumerate(stats['top_10_tags'][:5], 1):
        print(f"  {i}. {item['tag']} (count: {item['count']})")
    
    print(f"\nFiles created in {OUTPUT_DIR}:")
    print(f"  - tags_simple.json ({len(tags_list)} tags)")
    print(f"  - tags_with_counts.json ({len(tags_data)} tags)")
    print(f"  - preprocessing_stats.json")
    
    print("\n✓ Preprocessing complete!")


if __name__ == "__main__":
    main()

