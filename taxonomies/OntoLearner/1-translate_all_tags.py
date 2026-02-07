#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlindWiki Ontology Builder

This script processes tag data and creates translations.
It's structured in executable regions like a notebook.
"""

# %% [REGION 1] Imports and setup
import pandas as pd
import os
from deep_translator import GoogleTranslator
import time

# %% [REGION 2] Load CSV data
def load_tags_data(csv_path):
    """Load tags data from CSV file"""
    print(f"Loading data from {csv_path}")
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path, sep=',', encoding='utf-8')
    
    # Clean up column names (remove whitespace)
    df.columns = [col.strip() for col in df.columns]
    
    print(f"Loaded {len(df)} tags")
    return df

# %% [REGION 3] Translation functions
def translate_text(text, source='auto', target='en'):
    """Translate text to target language using Google Translate"""
    if pd.isna(text) or text == "":
        return ""
    
    try:
        translator = GoogleTranslator(source=source, target=target)
        translation = translator.translate(text)
        return translation
    except Exception as e:
        print(f"Translation error for '{text}': {e}")
        return text

def add_english_translations(df, batch_size=100, delay=1):
    """Add English translations for tag names"""
    print("Adding English translations...")
    
    # Create new column for English translations
    df['name_eng'] = ""
    
    # Get unique non-empty tag names
    unique_tags = df['name'].dropna().unique()
    unique_tags = [tag for tag in unique_tags if tag.strip() != ""]
    
    print(f"Found {len(unique_tags)} unique tag names to translate")
    
    # Process in batches to avoid rate limiting
    translations = {}
    
    for i in range(0, len(unique_tags), batch_size):
        batch = unique_tags[i:i+batch_size]
        print(f"Translating batch {i//batch_size + 1}/{(len(unique_tags) // batch_size) + 1}")
        
        for tag in batch:
            translations[tag] = translate_text(tag)
        
        # Sleep to avoid rate limiting
        if i + batch_size < len(unique_tags):
            print(f"Sleeping for {delay} seconds...")
            time.sleep(delay)
    
    # Apply translations to dataframe
    df['name_eng'] = df['name'].map(lambda x: translations.get(x, "") if pd.notna(x) and x.strip() != "" else "")
    
    print("Translation complete!")
    return df

# %% [REGION 4] Main execution
def main():
    # Define paths to CSV files in mappings directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "../mappings/all_tags_counts.csv")
    
    # Load data
    tags_df = load_tags_data(csv_path)
    
    # Display first few rows
    print("\nFirst 5 rows of the data:")
    print(tags_df.head())
    
    # Add English translations
    tags_df = add_english_translations(tags_df)
    
    # Display first few rows with translations
    print("\nFirst 5 rows with translations:")
    print(tags_df[['id', 'name', 'name_eng', 'count']].head())
    
    # Save to CSV in mappings directory
    output_path = os.path.join(current_dir, "../mappings/all_tags_counts_translated.csv")
    tags_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\nSaved translated data to {output_path}")

# %% [REGION 5] Execute if run directly
if __name__ == "__main__":
    main()


