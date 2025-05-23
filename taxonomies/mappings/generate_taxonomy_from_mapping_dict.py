#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Taxonomy File and Ancestors File from Mapping Dictionary

This script reads either the supertag mapping dictionary or the 16tags mapping dictionary
and generates two files:
1. A taxonomy file containing all unique tags, one per line
2. An ancestors file mapping each tag to its original tags, with newlines in tags replaced by "\n"

The output files will be automatically saved in the taxonomies folder with appropriate names:
- For supertag_mapping_dict.json -> supertags.txt and ancestors_supertags.txt
- For 16tags_mapping_dict.json -> 16tags.txt and ancestors_16tags.txt

Usage:
    python generate_taxonomy_from_mapping_dict.py --input supertag_mapping_dict.json
    python generate_taxonomy_from_mapping_dict.py --input 16tags_mapping_dict.json
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def get_output_paths(input_file):
    """Determine the output paths based on the input mapping file.
    
    Args:
        input_file (str): Path to the input mapping file
        
    Returns:
        tuple: (taxonomy_path, ancestors_path) Path objects for both output files
    """
    # Get the taxonomies directory (parent of mappings directory)
    taxonomies_dir = Path(input_file).parent.parent
    
    # Determine base name for output files
    if "supertag_mapping_dict.json" in input_file:
        base_name = "supertags"
    elif "16tags_mapping_dict.json" in input_file:
        base_name = "16tags"
    else:
        # If input filename doesn't match known patterns, use a generic name
        base_name = Path(input_file).stem.replace("_mapping_dict", "")
    
    taxonomy_path = taxonomies_dir / f"{base_name}.txt"
    ancestors_path = taxonomies_dir / f"ancestors_{base_name}.txt"
    
    return taxonomy_path, ancestors_path

def format_tag(tag):
    """Format a tag to match the taxonomy style.
    
    Args:
        tag (str): Original tag
        
    Returns:
        str: Formatted tag
    """
    # Replace spaces with underscores
    tag = tag.replace(' ', '_')
    # Capitalize first letter
    tag = tag[0].upper() + tag[1:]
    # Remove any special characters except underscores
    tag = ''.join(c for c in tag if c.isalnum() or c == '_')
    return tag

def clean_ancestor_tag(tag):
    """Clean an ancestor tag by replacing newlines with "\n".
    
    Args:
        tag (str): Original ancestor tag
        
    Returns:
        str: Cleaned tag with newlines replaced by "\n"
    """
    # Replace actual newlines with the literal string "\n"
    return tag.replace('\n', '\\n')

def generate_files(mapping_file):
    """Generate taxonomy and ancestors files from a mapping dictionary.
    
    Args:
        mapping_file (str): Path to the mapping JSON file (either supertag or 16tags mapping)
    """
    print(f"Reading mapping from {mapping_file}")
    
    # Read the mapping file
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    # Create reverse mapping (tag -> list of original tags)
    tag_to_originals = defaultdict(list)
    for original_tag, mapped_tag in mapping.items():
        formatted_tag = format_tag(mapped_tag)
        # Clean the original tag before adding it to the list
        cleaned_original = clean_ancestor_tag(original_tag)
        tag_to_originals[formatted_tag].append(cleaned_original)
    
    # Get sorted list of formatted tags
    formatted_tags = sorted(tag_to_originals.keys())
    
    # Determine output paths
    taxonomy_path, ancestors_path = get_output_paths(mapping_file)
    
    print(f"Found {len(formatted_tags)} unique tags")
    
    # Write taxonomy file
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        for tag in formatted_tags:
            f.write(f"{tag}\n")
    
    # Write ancestors file
    with open(ancestors_path, 'w', encoding='utf-8') as f:
        for tag in formatted_tags:
            original_tags = tag_to_originals[tag]
            f.write(f"{tag}:{','.join(original_tags)}\n")
    
    print(f"Taxonomy file generated at {taxonomy_path}")
    print(f"Ancestors file generated at {ancestors_path}")
    
    print("\nTaxonomy tags:")
    for tag in formatted_tags:
        print(f"- {tag}")
    
    print("\nSample of ancestors mapping:")
    for tag in formatted_tags[:3]:  # Show first 3 mappings as example
        print(f"{tag}: {','.join(tag_to_originals[tag][:3])}...")

def main():
    parser = argparse.ArgumentParser(description="Generate taxonomy and ancestors files from mapping dictionary")
    parser.add_argument("--input", type=str, required=True,
                      help="Path to the mapping JSON file (supertag_mapping_dict.json or 16tags_mapping_dict.json)")
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Mapping file not found: {args.input}")
    
    generate_files(args.input)

if __name__ == "__main__":
    main()
