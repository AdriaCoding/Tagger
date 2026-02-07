#!/usr/bin/env python3
"""
Script to regenerate all_tags_counts.csv from the JSON datasets.

This script reads tag_x_message.json and counts the occurrences of each tag,
then outputs the results sorted by count (descending) to all_tags_counts.csv.

Usage:
    python3 regenerate_tag_counts.py

Input:
    - AI/Tagger/taxonomies/datasets/tag_x_message.json

Output:
    - AI/Tagger/taxonomies/mappings/all_tags_counts.csv
    
The output CSV has the format:
    id, name, count
    "0000","","0"
    "5170","Umgebungsgeräusche","394"
    ...
    
Tags are sorted by count (descending), with the highest count first.
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

def parse_tag_x_message_json(json_file_path):
    """
    Parse the tag_x_message.json file and extract tag information.
    
    Returns:
        dict: A dictionary mapping tag_id to (tag_name, count)
    """
    tag_counts = defaultdict(int)
    tag_names = {}
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Read the file line by line to handle the custom JSON format
        for line in f:
            line = line.strip()
            if not line or line in ['[', ']']:
                continue
            
            # Remove trailing comma if present
            if line.endswith(','):
                line = line[:-1]
            
            try:
                data = json.loads(line)
                
                # Check if this is a data row (not header/metadata)
                if isinstance(data, dict) and 'tag_id' in data and 'name' in data:
                    tag_id = data['tag_id']
                    tag_name = data['name']
                    
                    # Store tag name (they should all be the same for a given tag_id)
                    if tag_id not in tag_names:
                        tag_names[tag_id] = tag_name
                    
                    # Increment count
                    tag_counts[tag_id] += 1
                    
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
    
    return tag_counts, tag_names

def write_tag_counts_csv(tag_counts, tag_names, output_file_path):
    """
    Write tag counts to CSV file sorted by count (descending).
    
    Args:
        tag_counts: Dictionary mapping tag_id to count
        tag_names: Dictionary mapping tag_id to tag name
        output_file_path: Path to output CSV file
    """
    # Combine the data and sort by count (descending)
    tag_data = []
    
    # Add all tags
    for tag_id, count in tag_counts.items():
        tag_name = tag_names.get(tag_id, '')
        tag_data.append((tag_id, tag_name, count))
    
    # Sort by count (descending), then by tag_id for consistency
    tag_data.sort(key=lambda x: (-x[2], x[0]))
    
    # Write to CSV
    with open(output_file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow(['id', ' name', ' count'])
        
        # Write the 0000 entry first
        writer.writerow(['0000', '', '0'])
        
        # Write data rows
        for tag_id, tag_name, count in tag_data:
            writer.writerow([tag_id, tag_name, count])
    
    print(f"Successfully wrote {len(tag_data)} tags to {output_file_path}")
    print(f"Total tag occurrences: {sum(count for _, _, count in tag_data)}")

def main():
    # Set up paths
    script_dir = Path(__file__).parent
    datasets_dir = script_dir.parent / 'datasets'
    
    json_file = datasets_dir / 'tag_x_message.json'
    output_file = script_dir / 'all_tags_counts.csv'
    
    print(f"Reading tag data from: {json_file}")
    
    # Parse the JSON file
    tag_counts, tag_names = parse_tag_x_message_json(json_file)
    
    print(f"Found {len(tag_counts)} unique tags")
    
    # Write the CSV file
    write_tag_counts_csv(tag_counts, tag_names, output_file)
    
    # Print some statistics
    if tag_counts:
        max_count = max(tag_counts.values())
        most_used_tag_id = max(tag_counts.items(), key=lambda x: x[1])[0]
        most_used_tag_name = tag_names[most_used_tag_id]
        print(f"\nMost used tag: '{most_used_tag_name}' (ID: {most_used_tag_id}) with {max_count} occurrences")

if __name__ == '__main__':
    main()

