#!/usr/bin/env python3
"""
Script to process tags from JSON file using syntactic tagging strategy.
Reads tags_to_break.json, processes each tag's name field, and outputs filtered results.
"""

import sys
import os
import logging
import json
from pathlib import Path

# Add the Tagger directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tagging_strategy import create_tagging_strategy

def main():
    # Set up logging
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    # File paths
    input_file = Path(__file__).parent / "tags_to_break.json"
    output_file = Path(__file__).parent / "tags_processed.json"

    print(f"Reading input file: {input_file}")
    print(f"Writing output to: {output_file}")
    print("-" * 60)

    try:
        # Read input JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the data array
        tags_data = data[2]['data']  # Index 2 should be the table data

        print(f"Found {len(tags_data)} tags to process")

        # Create syntactic tagging strategy
        strategy = create_tagging_strategy(
            strategy_type='syntactic',
            language='en',  # English language
            max_ngram_size=1,  # Single words
            deduplication_threshold=0.9,
            num_keywords=20  # Get more keywords to filter later
        )

        results = []

        # Process each tag
        for i, tag_item in enumerate(tags_data):
            tag_id = int(tag_item['id'])
            original_text = tag_item['name']

            if i % 50 == 0:  # Progress indicator
                print(f"Processing tag {i+1}/{len(tags_data)} (ID: {tag_id})")

            try:
                # Check if text contains separators (indicating manually separated tags)
                separators = ['\n', ';', '#']
                separator_found = any(sep in original_text for sep in separators)

                if separator_found:
                    # Split by all separators and clean up each part
                    parts = original_text
                    for sep in separators:
                        parts = parts.replace(sep, '\n')  # Normalize all separators to newlines

                    lines = parts.split('\n')
                    filtered_tags = []
                    for line in lines:
                        line = line.strip()  # Remove leading/trailing whitespace
                        # Remove any remaining separator characters
                        line = line.replace(';', '').replace('#', '').strip()
                        if line:  # Only add non-empty lines
                            filtered_tags.append(line.lower())
                else:
                    # Use syntactic tagger for single-line text
                    tag_results = strategy.tag_text(original_text, top_k=20)

                    # Filter tags with similarity > 0.4
                    filtered_tags = [
                        result['tag'] for result in tag_results
                        if result['similarity'] > 0.1
                    ]

                    # Contingency: If no tags meet the threshold, take the one with highest similarity
                    if not filtered_tags and tag_results:
                        # Find the tag with highest similarity
                        best_tag = max(tag_results, key=lambda x: x['similarity'])
                        filtered_tags = [best_tag['tag']]

                # Create result entry
                result_entry = {
                    "tag_id": tag_id,
                    "original_text": original_text,
                    "new_tags": filtered_tags
                }

                results.append(result_entry)

            except Exception as e:
                print(f"Error processing tag ID {tag_id}: {e}")
                # Add entry with empty tags on error
                result_entry = {
                    "id": tag_id,
                    "original_text": original_text,
                    "new_tags": []
                }
                results.append(result_entry)

        # Write output JSON
        print(f"\nWriting {len(results)} processed results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("Processing complete!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())