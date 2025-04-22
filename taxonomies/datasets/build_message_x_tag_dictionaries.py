import json
import os
import argparse

def build_message_x_tag_dictionaries(json_file, output_base=None):
    """
    Builds nested dictionaries from the JSON file.
    
    Args:
        json_file: Path to the JSON file with the data
        output_base: Prefix for output files (optional)
        
    Returns:
        Dictionary with the data structures
    """
    print(f"Reading JSON file: {json_file}")
    
    # Initialize dictionaries
    message_to_tags = {}
    tag_to_messages = {}
    message_metadata = {}
    tag_metadata = {}
    
    # Read JSON
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
        # Find the data section in the JSON
        for item in data:
            if isinstance(item, dict) and "type" in item and item["type"] == "table" and "data" in item:
                # Process each data row
                for row in item["data"]:
                    message_id = row["message_id"]
                    tag_id = row["tag_id"]
                    tag_name = row["name"]
                    
                    # Build message-tag relationships
                    if message_id not in message_to_tags:
                        message_to_tags[message_id] = set()
                    message_to_tags[message_id].add(tag_id)
                    
                    if tag_id not in tag_to_messages:
                        tag_to_messages[tag_id] = set()
                    tag_to_messages[tag_id].add(message_id)
                    
                    # Save message metadata (only once per message)
                    if message_id not in message_metadata:
                        message_metadata[message_id] = {
                            'text': row["text"],
                            'longitude': row["longitude"],
                            'latitude': row["latitude"],
                            'area_id': row["area_id"],
                            'author_id': row["author_user_id"],
                            'device': row["device_string"],
                            'address': row["address"],
                            'tags_fulltext': row["tags_fulltext"]
                        }
                    
                    # Save tag metadata (only once per tag)
                    if tag_id not in tag_metadata:
                        tag_metadata[tag_id] = {
                            'name': tag_name,
                            'visible': row["visible"]
                        }
    
    print(f"Processing completed.")
    print(f"Found {len(message_to_tags)} unique messages and {len(tag_metadata)} unique tags.")
    
    # Convert sets to lists for JSON serialization
    message_to_tags_serializable = {k: list(v) for k, v in message_to_tags.items()}
    tag_to_messages_serializable = {k: list(v) for k, v in tag_to_messages.items()}
    
    # Create complete data structure
    data_structure = {
        'message_to_tags': message_to_tags_serializable,
        'tag_to_messages': tag_to_messages_serializable,
        'message_metadata': message_metadata,
        'tag_metadata': tag_metadata
    }
    
    # Save result if output file was specified
    if output_base:
        # Save in JSON format
        json_file = f"{output_base}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data_structure, f, ensure_ascii=False, indent=2)
        print(f"Data structure saved in JSON format: {json_file}")
    
    return data_structure

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build message x tag dictionaries from JSON data')
    parser.add_argument('--input', default="tag_x_message.json", 
                        help='Input JSON file (default: tag_x_message.json)')
    parser.add_argument('--output', default="all_tags_x_message_dict", 
                        help='Output file base name (default: all_tags_x_message_dict)')
    
    args = parser.parse_args()
    
    # Build dictionaries
    build_message_x_tag_dictionaries(args.input, args.output)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 