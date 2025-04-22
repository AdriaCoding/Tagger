import json
import os
import argparse

def apply_tag_mapping(dict_file, mapping_name="16tags"):
    """
    Applies a taxonomy mapping to the data structure.
    
    Args:
        dict_file: Path to the dictionary file (.json) created by build_message_x_tag_dictionaries.py
        mapping_name: Name of the mapping to use (default: 16tags)
        output_suffix: Suffix to add to the output file name
        
    Returns:
        Dictionary with both message_to_tags and tags_to_messages mappings
    """
    print(f"Loading data structure from: {dict_file}")
    
    # Load the dictionary data structure
    with open(dict_file, 'r', encoding='utf-8') as f:
        data_structure = json.load(f)
    
    # Construct the mapping file path
    mapping_file = f"../mappings/{mapping_name}_mapping_dict.json"
    print(f"Applying taxonomy mapping from: {mapping_file}")
    
    # Check if mapping file exists
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file {mapping_file} not found.")
        return None
    
    # Load mapping
    with open(mapping_file, 'r', encoding='utf-8') as f:
        tag_mapping = json.load(f)
    
    message_to_tags = {}
    tags_to_messages = {}
    tag_metadata = data_structure['tag_metadata']
    
    # For each message, map its tags to the new taxonomy
    for msg_id, tag_ids in data_structure['message_to_tags'].items():
        new_tags = set()
        
        for tag_id in tag_ids:
            if tag_id in tag_metadata:
                tag_name = tag_metadata[tag_id]['name']
                if tag_name in tag_mapping:
                    new_tags.add(tag_mapping[tag_name])
        
        if new_tags:  # Only include if there are mapped tags
            message_to_tags[msg_id] = list(new_tags)
            
            # Build the inverse mapping (tags to messages)
            for tag in new_tags:
                if tag not in tags_to_messages:
                    tags_to_messages[tag] = []
                tags_to_messages[tag].append(msg_id)
    
    print(f"Mapping completed. {len(message_to_tags)} messages with {len(tags_to_messages)} tags in the new taxonomy.")
    
    # Create the final data structure
    result_data = {
        'message_to_tags': message_to_tags,
        'tags_to_messages': tags_to_messages
    }
    
    # Generate output filename
    output_file = f"{mapping_name}_x_message.json"
    
    # Save the mapping result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"Mapping result saved in: {output_file}")
    
    return result_data

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply tag mapping to message x tag dictionaries')
    parser.add_argument('--input', default="all_tags_x_message_dict.json", 
                        help='Input dictionary file (.json) (default: all_tags_x_message_dict.json)')
    parser.add_argument('--mapping', default="16tags", 
                        help='Mapping name to use (default: 16tags)')
    
    args = parser.parse_args()
    
    # Apply mapping
    apply_tag_mapping(args.input, args.mapping)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main() 