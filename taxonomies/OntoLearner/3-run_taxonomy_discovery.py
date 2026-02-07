#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OntoLearner Taxonomy Discovery for BlindWiki Tags

This script runs retriever-based taxonomy discovery using OntoLearner
to identify hierarchical relationships between tags.
"""

import json
import pandas as pd
from pathlib import Path
from ontolearner import AutoRetrieverLearner
import time

# %% [REGION 1] Configuration

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data/ontolearner_input"
OUTPUT_DIR = SCRIPT_DIR / "data/ontolearner_output"

INPUT_FILE = INPUT_DIR / "tags_simple.json"

# OntoLearner parameters
TOP_K = 5  # Number of similar tags to retrieve
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, general-purpose model
BATCH_SIZE = -1  # -1 for full batch processing


# %% [REGION 2] Load preprocessed data

def load_tags(json_path):
    """Load preprocessed tags from JSON"""
    print(f"Loading tags from: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Tags file not found: {json_path}\n"
            f"Please run 2-preprocess_for_ontolearner.py first"
        )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tags = json.load(f)
    
    print(f"Loaded {len(tags)} tags")
    
    # Show sample
    print("\nSample tags:")
    for tag in tags[:10]:
        print(f"  - {tag}")
    
    return tags


# %% [REGION 3] Run taxonomy discovery

def run_taxonomy_discovery(tags, top_k=5, model_id=MODEL_ID):
    """Execute OntoLearner retriever-based taxonomy discovery"""
    print("\n" + "="*60)
    print("TAXONOMY DISCOVERY WITH ONTOLEARNER")
    print("="*60)
    print(f"Model: {model_id}")
    print(f"Top-K: {top_k}")
    print(f"Number of tags: {len(tags)}")
    
    # Initialize retriever learner
    print("\nInitializing AutoRetrieverLearner...")
    learner = AutoRetrieverLearner(top_k=top_k, batch_size=BATCH_SIZE)
    
    # Load model
    print(f"Loading model: {model_id}")
    start_load = time.time()
    learner.load(model_id=model_id)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Run taxonomy discovery
    print("\nRunning taxonomy discovery...")
    print("(This may take a few minutes depending on the number of tags)")
    start_discovery = time.time()
    
    taxonomies = learner.fit_predict(
        train_data=tags,
        eval_data=tags,
        task='taxonomy-discovery'
    )
    
    discovery_time = time.time() - start_discovery
    print(f"\n✓ Discovery complete in {discovery_time:.2f} seconds")
    print(f"Found {len(taxonomies)} taxonomic relationships")
    
    return taxonomies


# %% [REGION 4] Export results

def export_taxonomies_json(taxonomies, output_path):
    """Export taxonomies as JSON"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(taxonomies, f, indent=2, ensure_ascii=False)
    
    print(f"Exported taxonomies to: {output_path}")


def export_taxonomies_csv(taxonomies, output_path):
    """Export taxonomies as CSV for easy viewing"""
    df = pd.DataFrame(taxonomies)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Exported taxonomies to: {output_path}")


def print_sample_taxonomies(taxonomies, n=20):
    """Print sample taxonomic relationships"""
    print("\n" + "="*60)
    print(f"SAMPLE TAXONOMIES (showing first {n})")
    print("="*60)
    
    for i, rel in enumerate(taxonomies[:n], 1):
        parent = rel['parent']
        child = rel['child']
        print(f"{i:2d}. {parent:30s} → {child}")


def generate_summary_stats(taxonomies):
    """Generate summary statistics about discovered taxonomies"""
    df = pd.DataFrame(taxonomies)
    
    # Count occurrences
    parent_counts = df['parent'].value_counts()
    child_counts = df['child'].value_counts()
    
    # Find tags that appear as both parent and child
    all_parents = set(df['parent'].unique())
    all_children = set(df['child'].unique())
    both = all_parents.intersection(all_children)
    only_parents = all_parents - all_children
    only_children = all_children - all_parents
    
    stats = {
        'total_relationships': len(taxonomies),
        'unique_parents': len(all_parents),
        'unique_children': len(all_children),
        'unique_tags_involved': len(all_parents.union(all_children)),
        'tags_as_both_parent_and_child': len(both),
        'tags_only_as_parent': len(only_parents),
        'tags_only_as_child': len(only_children),
        'avg_children_per_parent': len(taxonomies) / len(all_parents) if all_parents else 0,
        'avg_parents_per_child': len(taxonomies) / len(all_children) if all_children else 0,
        'top_10_parents': [
            {'tag': tag, 'child_count': int(count)}
            for tag, count in parent_counts.head(10).items()
        ],
        'top_10_children': [
            {'tag': tag, 'parent_count': int(count)}
            for tag, count in child_counts.head(10).items()
        ]
    }
    
    return stats


def print_summary_stats(stats):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TAXONOMY STATISTICS")
    print("="*60)
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Unique parent tags: {stats['unique_parents']}")
    print(f"Unique child tags: {stats['unique_children']}")
    print(f"Total unique tags involved: {stats['unique_tags_involved']}")
    print(f"\nTag role distribution:")
    print(f"  - Both parent and child: {stats['tags_as_both_parent_and_child']}")
    print(f"  - Only parent: {stats['tags_only_as_parent']}")
    print(f"  - Only child: {stats['tags_only_as_child']}")
    print(f"\nAverages:")
    print(f"  - Children per parent: {stats['avg_children_per_parent']:.2f}")
    print(f"  - Parents per child: {stats['avg_parents_per_child']:.2f}")
    
    print(f"\nTop 10 parent tags (most children):")
    for i, item in enumerate(stats['top_10_parents'], 1):
        print(f"  {i:2d}. {item['tag']:30s} ({item['child_count']} children)")
    
    print(f"\nTop 10 child tags (most parents):")
    for i, item in enumerate(stats['top_10_children'], 1):
        print(f"  {i:2d}. {item['tag']:30s} ({item['parent_count']} parents)")


# %% [REGION 5] Main execution

def main():
    print("="*60)
    print("ONTOLEARNER TAXONOMY DISCOVERY")
    print("="*60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load preprocessed tags
    tags = load_tags(INPUT_FILE)
    
    # Run taxonomy discovery
    taxonomies = run_taxonomy_discovery(tags, top_k=TOP_K, model_id=MODEL_ID)
    
    # Print samples
    print_sample_taxonomies(taxonomies, n=20)
    
    # Generate statistics
    stats = generate_summary_stats(taxonomies)
    print_summary_stats(stats)
    
    # Export results
    print("\n" + "="*60)
    print("EXPORTING RESULTS")
    print("="*60)
    
    json_path = OUTPUT_DIR / "discovered_taxonomies.json"
    csv_path = OUTPUT_DIR / "discovered_taxonomies.csv"
    stats_path = OUTPUT_DIR / "taxonomy_stats.json"
    
    export_taxonomies_json(taxonomies, json_path)
    export_taxonomies_csv(taxonomies, csv_path)
    
    # Export stats
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Exported statistics to: {stats_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("COMPLETED")
    print("="*60)
    print(f"Discovered {len(taxonomies)} taxonomic relationships")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - discovered_taxonomies.json")
    print(f"  - discovered_taxonomies.csv")
    print(f"  - taxonomy_stats.json")
    print("\n✓ Taxonomy discovery complete!")
    print("\nNext step: Run 4-analyze_taxonomy_results.py for detailed analysis")


if __name__ == "__main__":
    main()

