#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Hybrid Taxonomy Discovery for BlindWiki Tags

This script uses a Retrieval-Augmented Generation (RAG) approach:
1. Retriever finds the top-K most similar tags per tag (candidate pairs)
2. LLM judges only those candidate pairs (reduces pairs by ~97%)

This is much faster than full LLM (script 5) while still using LLM reasoning.

Expected runtime:
  - ~596 tags × top_k=10 = ~5,960 candidate pairs
  - CPU: ~20 minutes - 1 hour
  - GPU: ~5-15 minutes

Trade-off: Faster but might miss some distant hierarchical relationships.
"""

import json
import pandas as pd
from pathlib import Path
from ontolearner import (
    AutoRAGLearner,
    AutoRetrieverLearner,
    AutoLLMLearner,
    StandardizedPrompting,
    LabelMapper,
)
import time

# %% [REGION 1] Configuration

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data/ontolearner_input"
OUTPUT_DIR = SCRIPT_DIR / "data/ontolearner_output"

INPUT_FILE = INPUT_DIR / "tags_simple.json"

# Retriever parameters
RETRIEVER_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10  # Number of candidate pairs per tag
RETRIEVER_BATCH_SIZE = -1  # -1 for full batch

# LLM parameters
LLM_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_BATCH_SIZE = 8
MAX_NEW_TOKENS = 5
DEVICE = "cpu"  # Use 'cuda' if GPU available


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


# %% [REGION 3] Run RAG taxonomy discovery

def run_rag_taxonomy_discovery(tags, retriever_model_id=RETRIEVER_MODEL_ID,
                                llm_model_id=LLM_MODEL_ID,
                                top_k=TOP_K, device=DEVICE):
    """Execute OntoLearner RAG-based taxonomy discovery"""
    n_tags = len(tags)
    estimated_pairs = n_tags * top_k

    print("\n" + "=" * 60)
    print("RAG HYBRID TAXONOMY DISCOVERY")
    print("=" * 60)
    print(f"Retriever model: {retriever_model_id}")
    print(f"LLM model: {llm_model_id}")
    print(f"Top-K candidates: {top_k}")
    print(f"Device: {device}")
    print(f"Number of tags: {n_tags}")
    print(f"Estimated candidate pairs: ~{estimated_pairs:,}")
    print(f"  (vs {n_tags * (n_tags - 1) // 2:,} for full LLM — "
          f"{100 * estimated_pairs / (n_tags * (n_tags - 1) // 2):.1f}%)")

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = AutoRetrieverLearner(top_k=top_k, batch_size=RETRIEVER_BATCH_SIZE)

    # Initialize LLM
    print("Initializing LLM learner...")
    llm = AutoLLMLearner(
        prompting=StandardizedPrompting,
        label_mapper=LabelMapper(),
        device=device,
        batch_size=LLM_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS
    )

    # Combine in RAG
    rag_learner = AutoRAGLearner(retriever=retriever, llm=llm)

    # Load models
    print(f"\nLoading models...")
    start_load = time.time()
    rag_learner.load(
        retriever_id=retriever_model_id,
        llm_id=llm_model_id
    )
    load_time = time.time() - start_load
    print(f"Models loaded in {load_time:.2f} seconds")

    # Run RAG taxonomy discovery
    print("\nPhase 1: Retriever finding candidate pairs...")
    print("Phase 2: LLM judging candidate pairs...")
    start_discovery = time.time()

    # fit() is a no-op for taxonomy discovery (emits warning)
    rag_learner.fit(train_data=tags, task='taxonomy-discovery', ontologizer=False)
    taxonomies = rag_learner.predict(eval_data=tags, task='taxonomy-discovery', ontologizer=False)

    discovery_time = time.time() - start_discovery
    print(f"\n✓ Discovery complete in {discovery_time:.2f} seconds "
          f"({discovery_time / 60:.1f} minutes)")
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


def print_sample_taxonomies(taxonomies, n=30):
    """Print sample taxonomic relationships"""
    print("\n" + "=" * 60)
    print(f"SAMPLE TAXONOMIES (showing first {n})")
    print("=" * 60)

    for i, rel in enumerate(taxonomies[:n], 1):
        parent = rel['parent']
        child = rel['child']
        print(f"{i:2d}. {parent:30s} → {child}")


def generate_summary_stats(taxonomies):
    """Generate summary statistics about discovered taxonomies"""
    df = pd.DataFrame(taxonomies)

    parent_counts = df['parent'].value_counts()
    child_counts = df['child'].value_counts()

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
        'root_nodes': sorted(list(only_parents)),
        'leaf_nodes_count': len(only_children),
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
    print("\n" + "=" * 60)
    print("TAXONOMY STATISTICS")
    print("=" * 60)
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Unique parent tags: {stats['unique_parents']}")
    print(f"Unique child tags: {stats['unique_children']}")
    print(f"Total unique tags involved: {stats['unique_tags_involved']}")
    print(f"\nTag role distribution:")
    print(f"  - Both parent and child: {stats['tags_as_both_parent_and_child']}")
    print(f"  - Only parent (root nodes): {stats['tags_only_as_parent']}")
    print(f"  - Only child (leaf nodes): {stats['tags_only_as_child']}")
    print(f"\nAverages:")
    print(f"  - Children per parent: {stats['avg_children_per_parent']:.2f}")
    print(f"  - Parents per child: {stats['avg_parents_per_child']:.2f}")

    if stats['root_nodes']:
        print(f"\nRoot nodes (potential supertags):")
        for i, root in enumerate(stats['root_nodes'][:20], 1):
            print(f"  {i:2d}. {root}")

    print(f"\nTop 10 parent tags (most children):")
    for i, item in enumerate(stats['top_10_parents'], 1):
        print(f"  {i:2d}. {item['tag']:30s} ({item['child_count']} children)")


# %% [REGION 5] Main execution

def main():
    print("=" * 60)
    print("ONTOLEARNER RAG HYBRID TAXONOMY DISCOVERY")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load preprocessed tags
    tags = load_tags(INPUT_FILE)

    # Run RAG taxonomy discovery
    taxonomies = run_rag_taxonomy_discovery(tags)

    if not taxonomies:
        print("\n⚠ No taxonomic relationships discovered!")
        return

    # Print samples
    print_sample_taxonomies(taxonomies, n=30)

    # Generate statistics
    stats = generate_summary_stats(taxonomies)
    print_summary_stats(stats)

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    json_path = OUTPUT_DIR / "rag_taxonomies.json"
    csv_path = OUTPUT_DIR / "rag_taxonomies.csv"
    stats_path = OUTPUT_DIR / "rag_taxonomy_stats.json"

    export_taxonomies_json(taxonomies, json_path)
    export_taxonomies_csv(taxonomies, csv_path)

    # Export stats
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Exported statistics to: {stats_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Discovered {len(taxonomies)} taxonomic relationships")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - rag_taxonomies.json")
    print(f"  - rag_taxonomies.csv")
    print(f"  - rag_taxonomy_stats.json")
    print("\n✓ RAG hybrid taxonomy discovery complete!")
    print("\nNext step: Run 4-analyze_taxonomy_results.py for detailed analysis")
    print("  (update TAXONOMIES_FILE to point to rag_taxonomies.json)")


if __name__ == "__main__":
    main()

