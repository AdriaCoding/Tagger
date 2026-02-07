#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained LLM Taxonomy Discovery for BlindWiki Tags

This script uses the existing 95 supertags from clustering analysis as fixed 
root categories and asks the LLM to validate/refine the tag assignments.

Strategy:
  - For each tag, ask the LLM: "Is supertag X a parent of tag Y?"
  - Only evaluates: ~596 tags * 95 supertags = ~56,620 pairs
  - Much faster than full LLM discovery

Expected runtime:
  - CPU: ~2-3 hours
  - GPU: ~30-60 minutes

This validates your existing clustering-based taxonomy with LLM reasoning.
"""

import json
import pandas as pd
from pathlib import Path
from ontolearner import AutoLLMLearner, StandardizedPrompting, LabelMapper
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# %% [REGION 1] Configuration

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data/ontolearner_input"
OUTPUT_DIR = SCRIPT_DIR / "data/ontolearner_output"
MAPPING_DIR = SCRIPT_DIR / "../mappings"

INPUT_FILE = INPUT_DIR / "tags_simple.json"
SUPERTAG_MAPPING_FILE = MAPPING_DIR / "supertag_mapping_dict.json"

# LLM parameters
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 512
MAX_NEW_TOKENS = 5
DEVICE = "cuda"  # Use 'cuda' if GPU available


# %% [REGION 2] Load data

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
    return tags


def load_supertag_mapping(json_path):
    """Load the supertag mapping to extract supertags and existing assignments"""
    if not json_path.exists():
        raise FileNotFoundError(f"Supertag mapping not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    print(f"Loaded supertag mapping with {len(mapping)} entries")
    
    # Extract unique supertags (these will be our fixed roots)
    unique_supertags = sorted(set(mapping.values()))
    
    # Normalize the mapping for comparison (lowercase, underscores)
    normalized_mapping = {}
    for original_tag, supertag in mapping.items():
        tag_norm = original_tag.lower().strip().replace(' ', '_')
        supertag_norm = supertag.lower().strip().replace(' ', '_')
        normalized_mapping[tag_norm] = supertag_norm
    
    print(f"Found {len(unique_supertags)} unique supertags")
    
    return unique_supertags, normalized_mapping


# %% [REGION 3] Build constrained dataset

def build_constrained_dataset(tags, supertags):
    """Build dataset of (supertag, tag) pairs for LLM evaluation.

    For each tag, we create one prompt per supertag asking:
    "Is supertag X a superclass of tag Y?"

    This uses the same prompt template as OntoLearner's StandardizedPrompting
    for taxonomy-discovery, ensuring consistency with the library.
    """
    prompting = StandardizedPrompting(task='taxonomy-discovery')

    # Normalize supertags for comparison
    supertags_normalized = [st.lower().strip().replace(' ', '_') for st in supertags]
    
    # Filter out supertags from the tag list to avoid self-relationships
    tags_to_classify = [t for t in tags if t not in supertags_normalized]

    n_pairs = len(tags_to_classify) * len(supertags)
    print(f"\nBuilding constrained dataset:")
    print(f"  Tags to classify: {len(tags_to_classify)}")
    print(f"  Supertags (fixed roots): {len(supertags)}")
    print(f"  Total pairs: {n_pairs:,}")

    dataset = []
    for tag in tags_to_classify:
        for supertag in supertags:
            # Use normalized supertag for consistency
            supertag_normalized = supertag.lower().strip().replace(' ', '_')
            dataset.append({
                "parent": supertag_normalized,
                "child": tag,
                "prompt": prompting.format(parent=supertag_normalized, child=tag)
            })

    return dataset


# %% [REGION 4] Run constrained taxonomy discovery

def run_constrained_taxonomy_discovery(tags, supertags, model_id=MODEL_ID,
                                        batch_size=BATCH_SIZE,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        device=DEVICE):
    """Execute constrained LLM taxonomy discovery against fixed supertags"""
    print("\n" + "=" * 60)
    print("CONSTRAINED LLM TAXONOMY DISCOVERY (95 SUPERTAGS)")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Number of supertags: {len(supertags)}")
    print(f"Number of tags: {len(tags)}")

    # Build the constrained dataset
    dataset = build_constrained_dataset(tags, supertags)
    n_pairs = len(dataset)

    # Estimate runtime
    pairs_per_sec_cpu = 2.5
    pairs_per_sec_gpu = 15
    rate = pairs_per_sec_gpu if device == 'cuda' else pairs_per_sec_cpu
    est_minutes = n_pairs / rate / 60
    print(f"Estimated runtime: {est_minutes:.0f} minutes ({est_minutes / 60:.1f} hours)")

    # Initialize LLM learner
    print("\nInitializing AutoLLMLearner...")
    learner = AutoLLMLearner(
        prompting=StandardizedPrompting,
        label_mapper=LabelMapper(),
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens
    )

    # Load model
    print(f"Loading model: {model_id}")
    start_load = time.time()
    learner.load(model_id=model_id)
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f} seconds")

    # Run prediction using the internal predict method directly
    print(f"\nEvaluating {n_pairs:,} (supertag, tag) pairs...")
    start_discovery = time.time()

    taxonomies = learner._taxonomy_discovery_predict(dataset=dataset)

    discovery_time = time.time() - start_discovery
    print(f"\n✓ Discovery complete in {discovery_time:.2f} seconds "
          f"({discovery_time / 60:.1f} minutes)")
    print(f"Found {len(taxonomies)} taxonomic relationships")

    return taxonomies


# %% [REGION 5] Analyze and compare results

def compare_with_existing_mapping(taxonomies, existing_mapping):
    """Compare discovered taxonomies with existing clustering-based mapping"""
    print("\n" + "=" * 60)
    print("COMPARISON WITH EXISTING SUPERTAG MAPPING")
    print("=" * 60)

    # Build discovered mapping: tag -> [supertags]
    discovered_mapping = {}
    for rel in taxonomies:
        child = rel['child']
        parent = rel['parent']
        if child not in discovered_mapping:
            discovered_mapping[child] = []
        discovered_mapping[child].append(parent)

    # Compare
    agreement_count = 0
    disagreement_examples = []
    new_assignments = []
    tags_compared = 0

    for tag, existing_parent in existing_mapping.items():
        tags_compared += 1
        if tag in discovered_mapping:
            discovered_parents = discovered_mapping[tag]
            if existing_parent in discovered_parents:
                agreement_count += 1
            else:
                disagreement_examples.append({
                    'tag': tag,
                    'existing_parent': existing_parent,
                    'discovered_parents': discovered_parents
                })
        else:
            # Tag not assigned to any supertag by LLM
            disagreement_examples.append({
                'tag': tag,
                'existing_parent': existing_parent,
                'discovered_parents': []
            })

    # Tags with LLM assignments that weren't in original mapping
    for tag, parents in discovered_mapping.items():
        if tag not in existing_mapping:
            new_assignments.append({
                'tag': tag,
                'discovered_parents': parents
            })

    agreement_rate = agreement_count / tags_compared if tags_compared > 0 else 0

    print(f"Tags in existing mapping: {len(existing_mapping)}")
    print(f"Tags with LLM-discovered parents: {len(discovered_mapping)}")
    print(f"Agreements with existing mapping: {agreement_count}")
    print(f"Agreement rate: {agreement_rate:.1%}")

    if disagreement_examples:
        print(f"\nSample disagreements (showing first 20):")
        for ex in disagreement_examples[:20]:
            discovered_str = ', '.join(ex['discovered_parents']) if ex['discovered_parents'] else 'NONE'
            print(f"  {ex['tag']:30s}  existing: {ex['existing_parent']:20s}  "
                  f"LLM: {discovered_str}")

    if new_assignments:
        print(f"\nNew tag assignments by LLM (not in original mapping, showing first 15):")
        for ex in new_assignments[:15]:
            print(f"  {ex['tag']:30s} → {', '.join(ex['discovered_parents'])}")

    comparison = {
        'tags_in_existing_mapping': len(existing_mapping),
        'tags_with_llm_parents': len(discovered_mapping),
        'agreement_count': agreement_count,
        'agreement_rate': agreement_rate,
        'disagreement_count': len(disagreement_examples),
        'new_assignments_count': len(new_assignments),
        'disagreement_examples': disagreement_examples[:50],
        'new_assignments': new_assignments[:50]
    }

    return comparison


def analyze_coverage(taxonomies, tags, supertags):
    """Analyze how well the supertags cover all tags"""
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)

    # Normalize supertags
    supertags_normalized = set([st.lower().strip().replace(' ', '_') for st in supertags])

    # Build mapping: tag -> [supertags]
    tag_to_supertags = {}
    for rel in taxonomies:
        child = rel['child']
        parent = rel['parent']
        if child not in tag_to_supertags:
            tag_to_supertags[child] = []
        tag_to_supertags[child].append(parent)

    # Tags with exactly one parent (clean assignment)
    single_parent = {t: p for t, p in tag_to_supertags.items() if len(p) == 1}
    # Tags with multiple parents (ambiguous)
    multi_parent = {t: p for t, p in tag_to_supertags.items() if len(p) > 1}
    # Tags with no parent (unassigned)
    all_tags_set = set(t for t in tags if t not in supertags_normalized)
    assigned_tags = set(tag_to_supertags.keys())
    unassigned = all_tags_set - assigned_tags

    print(f"Total tags (excluding supertags): {len(all_tags_set)}")
    print(f"Tags with exactly 1 supertag: {len(single_parent)} "
          f"({100 * len(single_parent) / len(all_tags_set):.1f}%)")
    print(f"Tags with multiple supertags: {len(multi_parent)} "
          f"({100 * len(multi_parent) / len(all_tags_set):.1f}%)")
    print(f"Tags with no supertag: {len(unassigned)} "
          f"({100 * len(unassigned) / len(all_tags_set):.1f}%)")

    # Children per supertag
    supertag_counts = {}
    for supertag in supertags:
        supertag_normalized = supertag.lower().strip().replace(' ', '_')
        children = [rel['child'] for rel in taxonomies if rel['parent'] == supertag_normalized]
        supertag_counts[supertag] = len(children)

    print(f"\nTop 20 supertags by children count:")
    sorted_supertags = sorted(supertag_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (supertag, count) in enumerate(sorted_supertags[:20], 1):
        bar = "█" * (count // 2)
        print(f"  {i:2d}. {supertag:25s} {count:4d} {bar}")

    if multi_parent:
        print(f"\nSample tags with multiple supertags (showing first 20):")
        for tag, parents in list(multi_parent.items())[:20]:
            print(f"  {tag:30s} → {', '.join(parents)}")

    if unassigned:
        print(f"\nSample unassigned tags (showing first 20):")
        for tag in sorted(unassigned)[:20]:
            print(f"  - {tag}")

    analysis = {
        'total_tags': len(all_tags_set),
        'single_parent_count': len(single_parent),
        'multi_parent_count': len(multi_parent),
        'unassigned_count': len(unassigned),
        'supertag_child_counts': supertag_counts,
        'multi_parent_tags': {t: p for t, p in list(multi_parent.items())[:100]},
        'unassigned_tags': sorted(list(unassigned))[:200]
    }

    return analysis


# %% [REGION 6] Export results

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


def export_flat_mapping(taxonomies, output_path):
    """Export a flat tag -> supertag mapping (for tags with single parent)"""
    tag_to_supertags = {}
    for rel in taxonomies:
        child = rel['child']
        parent = rel['parent']
        if child not in tag_to_supertags:
            tag_to_supertags[child] = []
        tag_to_supertags[child].append(parent)

    # For tags with multiple parents, use the first one
    flat_mapping = {}
    for tag, parents in sorted(tag_to_supertags.items()):
        flat_mapping[tag] = parents[0] if len(parents) == 1 else parents

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(flat_mapping, f, indent=2, ensure_ascii=False)
    print(f"Exported flat mapping to: {output_path}")


def print_sample_taxonomies(taxonomies, n=30):
    """Print sample taxonomic relationships"""
    print("\n" + "=" * 60)
    print(f"SAMPLE TAXONOMIES (showing first {n})")
    print("=" * 60)

    for i, rel in enumerate(taxonomies[:n], 1):
        parent = rel['parent']
        child = rel['child']
        print(f"{i:2d}. {parent:25s} → {child}")


# %% [REGION 7] Main execution

def main():
    print("=" * 60)
    print("CONSTRAINED LLM TAXONOMY DISCOVERY (95 SUPERTAGS)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    tags = load_tags(INPUT_FILE)
    supertags, existing_mapping = load_supertag_mapping(SUPERTAG_MAPPING_FILE)

    # Print sample supertags
    print(f"\nFixed supertag roots ({len(supertags)}):")
    print("First 15 supertags:")
    for st in supertags[:15]:
        print(f"  - {st}")
    print(f"  ... and {len(supertags) - 15} more")

    # Run constrained taxonomy discovery
    taxonomies = run_constrained_taxonomy_discovery(tags, supertags)

    if not taxonomies:
        print("\n⚠ No taxonomic relationships discovered!")
        return

    # Print samples
    print_sample_taxonomies(taxonomies, n=30)

    # Analyze coverage
    coverage = analyze_coverage(taxonomies, tags, supertags)

    # Compare with existing mapping
    comparison = compare_with_existing_mapping(taxonomies, existing_mapping)

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    json_path = OUTPUT_DIR / "constrained_llm_taxonomies.json"
    csv_path = OUTPUT_DIR / "constrained_llm_taxonomies.csv"
    stats_path = OUTPUT_DIR / "constrained_llm_taxonomy_stats.json"
    mapping_path = OUTPUT_DIR / "constrained_llm_flat_mapping.json"

    export_taxonomies_json(taxonomies, json_path)
    export_taxonomies_csv(taxonomies, csv_path)
    export_flat_mapping(taxonomies, mapping_path)

    # Build and export stats
    stats = {
        'total_relationships': len(taxonomies),
        'num_supertags': len(supertags),
        'model_id': MODEL_ID,
        'device': DEVICE,
        'batch_size': BATCH_SIZE,
        'coverage': coverage,
    }
    if comparison:
        stats['existing_mapping_comparison'] = comparison

    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Exported statistics to: {stats_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    print(f"Discovered {len(taxonomies)} taxonomic relationships")
    print(f"  - Tags with single supertag: {coverage['single_parent_count']}")
    print(f"  - Tags with multiple supertags: {coverage['multi_parent_count']}")
    print(f"  - Unassigned tags: {coverage['unassigned_count']}")
    if comparison:
        print(f"  - Agreement with existing mapping: {comparison['agreement_rate']:.1%}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - constrained_llm_taxonomies.json (all relationships)")
    print(f"  - constrained_llm_taxonomies.csv (for spreadsheet viewing)")
    print(f"  - constrained_llm_flat_mapping.json (tag → supertag mapping)")
    print(f"  - constrained_llm_taxonomy_stats.json (statistics & comparison)")
    print("\n✓ Constrained taxonomy discovery complete!")
    print("\nThis validates your 95-supertag clustering with LLM reasoning.")


if __name__ == "__main__":
    main()

