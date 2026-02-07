#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Constrained LLM Taxonomy Discovery for BlindWiki Tags

This script uses the existing 16 supertags as fixed root categories and asks
the LLM to classify each tag under the appropriate supertag(s).

Strategy:
  - For each tag, ask the LLM: "Is supertag X a parent of tag Y?"
  - Only evaluates: ~596 tags × 16 supertags = ~9,536 pairs
  - Much faster than full LLM (script 5) or RAG (script 6)

Expected runtime:
  - CPU: ~30 minutes - 1 hour
  - GPU: ~5-15 minutes

This is the RECOMMENDED starting point for practical taxonomy building.
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
MANUAL_MAPPING_FILE = MAPPING_DIR / "16tags_mapping_dict.json"

# The 16 supertags (from the manual mapping)
SUPERTAGS = [
    "Ambient noise",
    "Sorocaba",
    "Neighborhood",
    "Paseo",
    "Accessible",
    "Restaurant",
    "Water",
    "Dangerous",
    "Culture",
    "Tram",
    "Maritime",
    "Music",
    "Bridges",
    "Piazza",
    "Garden",
    "Tactile",
    "Dead",
    "Venice",
    "Lande",
    "University",
    "Art",
    "Streets",
    "Caution",
    "Park",
    "Iglesia",
    "Wells",
    "Museum",
    "Shop",
    "Mislata",
    "Birds",
    "Bar",
    "Crowded Environment",
    "Odor",
    "Sport",
    "Tranquillity",
    "Work",
    ". Reference",
    "Tree",
    "Audible traffic lights",
    "Test",
    "Hiking",
    "History",
    "Library",
    "Easter",
    "Rain",
    "Hospital",
    "Vision",
    "Door",
    "Friendship",
    "Bike",
    "Girls",
    "Bells",
    "Steps",
    "He Lives Together",
    "Nature",
    "Movement",
    "33Bienal",
    "Binary",
    "Sole",
    "Obstacle",
    "Car",
    "Germany",
    "Market",
    "Temporary",
    "Machine",
    "Fruits",
    "Voices",
    "Fish",
    "Scale",
    "Meeting",
    "Touch",
    "Night",
    "Turismo",
    "Foundations",
    "Sensations",
    "Carnevale",
    "Hello",
    "Wall",
    "Theater",
    "Laramara",
    "Calle Closed",
    "Project",
    "Words And More",
    "Vaporetto",
    "Sculptures",
    "Luci",
    "Perfumes",
    "Memorial",
    "Vento",
    "People",
    "Description",
    "Sculpture",
    "Blindwiki",
    "Headquarters",
    "Legends",
    "Stop"
]

# LLM parameters
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 5
DEVICE = "cpu"  # Use 'cuda' if GPU available


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


def load_manual_mapping(json_path):
    """Load the manual 16-tag mapping for comparison"""
    if not json_path.exists():
        print(f"Info: Manual mapping not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    print(f"Loaded manual mapping with {len(mapping)} entries")
    return mapping


# %% [REGION 3] Build constrained dataset

def build_constrained_dataset(tags, supertags):
    """Build dataset of (supertag, tag) pairs for LLM evaluation.

    For each tag, we create one prompt per supertag asking:
    "Is supertag X a superclass of tag Y?"

    This uses the same prompt template as OntoLearner's StandardizedPrompting
    for taxonomy-discovery, ensuring consistency with the library.
    """
    prompting = StandardizedPrompting(task='taxonomy-discovery')

    # Filter out supertags from the tag list to avoid self-relationships
    tags_to_classify = [t for t in tags if t not in supertags]

    n_pairs = len(tags_to_classify) * len(supertags)
    print(f"\nBuilding constrained dataset:")
    print(f"  Tags to classify: {len(tags_to_classify)}")
    print(f"  Supertags (fixed roots): {len(supertags)}")
    print(f"  Total pairs: {n_pairs:,}")

    dataset = []
    for tag in tags_to_classify:
        for supertag in supertags:
            dataset.append({
                "parent": supertag,
                "child": tag,
                "prompt": prompting.format(parent=supertag, child=tag)
            })

    return dataset


# %% [REGION 4] Run constrained taxonomy discovery

def run_constrained_taxonomy_discovery(tags, supertags, model_id=MODEL_ID,
                                        batch_size=BATCH_SIZE,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        device=DEVICE):
    """Execute constrained LLM taxonomy discovery against fixed supertags"""
    print("\n" + "=" * 60)
    print("CONSTRAINED LLM TAXONOMY DISCOVERY")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Supertags: {', '.join(supertags)}")
    print(f"Number of tags: {len(tags)}")

    # Build the constrained dataset
    dataset = build_constrained_dataset(tags, supertags)
    n_pairs = len(dataset)

    # Estimate runtime
    pairs_per_sec_cpu = 2.5
    pairs_per_sec_gpu = 15
    rate = pairs_per_sec_gpu if device == 'cuda' else pairs_per_sec_cpu
    est_minutes = n_pairs / rate / 60
    print(f"Estimated runtime: {est_minutes:.0f} minutes")

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
    # This uses the same mechanism as the full LLM approach but with
    # our constrained dataset instead of all N×(N-1)/2 pairs
    print(f"\nEvaluating {n_pairs:,} (supertag, tag) pairs...")
    start_discovery = time.time()

    taxonomies = learner._taxonomy_discovery_predict(dataset=dataset)

    discovery_time = time.time() - start_discovery
    print(f"\n✓ Discovery complete in {discovery_time:.2f} seconds "
          f"({discovery_time / 60:.1f} minutes)")
    print(f"Found {len(taxonomies)} taxonomic relationships")

    return taxonomies


# %% [REGION 5] Analyze and compare results

def compare_with_manual_mapping(taxonomies, manual_mapping):
    """Compare discovered constrained taxonomies with manual mapping"""
    if manual_mapping is None:
        print("\nSkipping manual mapping comparison (file not found)")
        return None

    print("\n" + "=" * 60)
    print("COMPARISON WITH MANUAL 16-TAG MAPPING")
    print("=" * 60)

    # Normalize manual mapping: original_tag -> supertag (lowercase)
    normalized_manual = {}
    for original_tag, supertag in manual_mapping.items():
        tag_norm = original_tag.lower().strip().replace(' ', '_')
        supertag_norm = supertag.lower().strip().replace(' ', '_')
        normalized_manual[tag_norm] = supertag_norm

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
    tags_compared = 0

    for tag, manual_parent in normalized_manual.items():
        if tag in discovered_mapping:
            tags_compared += 1
            discovered_parents = discovered_mapping[tag]
            if manual_parent in discovered_parents:
                agreement_count += 1
            else:
                disagreement_examples.append({
                    'tag': tag,
                    'manual_parent': manual_parent,
                    'discovered_parents': discovered_parents
                })

    agreement_rate = agreement_count / tags_compared if tags_compared > 0 else 0

    print(f"Tags in manual mapping: {len(normalized_manual)}")
    print(f"Tags with discovered parents: {tags_compared}")
    print(f"Agreements: {agreement_count}")
    print(f"Agreement rate: {agreement_rate:.1%}")

    if disagreement_examples:
        print(f"\nSample disagreements (showing first 15):")
        for ex in disagreement_examples[:15]:
            print(f"  {ex['tag']:30s}  manual: {ex['manual_parent']:20s}  "
                  f"discovered: {', '.join(ex['discovered_parents'])}")

    # Tags not assigned to any supertag
    unassigned = [tag for tag in normalized_manual.keys()
                  if tag not in discovered_mapping]
    if unassigned:
        print(f"\nTags from manual mapping with no discovered parent ({len(unassigned)}):")
        for tag in unassigned[:20]:
            print(f"  - {tag} (manual: {normalized_manual[tag]})")

    comparison = {
        'tags_in_manual_mapping': len(normalized_manual),
        'tags_compared': tags_compared,
        'agreement_count': agreement_count,
        'agreement_rate': agreement_rate,
        'unassigned_count': len(unassigned),
        'disagreement_examples': disagreement_examples[:30],
        'unassigned_tags': unassigned[:50]
    }

    return comparison


def analyze_coverage(taxonomies, tags, supertags):
    """Analyze how well the supertags cover all tags"""
    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)

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
    all_tags_set = set(t for t in tags if t not in supertags)
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
        children = [rel['child'] for rel in taxonomies if rel['parent'] == supertag]
        supertag_counts[supertag] = len(children)

    print(f"\nChildren per supertag:")
    for supertag in sorted(supertag_counts, key=supertag_counts.get, reverse=True):
        count = supertag_counts[supertag]
        bar = "█" * (count // 2)
        print(f"  {supertag:25s} {count:4d} {bar}")

    if multi_parent:
        print(f"\nSample tags with multiple supertags:")
        for tag, parents in list(multi_parent.items())[:15]:
            print(f"  {tag:30s} → {', '.join(parents)}")

    if unassigned:
        print(f"\nSample unassigned tags:")
        for tag in sorted(unassigned)[:20]:
            print(f"  - {tag}")

    analysis = {
        'total_tags': len(all_tags_set),
        'single_parent_count': len(single_parent),
        'multi_parent_count': len(multi_parent),
        'unassigned_count': len(unassigned),
        'supertag_child_counts': supertag_counts,
        'multi_parent_tags': {t: p for t, p in list(multi_parent.items())[:50]},
        'unassigned_tags': sorted(list(unassigned))[:100]
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
    print("CONSTRAINED LLM TAXONOMY DISCOVERY (16 SUPERTAGS)")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Load data
    tags = load_tags(INPUT_FILE)
    manual_mapping = load_manual_mapping(MANUAL_MAPPING_FILE)

    # Print supertags
    print(f"\nFixed supertag roots ({len(SUPERTAGS)}):")
    for st in SUPERTAGS:
        print(f"  - {st}")

    # Run constrained taxonomy discovery
    taxonomies = run_constrained_taxonomy_discovery(tags, SUPERTAGS)

    if not taxonomies:
        print("\n⚠ No taxonomic relationships discovered!")
        return

    # Print samples
    print_sample_taxonomies(taxonomies, n=30)

    # Analyze coverage
    coverage = analyze_coverage(taxonomies, tags, SUPERTAGS)

    # Compare with manual mapping
    comparison = compare_with_manual_mapping(taxonomies, manual_mapping)

    # Export results
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)

    json_path = OUTPUT_DIR / "constrained_taxonomies.json"
    csv_path = OUTPUT_DIR / "constrained_taxonomies.csv"
    stats_path = OUTPUT_DIR / "constrained_taxonomy_stats.json"
    mapping_path = OUTPUT_DIR / "constrained_flat_mapping.json"

    export_taxonomies_json(taxonomies, json_path)
    export_taxonomies_csv(taxonomies, csv_path)
    export_flat_mapping(taxonomies, mapping_path)

    # Build and export stats
    stats = {
        'total_relationships': len(taxonomies),
        'supertags': SUPERTAGS,
        'model_id': MODEL_ID,
        'device': DEVICE,
        'coverage': coverage,
    }
    if comparison:
        stats['manual_comparison'] = comparison

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
        print(f"  - Agreement with manual mapping: {comparison['agreement_rate']:.1%}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print(f"  - constrained_taxonomies.json (all relationships)")
    print(f"  - constrained_taxonomies.csv (for spreadsheet viewing)")
    print(f"  - constrained_flat_mapping.json (tag → supertag mapping)")
    print(f"  - constrained_taxonomy_stats.json (statistics & comparison)")
    print("\n✓ Constrained taxonomy discovery complete!")


if __name__ == "__main__":
    main()

