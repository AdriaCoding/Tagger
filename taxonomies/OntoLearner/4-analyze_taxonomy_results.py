#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Taxonomy Results Analysis for BlindWiki

This script analyzes the discovered taxonomies, generates visualizations,
and optionally compares with the existing 16-tag manual mapping.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# %% [REGION 1] Configuration

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "data/ontolearner_output"
ANALYSIS_DIR = INPUT_DIR / "analysis"
TAGS_INPUT_DIR = SCRIPT_DIR / "data/ontolearner_input"
MAPPING_DIR = SCRIPT_DIR / "../mappings"

# Input files
TAXONOMIES_FILE = INPUT_DIR / "discovered_taxonomies.json"
TAGS_COUNTS_FILE = TAGS_INPUT_DIR / "tags_with_counts.json"
MANUAL_MAPPING_FILE = MAPPING_DIR / "16tags_mapping_dict.json"


# %% [REGION 2] Load data

def load_taxonomies(json_path):
    """Load discovered taxonomies"""
    print(f"Loading taxonomies from: {json_path}")
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Taxonomies file not found: {json_path}\n"
            f"Please run 3-run_taxonomy_discovery.py first"
        )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        taxonomies = json.load(f)
    
    print(f"Loaded {len(taxonomies)} taxonomic relationships")
    return taxonomies


def load_tags_with_counts(json_path):
    """Load tags with occurrence counts"""
    if not json_path.exists():
        print(f"Warning: Tags file not found: {json_path}")
        return {}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        tags_data = json.load(f)
    
    # Create lookup dict
    count_dict = {item['tag_normalized']: item['count'] for item in tags_data}
    print(f"Loaded occurrence counts for {len(count_dict)} tags")
    
    return count_dict


def load_manual_mapping(json_path):
    """Load manual 16-tag mapping if available"""
    if not json_path.exists():
        print(f"Info: Manual mapping not found: {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    print(f"Loaded manual mapping with {len(mapping)} entries")
    return mapping


# %% [REGION 3] Build taxonomy structures

def build_hierarchy_graph(taxonomies):
    """Build parent-child graph from taxonomies"""
    graph = defaultdict(list)  # parent -> [children]
    reverse_graph = defaultdict(list)  # child -> [parents]
    
    for rel in taxonomies:
        parent = rel['parent']
        child = rel['child']
        graph[parent].append(child)
        reverse_graph[child].append(parent)
    
    return graph, reverse_graph


def find_root_nodes(graph, reverse_graph):
    """Find potential root nodes (parents with no parents)"""
    all_parents = set(graph.keys())
    all_children = set(reverse_graph.keys())
    
    # Root nodes are parents that never appear as children
    roots = all_parents - all_children
    
    return sorted(roots)


def find_leaf_nodes(graph, reverse_graph):
    """Find leaf nodes (children with no children)"""
    all_parents = set(graph.keys())
    all_children = set(reverse_graph.keys())
    
    # Leaf nodes are children that never appear as parents
    leaves = all_children - all_parents
    
    return sorted(leaves)


def calculate_hierarchy_depths(graph, roots):
    """Calculate depth of each node in the hierarchy"""
    depths = {}
    
    def dfs(node, depth):
        if node in depths:
            depths[node] = max(depths[node], depth)
        else:
            depths[node] = depth
        
        if node in graph:
            for child in graph[node]:
                dfs(child, depth + 1)
    
    for root in roots:
        dfs(root, 0)
    
    return depths


# %% [REGION 4] Analysis functions

def analyze_hierarchy_structure(taxonomies):
    """Analyze the structure of the discovered hierarchy"""
    print("\n" + "="*60)
    print("HIERARCHY STRUCTURE ANALYSIS")
    print("="*60)
    
    graph, reverse_graph = build_hierarchy_graph(taxonomies)
    roots = find_root_nodes(graph, reverse_graph)
    leaves = find_leaf_nodes(graph, reverse_graph)
    
    # Calculate depths
    depths = calculate_hierarchy_depths(graph, roots)
    max_depth = max(depths.values()) if depths else 0
    
    # Nodes at each level
    depth_distribution = Counter(depths.values())
    
    analysis = {
        'total_nodes': len(set(graph.keys()).union(set(reverse_graph.keys()))),
        'root_nodes': len(roots),
        'leaf_nodes': len(leaves),
        'intermediate_nodes': len(set(graph.keys()).intersection(set(reverse_graph.keys()))),
        'max_depth': max_depth,
        'depth_distribution': dict(depth_distribution),
        'roots_list': roots[:20],  # Top 20 roots
        'sample_leaves': sorted(leaves)[:30]  # Sample 30 leaves
    }
    
    # Print summary
    print(f"Total nodes in hierarchy: {analysis['total_nodes']}")
    print(f"Root nodes (potential supertags): {analysis['root_nodes']}")
    print(f"Leaf nodes: {analysis['leaf_nodes']}")
    print(f"Intermediate nodes: {analysis['intermediate_nodes']}")
    print(f"Maximum hierarchy depth: {analysis['max_depth']}")
    
    print(f"\nNodes per depth level:")
    for depth in sorted(depth_distribution.keys()):
        print(f"  Level {depth}: {depth_distribution[depth]} nodes")
    
    print(f"\nTop root nodes (potential supertags):")
    for i, root in enumerate(roots[:20], 1):
        child_count = len(graph[root])
        print(f"  {i:2d}. {root:30s} ({child_count} children)")
    
    return analysis, graph, reverse_graph


def compare_with_manual_mapping(taxonomies, manual_mapping, tag_counts):
    """Compare discovered taxonomies with manual 16-tag mapping"""
    if manual_mapping is None:
        print("\nSkipping manual mapping comparison (file not found)")
        return None
    
    print("\n" + "="*60)
    print("COMPARISON WITH MANUAL 16-TAG MAPPING")
    print("="*60)
    
    # Normalize manual mapping (original tag -> supertag)
    # The file maps original tags to English supertags
    normalized_mapping = {}
    supertags = set()
    
    for original_tag, supertag in manual_mapping.items():
        # Normalize both
        original_norm = original_tag.lower().strip().replace(' ', '_')
        supertag_norm = supertag.lower().strip().replace(' ', '_')
        normalized_mapping[original_norm] = supertag_norm
        supertags.add(supertag_norm)
    
    print(f"Manual mapping: {len(normalized_mapping)} tags → {len(supertags)} supertags")
    
    # Build discovered hierarchy
    graph, reverse_graph = build_hierarchy_graph(taxonomies)
    discovered_roots = find_root_nodes(graph, reverse_graph)
    
    # Compare supertags
    manual_supertags = supertags
    discovered_supertags = set(discovered_roots)
    
    overlap = manual_supertags.intersection(discovered_supertags)
    only_manual = manual_supertags - discovered_supertags
    only_discovered = discovered_supertags - manual_supertags
    
    print(f"\nSupertag comparison:")
    print(f"  Manual supertags: {len(manual_supertags)}")
    print(f"  Discovered root tags: {len(discovered_supertags)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Only in manual: {len(only_manual)}")
    print(f"  Only in discovered: {len(only_discovered)}")
    
    if overlap:
        print(f"\nOverlapping supertags:")
        for tag in sorted(overlap):
            print(f"  - {tag}")
    
    if only_manual:
        print(f"\nSupertags only in manual mapping:")
        for tag in sorted(only_manual)[:10]:
            print(f"  - {tag}")
    
    # Analyze agreement for tags with manual mappings
    agreement_count = 0
    disagreement_examples = []
    
    for tag, manual_parent in normalized_mapping.items():
        if tag in reverse_graph:
            discovered_parents = reverse_graph[tag]
            if manual_parent in discovered_parents:
                agreement_count += 1
            else:
                disagreement_examples.append({
                    'tag': tag,
                    'manual_parent': manual_parent,
                    'discovered_parents': discovered_parents
                })
    
    tags_with_discovered_parents = len([t for t in normalized_mapping.keys() if t in reverse_graph])
    
    comparison = {
        'manual_supertags': sorted(list(manual_supertags)),
        'discovered_roots': sorted(list(discovered_supertags)),
        'overlap_count': len(overlap),
        'overlap_tags': sorted(list(overlap)),
        'only_manual': sorted(list(only_manual)),
        'only_discovered': sorted(list(only_discovered))[:50],  # Limit size
        'agreement_count': agreement_count,
        'tags_with_discovered_parents': tags_with_discovered_parents,
        'agreement_rate': agreement_count / tags_with_discovered_parents if tags_with_discovered_parents > 0 else 0,
        'disagreement_examples': disagreement_examples[:20]
    }
    
    print(f"\nParent assignment agreement:")
    print(f"  Tags with discovered parents: {tags_with_discovered_parents}")
    print(f"  Agreements with manual mapping: {agreement_count}")
    print(f"  Agreement rate: {comparison['agreement_rate']:.1%}")
    
    return comparison


# %% [REGION 5] Visualization functions

def visualize_parent_frequency(taxonomies, tag_counts, output_path):
    """Create bar chart of most frequent parent tags"""
    df = pd.DataFrame(taxonomies)
    parent_counts = df['parent'].value_counts().head(20)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot
    bars = ax.barh(range(len(parent_counts)), parent_counts.values)
    ax.set_yticks(range(len(parent_counts)))
    ax.set_yticklabels(parent_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Children')
    ax.set_title('Top 20 Parent Tags by Number of Children')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (tag, count) in enumerate(parent_counts.items()):
        occurrence = tag_counts.get(tag, 0)
        ax.text(count, i, f' {count} ({occurrence} occ.)', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parent frequency chart to: {output_path}")


def visualize_depth_distribution(depths, output_path):
    """Create bar chart of hierarchy depth distribution"""
    depth_counts = Counter(depths.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    depths_sorted = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in depths_sorted]
    
    ax.bar(depths_sorted, counts)
    ax.set_xlabel('Hierarchy Depth')
    ax.set_ylabel('Number of Tags')
    ax.set_title('Distribution of Tags by Hierarchy Depth')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for depth, count in zip(depths_sorted, counts):
        ax.text(depth, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved depth distribution chart to: {output_path}")


def generate_taxonomy_tree(graph, roots, tag_counts, output_path, max_depth=3):
    """Generate text-based tree visualization"""
    lines = []
    lines.append("TAXONOMY TREE VISUALIZATION")
    lines.append("="*60)
    lines.append(f"Showing top {len(roots[:10])} root nodes, max depth {max_depth}")
    lines.append("="*60)
    lines.append("")
    
    def print_tree(node, prefix="", depth=0):
        if depth > max_depth:
            return
        
        count = tag_counts.get(node, 0)
        lines.append(f"{prefix}{node} (occ: {count})")
        
        if node in graph and depth < max_depth:
            children = sorted(graph[node])
            for i, child in enumerate(children[:10]):  # Limit children shown
                is_last = (i == len(children) - 1) or (i == 9)
                child_prefix = prefix + ("└── " if is_last else "├── ")
                next_prefix = prefix + ("    " if is_last else "│   ")
                
                child_count = tag_counts.get(child, 0)
                lines.append(f"{child_prefix}{child} (occ: {child_count})")
                
                # Recurse for grandchildren
                if child in graph and depth < max_depth - 1:
                    grandchildren = sorted(graph[child])[:5]  # Limit grandchildren
                    for j, grandchild in enumerate(grandchildren):
                        is_last_gc = (j == len(grandchildren) - 1)
                        gc_prefix = next_prefix + ("└── " if is_last_gc else "├── ")
                        gc_count = tag_counts.get(grandchild, 0)
                        lines.append(f"{gc_prefix}{grandchild} (occ: {gc_count})")
            
            if len(children) > 10:
                lines.append(f"{prefix}    ... and {len(children) - 10} more children")
        
        lines.append("")
    
    # Print trees for top roots
    for root in roots[:10]:
        print_tree(root)
        lines.append("")
    
    if len(roots) > 10:
        lines.append(f"... and {len(roots) - 10} more root nodes")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Saved taxonomy tree to: {output_path}")


# %% [REGION 6] Main execution

def main():
    print("="*60)
    print("TAXONOMY RESULTS ANALYSIS")
    print("="*60)
    
    # Create analysis directory
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Analysis output directory: {ANALYSIS_DIR}\n")
    
    # Load data
    taxonomies = load_taxonomies(TAXONOMIES_FILE)
    tag_counts = load_tags_with_counts(TAGS_COUNTS_FILE)
    manual_mapping = load_manual_mapping(MANUAL_MAPPING_FILE)
    
    # Analyze hierarchy structure
    structure_analysis, graph, reverse_graph = analyze_hierarchy_structure(taxonomies)
    
    # Calculate depths
    roots = find_root_nodes(graph, reverse_graph)
    depths = calculate_hierarchy_depths(graph, roots)
    
    # Compare with manual mapping
    comparison = compare_with_manual_mapping(taxonomies, manual_mapping, tag_counts)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    parent_freq_path = ANALYSIS_DIR / "parent_frequency.png"
    depth_dist_path = ANALYSIS_DIR / "depth_distribution.png"
    tree_path = ANALYSIS_DIR / "taxonomy_tree.txt"
    
    visualize_parent_frequency(taxonomies, tag_counts, parent_freq_path)
    visualize_depth_distribution(depths, depth_dist_path)
    generate_taxonomy_tree(graph, roots, tag_counts, tree_path, max_depth=3)
    
    # Export analysis report
    print("\n" + "="*60)
    print("EXPORTING ANALYSIS REPORT")
    print("="*60)
    
    report = {
        'structure': structure_analysis,
        'comparison_with_manual': comparison
    }
    
    report_path = ANALYSIS_DIR / "analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Saved analysis report to: {report_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results saved to: {ANALYSIS_DIR}")
    print("\nFiles created:")
    print("  - analysis_report.json (detailed metrics)")
    print("  - taxonomy_tree.txt (text visualization)")
    print("  - parent_frequency.png (chart)")
    print("  - depth_distribution.png (chart)")
    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()

