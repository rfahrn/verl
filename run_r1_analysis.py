#!/usr/bin/env python3
"""
Run Complete R1 Grounding Reward Analysis
=========================================
Main script to execute all analysis and generate thesis materials.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Check for required packages
required_packages = ['numpy', 'matplotlib', 'pandas', 'seaborn', 'scipy', 'tabulate']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Missing packages: {missing_packages}")
    print("Installing missing packages...")
    import subprocess
    for package in missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
    print("Installation complete!\n")

# Import analysis modules
from r1_grounding_improved import compute_score, RewardConfig, compute_grounding_reward
from r1_visualizations import (
    visualize_reward_curves,
    visualize_heatmap, 
    visualize_3d_surface,
    visualize_edge_case_examples
)
from r1_analysis_demo import (
    print_mathematical_formulation,
    compare_reward_functions,
    analyze_edge_cases,
    demonstrate_parameter_sensitivity,
    run_synthetic_benchmark
)

def main():
    """Main execution function."""
    
    print("="*80)
    print(" R1 GROUNDING REWARD FUNCTION - COMPLETE ANALYSIS FOR THESIS")
    print("="*80)
    print("\nThis script will generate all analysis materials for your thesis.\n")
    
    # Part 1: Mathematical Formulation
    print("\n" + "="*80)
    print("PART 1: MATHEMATICAL FORMULATION")
    print("="*80)
    print_mathematical_formulation()
    
    # Part 2: Core Functionality Test
    print("\n" + "="*80)
    print("PART 2: CORE FUNCTIONALITY TEST")
    print("="*80)
    
    test_cases = [
        ("Perfect match", '<answer>[0.1, 0.2, 0.3, 0.4]</answer>', '[0.1, 0.2, 0.3, 0.4]'),
        ("True negative", '<answer></answer>', ''),
        ("Hallucination", '<answer>[0.1, 0.2, 0.3, 0.4]</answer>', ''),
        ("Missed detection", '<answer></answer>', '[0.1, 0.2, 0.3, 0.4]'),
        ("Partial overlap", '<answer>[0.15, 0.25, 0.35, 0.45]</answer>', '[0.1, 0.2, 0.3, 0.4]'),
        ("Multiple boxes", '<answer>[0.1,0.1,0.2,0.2],[0.5,0.5,0.6,0.6]</answer>', 
         '[0.1,0.1,0.2,0.2],[0.5,0.5,0.6,0.6]')
    ]
    
    print("\nTesting core reward function:")
    print("-" * 40)
    for name, pred, gt in test_cases:
        reward = compute_score("test", pred, gt)
        print(f"{name:20} → Reward: {reward:.3f}")
    
    # Part 3: Comparative Analysis
    print("\n" + "="*80)
    print("PART 3: COMPARATIVE ANALYSIS")
    print("="*80)
    compare_reward_functions()
    
    # Part 4: Edge Case Analysis
    print("\n" + "="*80)
    print("PART 4: EDGE CASE ANALYSIS")
    print("="*80)
    analyze_edge_cases()
    
    # Part 5: Parameter Sensitivity
    print("\n" + "="*80)
    print("PART 5: PARAMETER SENSITIVITY")
    print("="*80)
    demonstrate_parameter_sensitivity()
    
    # Part 6: Visualizations
    print("\n" + "="*80)
    print("PART 6: GENERATING VISUALIZATIONS")
    print("="*80)
    
    print("\nGenerating visualization plots...")
    print("(This may take a few moments)")
    print("-" * 40)
    
    try:
        # Suppress matplotlib output during generation
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        print("1. Creating reward curves...")
        visualize_reward_curves(save_path="r1_reward_curves.png")
        print("   ✓ Saved to r1_reward_curves.png")
        
        print("2. Creating heatmap analysis...")
        visualize_heatmap(save_path="r1_heatmap.png")
        print("   ✓ Saved to r1_heatmap.png")
        
        print("3. Creating edge case visualizations...")
        visualize_edge_case_examples(save_path="r1_edge_cases.png")
        print("   ✓ Saved to r1_edge_cases.png")
        
        print("4. Creating 3D surface plot...")
        visualize_3d_surface(save_path="r1_3d_surface.png")
        print("   ✓ Saved to r1_3d_surface.png")
        
    except Exception as e:
        print(f"   ⚠ Visualization generation had issues: {e}")
        print("   Continuing with analysis...")
    
    # Part 7: Synthetic Benchmark
    print("\n" + "="*80)
    print("PART 7: SYNTHETIC BENCHMARK")
    print("="*80)
    run_synthetic_benchmark()
    
    # Part 8: Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - THESIS MATERIALS READY")
    print("="*80)
    
    print("""
    Generated Files:
    ---------------
    Code Files:
    • r1_grounding_improved.py - Enhanced reward function with full documentation
    • r1_visualizations.py     - Comprehensive visualization module
    • r1_analysis_demo.py      - Detailed analysis demonstrations
    • run_r1_analysis.py       - This runner script
    
    Visualization Files:
    • r1_reward_curves.png     - Reward behavior for different scenarios
    • r1_heatmap.png          - Sensitivity analysis heatmaps
    • r1_edge_cases.png       - Visual examples of edge cases
    • r1_3d_surface.png       - 3D reward surface visualization
    • thesis_distribution.png  - Reward distribution analysis
    
    Key Improvements Over Original:
    ------------------------------
    1. Mathematical formulation with LaTeX equations
    2. Comprehensive edge case handling
    3. Detailed metrics (precision, recall, F1)
    4. Visualization tools for analysis
    5. Parameter sensitivity analysis
    6. Comparison with pure IoU rewards
    7. Robust multi-box support
    8. Clean, documented, thesis-ready code
    
    Recommended Thesis Presentation:
    -------------------------------
    1. Start with mathematical formulation (LaTeX equations)
    2. Show edge case handling table
    3. Present reward curves visualization
    4. Compare mAP vs IoU approaches
    5. Discuss parameter choices (τ=0.5, α=0.2)
    6. Show distribution analysis on your dataset
    
    Configuration for Production:
    ----------------------------
    config = RewardConfig(
        no_box_bonus=0.2,      # Reward for correct negatives
        iou_threshold=0.5,     # Standard COCO threshold
        normalize_coordinates=True,
        max_boxes=100
    )
    """)
    
    print("\nAll analysis complete! Your improved R1 grounding reward function is ready.")
    print("Check the generated files for thesis materials.\n")

if __name__ == "__main__":
    main()