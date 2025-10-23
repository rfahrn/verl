"""
R1 Grounding Reward Analysis and Demonstration
==============================================
Comprehensive analysis script for thesis presentation showing:
1. Mathematical formulation
2. Edge case handling
3. Comparison with pure IoU
4. Reward behavior analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import pandas as pd
from tabulate import tabulate

# Import improved reward function
from r1_grounding_improved import (
    compute_grounding_reward,
    compute_iou,
    RewardConfig,
    compute_map_detailed,
    analyze_reward_distribution
)

# Import visualization module
from r1_visualizations import (
    visualize_reward_curves,
    visualize_heatmap,
    visualize_3d_surface,
    visualize_edge_case_examples,
    plot_reward_distribution
)


def print_mathematical_formulation():
    """
    Print the mathematical formulation in LaTeX format for thesis.
    """
    latex_formula = r"""
    ================================================================================
    MATHEMATICAL FORMULATION FOR THESIS
    ================================================================================
    
    Main Reward Function:
    --------------------
    
    $$
    R(\hat{B}, B) = \begin{cases}
        \alpha,                & \text{if } |\hat{B}| = 0 \land |B| = 0 \quad \text{(True Negative)} \\
        0,                     & \text{if } |\hat{B}| > 0 \land |B| = 0 \quad \text{(Hallucination)} \\
        0,                     & \text{if } |\hat{B}| = 0 \land |B| > 0 \quad \text{(Missed Detection)} \\
        \text{mAP}(\hat{B}, B; \tau), & \text{if } |\hat{B}| > 0 \land |B| > 0 \quad \text{(Detection)}
    \end{cases}
    $$
    
    Where:
    - $\hat{B} = \{\hat{b}_1, \hat{b}_2, ..., \hat{b}_n\}$: Set of predicted bounding boxes
    - $B = \{b_1, b_2, ..., b_m\}$: Set of ground truth bounding boxes
    - $\alpha$: No-box bonus reward (default: 0.2)
    - $\tau$: IoU threshold (default: 0.5)
    
    Mean Average Precision (mAP):
    -----------------------------
    
    $$
    \text{mAP}(\hat{B}, B; \tau) = \text{AP}@\tau = \int_0^1 P(r) \, dr
    $$
    
    Where precision $P$ at recall $r$ is computed as:
    
    $$
    P = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad R = \frac{\text{TP}}{\text{TP} + \text{FN}}
    $$
    
    Intersection over Union (IoU):
    ------------------------------
    
    $$
    \text{IoU}(\hat{b}, b) = \frac{|\hat{b} \cap b|}{|\hat{b} \cup b|} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
    $$
    
    Matching Criterion:
    -------------------
    
    A prediction $\hat{b}_i$ matches ground truth $b_j$ if:
    1. $\text{IoU}(\hat{b}_i, b_j) \geq \tau$
    2. $b_j$ has not been matched to another prediction
    
    Edge Case Formalization:
    ------------------------
    
    1. **True Negative (TN)**: $|\hat{B}| = 0 \land |B| = 0 \Rightarrow R = \alpha$
       - Correctly predicting absence of objects
    
    2. **False Positive (FP)**: $|\hat{B}| > 0 \land |B| = 0 \Rightarrow R = 0$
       - Model hallucinates non-existent objects
    
    3. **False Negative (FN)**: $|\hat{B}| = 0 \land |B| > 0 \Rightarrow R = 0$
       - Model fails to detect existing objects
    
    4. **One-to-Many**: $|\hat{B}| = 1 \land |B| > 1$
       - Single prediction for multiple objects
       - $R = \text{mAP}$ with at most one match
    
    5. **Many-to-One**: $|\hat{B}| > 1 \land |B| = 1$
       - Multiple predictions for single object
       - Only best prediction counts as TP
    
    6. **Many-to-Many**: $|\hat{B}| > 1 \land |B| > 1$
       - Full mAP calculation with greedy matching
    
    ================================================================================
    """
    print(latex_formula)


def compare_reward_functions():
    """
    Compare mAP-based reward with pure IoU reward.
    """
    print("\n" + "="*80)
    print("COMPARISON: mAP@0.5 vs Pure IoU Reward")
    print("="*80)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Perfect Match',
            'pred': '[0.1,0.1,0.5,0.5]',
            'gt': '[0.1,0.1,0.5,0.5]',
            'expected_iou': 1.0
        },
        {
            'name': 'Good Match (IoU=0.7)',
            'pred': '[0.15,0.15,0.55,0.55]',
            'gt': '[0.1,0.1,0.5,0.5]',
            'expected_iou': 0.64  # Approximate
        },
        {
            'name': 'Threshold Match (IoU=0.5)',
            'pred': '[0.2,0.2,0.6,0.6]',
            'gt': '[0.1,0.1,0.5,0.5]',
            'expected_iou': 0.44  # Approximate
        },
        {
            'name': 'Poor Match (IoU=0.3)',
            'pred': '[0.3,0.3,0.7,0.7]',
            'gt': '[0.1,0.1,0.5,0.5]',
            'expected_iou': 0.14  # Approximate
        },
        {
            'name': 'No Overlap',
            'pred': '[0.6,0.6,0.9,0.9]',
            'gt': '[0.1,0.1,0.4,0.4]',
            'expected_iou': 0.0
        },
        {
            'name': 'True Negative',
            'pred': '',
            'gt': '',
            'expected_iou': None
        },
        {
            'name': 'Hallucination',
            'pred': '[0.1,0.1,0.5,0.5]',
            'gt': '',
            'expected_iou': None
        },
        {
            'name': 'Missed Detection',
            'pred': '',
            'gt': '[0.1,0.1,0.5,0.5]',
            'expected_iou': None
        },
        {
            'name': 'Multiple Correct',
            'pred': '[0.1,0.1,0.3,0.3],[0.4,0.4,0.6,0.6]',
            'gt': '[0.1,0.1,0.3,0.3],[0.4,0.4,0.6,0.6]',
            'expected_iou': 1.0
        },
        {
            'name': 'Duplicate Predictions',
            'pred': '[0.1,0.1,0.3,0.3],[0.12,0.12,0.32,0.32]',
            'gt': '[0.1,0.1,0.3,0.3]',
            'expected_iou': None
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        # Prepare inputs
        pred_str = f"<answer>{scenario['pred']}</answer>" if scenario['pred'] else "<answer></answer>"
        
        # Calculate mAP reward
        config = RewardConfig(iou_threshold=0.5)
        mAP_reward = compute_grounding_reward(pred_str, scenario['gt'], config)
        
        # Calculate pure IoU reward (simplified)
        if scenario['pred'] and scenario['gt']:
            pred_boxes = eval(f"[{scenario['pred']}]" if ',' in scenario['pred'] and not '],[' in scenario['pred'] 
                             else f"[{scenario['pred']}]".replace('],[', '],['))
            gt_boxes = eval(f"[{scenario['gt']}]" if ',' in scenario['gt'] and not '],[' in scenario['gt'] 
                           else f"[{scenario['gt']}]".replace('],[', '],['))
            
            if isinstance(pred_boxes[0], list) and isinstance(gt_boxes[0], list):
                # Multiple boxes - average IoU
                ious = []
                for pb in pred_boxes:
                    for gb in gt_boxes:
                        if isinstance(pb, list) and isinstance(gb, list):
                            ious.append(compute_iou(pb, gb))
                iou_reward = np.mean(ious) if ious else 0.0
            else:
                iou_reward = compute_iou(pred_boxes, gt_boxes)
        elif not scenario['pred'] and not scenario['gt']:
            iou_reward = 1.0  # Perfect match for true negative
        else:
            iou_reward = 0.0
        
        results.append({
            'Scenario': scenario['name'],
            'mAP@0.5': f"{mAP_reward:.3f}",
            'Pure IoU': f"{iou_reward:.3f}",
            'Difference': f"{mAP_reward - iou_reward:+.3f}"
        })
    
    # Print comparison table
    df = pd.DataFrame(results)
    print("\n" + tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-------------")
    print("1. mAP@0.5 provides binary reward above/below IoU threshold")
    print("2. Pure IoU gives continuous reward but doesn't capture precision-recall trade-off")
    print("3. mAP handles multiple boxes better through proper matching")
    print("4. True negatives are explicitly rewarded in mAP formulation")
    print("5. mAP penalizes duplicates while IoU might give false high scores")


def analyze_edge_cases():
    """
    Detailed analysis of edge cases with metrics.
    """
    print("\n" + "="*80)
    print("EDGE CASE ANALYSIS WITH DETAILED METRICS")
    print("="*80)
    
    edge_cases = {
        'True Negative': {
            'pred': '',
            'gt': '',
            'description': 'No predictions, no ground truth boxes'
        },
        'Perfect Single Match': {
            'pred': '[0.2,0.2,0.4,0.4]',
            'gt': '[0.2,0.2,0.4,0.4]',
            'description': 'Single box with perfect overlap'
        },
        'Partial Overlap (Good)': {
            'pred': '[0.22,0.22,0.42,0.42]',
            'gt': '[0.2,0.2,0.4,0.4]',
            'description': 'IoU > 0.5, counts as true positive'
        },
        'Partial Overlap (Poor)': {
            'pred': '[0.3,0.3,0.5,0.5]',
            'gt': '[0.2,0.2,0.4,0.4]',
            'description': 'IoU < 0.5, counts as false positive'
        },
        'Hallucination': {
            'pred': '[0.2,0.2,0.4,0.4]',
            'gt': '',
            'description': 'Model predicts box when none exists'
        },
        'Missed Detection': {
            'pred': '',
            'gt': '[0.2,0.2,0.4,0.4]',
            'description': 'Model misses existing box'
        },
        'One-to-Many': {
            'pred': '[0.1,0.1,0.5,0.5]',
            'gt': '[0.1,0.1,0.3,0.3],[0.3,0.3,0.5,0.5]',
            'description': 'Single pred covers multiple GT boxes'
        },
        'Many-to-One': {
            'pred': '[0.1,0.1,0.3,0.3],[0.25,0.25,0.45,0.45]',
            'gt': '[0.1,0.1,0.3,0.3]',
            'description': 'Multiple preds for single GT box'
        },
        'Complex Multi-box': {
            'pred': '[0.1,0.1,0.2,0.2],[0.3,0.3,0.4,0.4],[0.5,0.5,0.6,0.6]',
            'gt': '[0.1,0.1,0.2,0.2],[0.3,0.3,0.4,0.4]',
            'description': '3 predictions, 2 GT boxes'
        }
    }
    
    results = []
    
    for case_name, case_data in edge_cases.items():
        pred_str = f"<answer>{case_data['pred']}</answer>" if case_data['pred'] else "<answer></answer>"
        
        # Get detailed metrics
        config = RewardConfig()
        metrics = compute_grounding_reward(pred_str, case_data['gt'], config, return_details=True)
        
        results.append({
            'Edge Case': case_name,
            'Reward': f"{metrics['reward']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
            'Type': metrics['edge_case']
        })
        
        print(f"\n{case_name}:")
        print(f"  Description: {case_data['description']}")
        print(f"  Classification: {metrics['edge_case']}")
        print(f"  Metrics: Reward={metrics['reward']:.3f}, P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    # Summary table
    print("\n" + "="*80)
    print("EDGE CASE SUMMARY TABLE")
    print("="*80)
    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


def demonstrate_parameter_sensitivity():
    """
    Show how different parameters affect the reward function.
    """
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Test with different configurations
    test_pred = '<answer>[0.22,0.22,0.42,0.42]</answer>'
    test_gt = '[0.2,0.2,0.4,0.4]'
    
    print("\n1. IoU Threshold Sensitivity:")
    print("-" * 40)
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for thresh in thresholds:
        config = RewardConfig(iou_threshold=thresh)
        reward = compute_grounding_reward(test_pred, test_gt, config)
        print(f"  τ = {thresh:.1f}: Reward = {reward:.3f}")
    
    print("\n2. No-Box Bonus Sensitivity:")
    print("-" * 40)
    empty_pred = '<answer></answer>'
    empty_gt = ''
    bonuses = [0.0, 0.1, 0.2, 0.3, 0.5]
    for bonus in bonuses:
        config = RewardConfig(no_box_bonus=bonus)
        reward = compute_grounding_reward(empty_pred, empty_gt, config)
        print(f"  α = {bonus:.1f}: Reward = {reward:.3f}")
    
    print("\n3. IoU Power (Strictness) Effect:")
    print("-" * 40)
    powers = [0.5, 1.0, 2.0, 3.0]
    for power in powers:
        config = RewardConfig(iou_power=power)
        metrics = compute_grounding_reward(test_pred, test_gt, config, return_details=True)
        print(f"  IoU^{power:.1f}: Reward = {metrics['reward']:.3f}")


def generate_thesis_ready_analysis():
    """
    Generate complete analysis for thesis presentation.
    """
    print("\n" + "="*80)
    print("R1 GROUNDING REWARD FUNCTION - THESIS ANALYSIS")
    print("="*80)
    
    # 1. Mathematical Formulation
    print_mathematical_formulation()
    
    # 2. Comparison Analysis
    compare_reward_functions()
    
    # 3. Edge Case Analysis
    analyze_edge_cases()
    
    # 4. Parameter Sensitivity
    demonstrate_parameter_sensitivity()
    
    # 5. Generate Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    print("\nGenerating comprehensive visualizations...")
    
    # Create all plots
    visualize_reward_curves(save_path="thesis_reward_curves.png")
    print("✓ Reward curves saved to thesis_reward_curves.png")
    
    visualize_heatmap(save_path="thesis_heatmap.png")
    print("✓ Heatmap analysis saved to thesis_heatmap.png")
    
    visualize_edge_case_examples(save_path="thesis_edge_cases.png")
    print("✓ Edge case examples saved to thesis_edge_cases.png")
    
    # 6. Summary Statistics
    print("\n" + "="*80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*80)
    
    print("""
    Key Advantages of mAP@0.5 over Pure IoU:
    ----------------------------------------
    1. Binary decision boundary provides clear success/failure signal
    2. Handles multiple objects through proper matching algorithm
    3. Incorporates precision-recall trade-off naturally
    4. Penalizes both false positives and false negatives
    5. Standard metric in object detection literature
    
    Recommended Configuration:
    -------------------------
    • IoU Threshold (τ): 0.5 (standard in COCO/PASCAL VOC)
    • No-Box Bonus (α): 0.2 (encourages but doesn't over-reward negatives)
    • Matching: Greedy (efficient and deterministic)
    
    Implementation Highlights:
    -------------------------
    • Robust parsing handles multiple bbox formats
    • Comprehensive edge case handling
    • Detailed metrics for debugging
    • Visualization tools for analysis
    • Clean, modular, thesis-ready code
    """)


def run_synthetic_benchmark():
    """
    Run benchmark on synthetic data to show reward distribution.
    """
    print("\n" + "="*80)
    print("SYNTHETIC BENCHMARK ANALYSIS")
    print("="*80)
    
    np.random.seed(42)
    n_samples = 500
    
    # Generate diverse test cases
    predictions = []
    ground_truths = []
    categories = []
    
    # Distribution of cases
    distributions = {
        'perfect': 0.15,
        'good': 0.25,
        'marginal': 0.20,
        'poor': 0.15,
        'hallucination': 0.10,
        'missed': 0.10,
        'true_negative': 0.05
    }
    
    for _ in range(n_samples):
        case_type = np.random.choice(list(distributions.keys()), 
                                    p=list(distributions.values()))
        categories.append(case_type)
        
        if case_type == 'perfect':
            # Perfect match
            box = [np.random.rand()*0.5, np.random.rand()*0.5, 
                  np.random.rand()*0.5+0.5, np.random.rand()*0.5+0.5]
            pred = f"<answer>[{','.join(map(lambda x: f'{x:.3f}', box))}]</answer>"
            gt = f"[{','.join(map(lambda x: f'{x:.3f}', box))}]"
            
        elif case_type == 'good':
            # Good overlap (IoU > 0.5)
            gt_box = [np.random.rand()*0.5, np.random.rand()*0.5,
                     np.random.rand()*0.5+0.5, np.random.rand()*0.5+0.5]
            noise = np.random.randn(4) * 0.05  # Small noise
            pred_box = [gt_box[i] + noise[i] for i in range(4)]
            pred = f"<answer>[{','.join(map(lambda x: f'{x:.3f}', pred_box))}]</answer>"
            gt = f"[{','.join(map(lambda x: f'{x:.3f}', gt_box))}]"
            
        elif case_type == 'marginal':
            # Marginal overlap (IoU ≈ 0.5)
            gt_box = [0.3, 0.3, 0.6, 0.6]
            pred_box = [0.35, 0.35, 0.65, 0.65]  # Slight offset
            pred = f"<answer>[{','.join(map(str, pred_box))}]</answer>"
            gt = f"[{','.join(map(str, gt_box))}]"
            
        elif case_type == 'poor':
            # Poor overlap (IoU < 0.5)
            gt_box = [0.2, 0.2, 0.4, 0.4]
            pred_box = [0.5, 0.5, 0.7, 0.7]  # Different location
            pred = f"<answer>[{','.join(map(str, pred_box))}]</answer>"
            gt = f"[{','.join(map(str, gt_box))}]"
            
        elif case_type == 'hallucination':
            # False positive
            pred_box = [np.random.rand()*0.5, np.random.rand()*0.5,
                       np.random.rand()*0.5+0.5, np.random.rand()*0.5+0.5]
            pred = f"<answer>[{','.join(map(lambda x: f'{x:.3f}', pred_box))}]</answer>"
            gt = ""
            
        elif case_type == 'missed':
            # False negative
            pred = "<answer></answer>"
            gt_box = [np.random.rand()*0.5, np.random.rand()*0.5,
                     np.random.rand()*0.5+0.5, np.random.rand()*0.5+0.5]
            gt = f"[{','.join(map(lambda x: f'{x:.3f}', gt_box))}]"
            
        else:  # true_negative
            # Correct negative
            pred = "<answer></answer>"
            gt = ""
        
        predictions.append(pred)
        ground_truths.append(gt)
    
    # Analyze distribution
    analysis = analyze_reward_distribution(predictions, ground_truths)
    
    # Print statistics by category
    print("\nReward Statistics by Category:")
    print("-" * 40)
    
    category_rewards = {}
    for i, cat in enumerate(categories):
        reward = analysis['detailed_metrics'][i]['reward']
        if cat not in category_rewards:
            category_rewards[cat] = []
        category_rewards[cat].append(reward)
    
    for cat in distributions.keys():
        if cat in category_rewards:
            rewards = category_rewards[cat]
            print(f"{cat.capitalize():15} - Mean: {np.mean(rewards):.3f}, "
                  f"Std: {np.std(rewards):.3f}, N: {len(rewards)}")
    
    # Overall statistics
    print("\nOverall Statistics:")
    print("-" * 40)
    print(f"Mean Reward:     {analysis['mean_reward']:.4f}")
    print(f"Std Deviation:   {analysis['std_reward']:.4f}")
    print(f"Min Reward:      {analysis['min_reward']:.4f}")
    print(f"Max Reward:      {analysis['max_reward']:.4f}")
    print(f"Median Reward:   {analysis['median_reward']:.4f}")
    
    # Plot distribution
    plot_reward_distribution(predictions, ground_truths, 
                           save_path="thesis_distribution.png")
    print("\n✓ Distribution analysis saved to thesis_distribution.png")


if __name__ == "__main__":
    # Run complete thesis analysis
    generate_thesis_ready_analysis()
    
    # Run synthetic benchmark
    run_synthetic_benchmark()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files for thesis:")
    print("  1. r1_grounding_improved.py - Enhanced reward function implementation")
    print("  2. r1_visualizations.py - Visualization module")
    print("  3. r1_analysis_demo.py - This analysis script")
    print("  4. thesis_reward_curves.png - Reward behavior curves")
    print("  5. thesis_heatmap.png - Sensitivity heatmaps")
    print("  6. thesis_edge_cases.png - Edge case visualizations")
    print("  7. thesis_distribution.png - Reward distribution analysis")
    
    print("\nAll components are ready for thesis presentation!")
    print("LaTeX formulations and analysis results can be directly included.")