"""
R1 Reward Function - Comprehensive Analysis & Visualization
=============================================================
Analysis of the R1 grounding reward function for multi-object bounding box predictions.
This script demonstrates the behavior of the reward signal across different scenarios,
edge cases, and prediction distributions for master thesis presentation.

Author: Master Thesis Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from r1_grounding_improved import (
    compute_grounding_reward, 
    compute_iou, 
    compute_map_detailed,
    RewardConfig,
    extract_bounding_boxes
)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ============================================================================
# Scenario Generation Functions
# ============================================================================

def generate_test_scenarios() -> Dict[str, Dict]:
    """Generate comprehensive test scenarios for R1 analysis."""
    
    scenarios = {
        # === EDGE CASES ===
        "true_negative": {
            "description": "Correct negative prediction (no objects)",
            "predictions": [],
            "ground_truth": [],
            "expected_behavior": "Small reward (NO_BOX_BONUS = 0.2)"
        },
        
        "hallucination_single": {
            "description": "Single hallucinated box (FP)",
            "predictions": [[0.2, 0.3, 0.4, 0.5]],
            "ground_truth": [],
            "expected_behavior": "Zero reward (false positive)"
        },
        
        "hallucination_multiple": {
            "description": "Multiple hallucinated boxes (multiple FP)",
            "predictions": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.9, 0.9]],
            "ground_truth": [],
            "expected_behavior": "Zero reward (all false positives)"
        },
        
        "missed_detection_single": {
            "description": "Single missed detection (FN)",
            "predictions": [],
            "ground_truth": [[0.3, 0.3, 0.5, 0.5]],
            "expected_behavior": "Zero reward (false negative)"
        },
        
        "missed_detection_multiple": {
            "description": "Multiple missed detections (multiple FN)",
            "predictions": [],
            "ground_truth": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], [0.8, 0.8, 0.9, 0.9]],
            "expected_behavior": "Zero reward (all false negatives)"
        },
        
        # === ONE-TO-ONE SCENARIOS ===
        "perfect_match": {
            "description": "Perfect single box match",
            "predictions": [[0.2, 0.3, 0.4, 0.5]],
            "ground_truth": [[0.2, 0.3, 0.4, 0.5]],
            "expected_behavior": "Maximum reward (1.0)"
        },
        
        "good_overlap": {
            "description": "Good overlap (IoU > 0.5)",
            "predictions": [[0.22, 0.32, 0.42, 0.52]],
            "ground_truth": [[0.2, 0.3, 0.4, 0.5]],
            "expected_behavior": "High reward (mAP based on IoU)"
        },
        
        "poor_overlap": {
            "description": "Poor overlap (IoU < 0.5)",
            "predictions": [[0.35, 0.45, 0.55, 0.65]],
            "ground_truth": [[0.2, 0.3, 0.4, 0.5]],
            "expected_behavior": "Zero reward (below threshold)"
        },
        
        # === ONE-TO-MANY SCENARIOS (1 prediction, multiple GT) ===
        "one_pred_two_gt": {
            "description": "1 prediction, 2 GT boxes",
            "predictions": [[0.2, 0.3, 0.4, 0.5]],
            "ground_truth": [[0.2, 0.3, 0.4, 0.5], [0.6, 0.6, 0.8, 0.8]],
            "expected_behavior": "Partial reward (50% recall)"
        },
        
        "one_pred_five_gt": {
            "description": "1 prediction, 5 GT boxes",
            "predictions": [[0.1, 0.1, 0.2, 0.2]],
            "ground_truth": [
                [0.1, 0.1, 0.2, 0.2],
                [0.3, 0.3, 0.4, 0.4],
                [0.5, 0.5, 0.6, 0.6],
                [0.7, 0.7, 0.8, 0.8],
                [0.85, 0.85, 0.95, 0.95]
            ],
            "expected_behavior": "Low reward (20% recall)"
        },
        
        # === MANY-TO-ONE SCENARIOS (multiple predictions, 1 GT) ===
        "two_pred_one_gt": {
            "description": "2 predictions, 1 GT box",
            "predictions": [[0.2, 0.3, 0.4, 0.5], [0.6, 0.6, 0.8, 0.8]],
            "ground_truth": [[0.2, 0.3, 0.4, 0.5]],
            "expected_behavior": "Partial reward (50% precision)"
        },
        
        "five_pred_one_gt": {
            "description": "5 predictions, 1 GT box",
            "predictions": [
                [0.1, 0.1, 0.2, 0.2],
                [0.3, 0.3, 0.4, 0.4],
                [0.5, 0.5, 0.6, 0.6],
                [0.7, 0.7, 0.8, 0.8],
                [0.85, 0.85, 0.95, 0.95]
            ],
            "ground_truth": [[0.1, 0.1, 0.2, 0.2]],
            "expected_behavior": "Low reward (20% precision)"
        },
        
        # === MANY-TO-MANY SCENARIOS ===
        "perfect_multi_match": {
            "description": "Perfect match with 3 boxes",
            "predictions": [
                [0.1, 0.1, 0.2, 0.2],
                [0.4, 0.4, 0.5, 0.5],
                [0.7, 0.7, 0.8, 0.8]
            ],
            "ground_truth": [
                [0.1, 0.1, 0.2, 0.2],
                [0.4, 0.4, 0.5, 0.5],
                [0.7, 0.7, 0.8, 0.8]
            ],
            "expected_behavior": "Maximum reward (1.0)"
        },
        
        "partial_multi_match": {
            "description": "3 predictions, 3 GT, 2 matches",
            "predictions": [
                [0.1, 0.1, 0.2, 0.2],
                [0.4, 0.4, 0.5, 0.5],
                [0.85, 0.85, 0.95, 0.95]  # Doesn't match any GT
            ],
            "ground_truth": [
                [0.1, 0.1, 0.2, 0.2],
                [0.4, 0.4, 0.5, 0.5],
                [0.7, 0.7, 0.8, 0.8]  # Not matched
            ],
            "expected_behavior": "Medium reward (~0.44, 2/3 recall, 2/3 precision)"
        },
        
        "complex_multi_match": {
            "description": "5 predictions, 4 GT boxes, mixed quality",
            "predictions": [
                [0.09, 0.09, 0.21, 0.21],  # Good match to GT1
                [0.39, 0.39, 0.51, 0.51],  # Good match to GT2
                [0.68, 0.68, 0.82, 0.82],  # Slight overlap with GT3
                [0.85, 0.85, 0.95, 0.95],  # No match (FP)
                [0.15, 0.75, 0.25, 0.85]   # Good match to GT4
            ],
            "ground_truth": [
                [0.1, 0.1, 0.2, 0.2],      # GT1
                [0.4, 0.4, 0.5, 0.5],      # GT2
                [0.7, 0.7, 0.8, 0.8],      # GT3
                [0.15, 0.75, 0.25, 0.85]   # GT4
            ],
            "expected_behavior": "High reward (~0.6-0.75, 3/4 matches)"
        },
        
        # === SCALE SCENARIOS (1 to 10 boxes) ===
        "scale_10_boxes_perfect": {
            "description": "10 GT boxes, all perfectly matched",
            "predictions": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(10)],
            "ground_truth": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(10)],
            "expected_behavior": "Maximum reward (1.0)"
        },
        
        "scale_10_boxes_half": {
            "description": "10 GT boxes, 5 matched",
            "predictions": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(5)],
            "ground_truth": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(10)],
            "expected_behavior": "Medium reward (50% recall)"
        },
        
        "scale_10_boxes_overpred": {
            "description": "5 GT boxes, 10 predictions",
            "predictions": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(10)],
            "ground_truth": [[i/10, i/10, (i+0.8)/10, (i+0.8)/10] for i in range(5)],
            "expected_behavior": "Medium reward (50% precision)"
        }
    }
    
    return scenarios


def calculate_scenario_rewards(scenarios: Dict) -> Dict:
    """Calculate rewards for all scenarios."""
    config = RewardConfig(no_box_bonus=0.2, iou_threshold=0.5)
    
    for name, scenario in scenarios.items():
        # Convert to string format for reward function
        pred_str = format_predictions(scenario['predictions'])
        gt_str = format_ground_truth(scenario['ground_truth'])
        
        # Calculate reward with details
        metrics = compute_grounding_reward(
            pred_str, 
            gt_str, 
            config, 
            return_details=True
        )
        
        scenario['reward'] = metrics['reward']
        scenario['metrics'] = metrics
        
    return scenarios


def format_predictions(boxes: List) -> str:
    """Format predictions as model output string."""
    if not boxes:
        return "<answer></answer>"
    box_strs = [f"[{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}]" for b in boxes]
    return f"<answer>{', '.join(box_strs)}</answer>"


def format_ground_truth(boxes: List) -> str:
    """Format ground truth boxes as string."""
    if not boxes:
        return ""
    box_strs = [f"[{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}]" for b in boxes]
    return ', '.join(box_strs)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_scenario_grid(scenarios: Dict, save_path: str = "r1_scenarios.png"):
    """Create a comprehensive grid visualization of different scenarios."""
    
    # Select key scenarios for visualization
    key_scenarios = [
        "true_negative", "hallucination_single", "missed_detection_single",
        "perfect_match", "good_overlap", "poor_overlap",
        "one_pred_two_gt", "two_pred_one_gt", "partial_multi_match",
        "complex_multi_match", "scale_10_boxes_perfect", "scale_10_boxes_half"
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()
    
    for idx, scenario_name in enumerate(key_scenarios):
        ax = axes[idx]
        scenario = scenarios[scenario_name]
        
        # Plot boxes
        plot_boxes_on_axis(
            ax,
            scenario['predictions'],
            scenario['ground_truth'],
            scenario['reward']
        )
        
        # Set title with reward
        ax.set_title(
            f"{scenario['description']}\nReward: {scenario['reward']:.3f}",
            fontsize=10,
            pad=10
        )
        
        # Add metrics text
        metrics = scenario['metrics']
        metrics_text = f"TP:{metrics['tp']} FP:{metrics['fp']} FN:{metrics['fn']}"
        ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes, 
                ha='center', fontsize=8, color='gray')
    
    plt.suptitle("R1 Reward Function: Scenario Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved scenario grid to {save_path}")
    return fig


def plot_boxes_on_axis(ax, predictions: List, ground_truth: List, reward: float):
    """Plot bounding boxes on a single axis."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot ground truth boxes in green
    for gt_box in ground_truth:
        rect = Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2,
            edgecolor='green',
            facecolor='none',
            label='GT' if ground_truth.index(gt_box) == 0 else ""
        )
        ax.add_patch(rect)
    
    # Plot predicted boxes in blue
    for pred_box in predictions:
        rect = Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2,
            edgecolor='blue',
            facecolor='none',
            linestyle='--',
            label='Pred' if predictions.index(pred_box) == 0 else ""
        )
        ax.add_patch(rect)
    
    # Add legend if there are boxes
    if predictions or ground_truth:
        ax.legend(loc='upper right', fontsize=8)
    
    # Color background based on reward
    if reward >= 0.8:
        bg_color = (0.9, 1.0, 0.9, 0.3)  # Light green
    elif reward >= 0.5:
        bg_color = (1.0, 1.0, 0.9, 0.3)  # Light yellow
    elif reward > 0:
        bg_color = (1.0, 0.95, 0.9, 0.3)  # Light orange
    else:
        bg_color = (1.0, 0.9, 0.9, 0.3)  # Light red
    
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_color, zorder=-1))


def plot_reward_distribution_analysis(save_path: str = "r1_distribution.png"):
    """Analyze and plot reward distribution for different IoU values and box counts."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. IoU vs Reward (single box)
    ax = axes[0, 0]
    ious = np.linspace(0, 1, 100)
    rewards_single = []
    
    for iou_val in ious:
        # Create boxes with specific IoU
        pred_box = [0.2, 0.2, 0.2 + 0.3 * iou_val, 0.2 + 0.3 * iou_val]
        gt_box = [0.2, 0.2, 0.5, 0.5]
        
        pred_str = format_predictions([pred_box])
        gt_str = format_ground_truth([gt_box])
        
        config = RewardConfig(no_box_bonus=0.2, iou_threshold=0.5)
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_single.append(reward)
    
    ax.plot(ious, rewards_single, linewidth=2)
    ax.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='IoU threshold')
    ax.set_xlabel('IoU')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs IoU (Single Box)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Number of GT boxes vs Reward (perfect matches)
    ax = axes[0, 1]
    n_boxes_range = range(0, 11)
    rewards_perfect = []
    rewards_half = []
    rewards_none = []
    
    for n in n_boxes_range:
        if n == 0:
            # True negative case
            pred_str = format_predictions([])
            gt_str = format_ground_truth([])
            config = RewardConfig(no_box_bonus=0.2)
            reward = compute_grounding_reward(pred_str, gt_str, config)
            rewards_perfect.append(reward)
            rewards_half.append(0)
            rewards_none.append(reward)
        else:
            # Generate n boxes
            gt_boxes = [[i/(n+1), 0.3, (i+0.5)/(n+1), 0.7] for i in range(n)]
            
            # Perfect match
            pred_str = format_predictions(gt_boxes)
            gt_str = format_ground_truth(gt_boxes)
            reward = compute_grounding_reward(pred_str, gt_str, config)
            rewards_perfect.append(reward)
            
            # Half match
            pred_str = format_predictions(gt_boxes[:n//2])
            reward = compute_grounding_reward(pred_str, gt_str, config)
            rewards_half.append(reward)
            
            # No match
            pred_str = format_predictions([])
            reward = compute_grounding_reward(pred_str, gt_str, config)
            rewards_none.append(reward)
    
    ax.plot(n_boxes_range, rewards_perfect, 'g-', label='Perfect match', linewidth=2)
    ax.plot(n_boxes_range, rewards_half, 'b--', label='50% match', linewidth=2)
    ax.plot(n_boxes_range, rewards_none, 'r:', label='No match', linewidth=2)
    ax.set_xlabel('Number of GT Boxes')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs Number of Boxes')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Precision-Recall trade-off
    ax = axes[0, 2]
    precisions = np.linspace(0, 1, 11)
    recalls = np.linspace(0, 1, 11)
    reward_matrix = np.zeros((len(precisions), len(recalls)))
    
    for i, prec in enumerate(precisions):
        for j, rec in enumerate(recalls):
            if prec == 0 or rec == 0:
                reward_matrix[i, j] = 0
            else:
                # Simulate scenario with given precision and recall
                n_tp = int(rec * 10)
                n_fn = 10 - n_tp
                n_fp = int(n_tp * (1/prec - 1)) if prec > 0 else 0
                
                # Approximate mAP (simplified)
                if n_tp + n_fp > 0:
                    reward_matrix[i, j] = (n_tp / (n_tp + n_fp)) * (n_tp / 10)
    
    im = ax.imshow(reward_matrix, cmap='viridis', aspect='auto', origin='lower')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Reward Surface (Precision-Recall)')
    ax.set_xticks(range(0, 11, 2))
    ax.set_xticklabels([f'{r:.1f}' for r in recalls[::2]])
    ax.set_yticks(range(0, 11, 2))
    ax.set_yticklabels([f'{p:.1f}' for p in precisions[::2]])
    plt.colorbar(im, ax=ax)
    
    # 4. False Positive Impact
    ax = axes[1, 0]
    n_fp_range = range(0, 11)
    rewards_fp = []
    
    gt_boxes = [[0.2, 0.2, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]]
    
    for n_fp in n_fp_range:
        pred_boxes = gt_boxes.copy()  # Start with perfect matches
        # Add false positives
        for i in range(n_fp):
            pred_boxes.append([0.1 + i*0.08, 0.85, 0.15 + i*0.08, 0.95])
        
        pred_str = format_predictions(pred_boxes)
        gt_str = format_ground_truth(gt_boxes)
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_fp.append(reward)
    
    ax.plot(n_fp_range, rewards_fp, 'r-', linewidth=2)
    ax.set_xlabel('Number of False Positives')
    ax.set_ylabel('Reward')
    ax.set_title('Impact of False Positives (2 GT boxes)')
    ax.grid(True, alpha=0.3)
    
    # 5. False Negative Impact
    ax = axes[1, 1]
    n_fn_range = range(0, 6)
    rewards_fn = []
    
    gt_boxes_5 = [[i*0.18, 0.3, i*0.18 + 0.15, 0.5] for i in range(5)]
    
    for n_fn in n_fn_range:
        n_matched = 5 - n_fn
        pred_boxes = gt_boxes_5[:n_matched]
        
        pred_str = format_predictions(pred_boxes)
        gt_str = format_ground_truth(gt_boxes_5)
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_fn.append(reward)
    
    ax.plot(n_fn_range, rewards_fn, 'b-', linewidth=2)
    ax.set_xlabel('Number of False Negatives')
    ax.set_ylabel('Reward')
    ax.set_title('Impact of False Negatives (5 GT boxes)')
    ax.grid(True, alpha=0.3)
    
    # 6. Reward Distribution Histogram
    ax = axes[1, 2]
    
    # Generate random scenarios
    np.random.seed(42)
    rewards_random = []
    
    for _ in range(1000):
        n_gt = np.random.randint(0, 6)
        n_pred = np.random.randint(0, 6)
        
        if n_gt == 0:
            gt_boxes = []
        else:
            gt_boxes = [[np.random.rand()*0.7, np.random.rand()*0.7, 
                        np.random.rand()*0.3+0.7, np.random.rand()*0.3+0.7] 
                       for _ in range(n_gt)]
        
        if n_pred == 0:
            pred_boxes = []
        else:
            # Mix of good and bad predictions
            pred_boxes = []
            for _ in range(n_pred):
                if np.random.rand() < 0.6 and len(gt_boxes) > 0:  # 60% chance of near-match
                    idx = np.random.randint(len(gt_boxes))
                    base_box = gt_boxes[idx]
                    noise = np.random.randn(4) * 0.05
                    pred_box = [base_box[i] + noise[i] for i in range(4)]
                else:  # Random box
                    pred_box = [np.random.rand()*0.7, np.random.rand()*0.7,
                               np.random.rand()*0.3+0.7, np.random.rand()*0.3+0.7]
                pred_boxes.append(pred_box)
        
        pred_str = format_predictions(pred_boxes)
        gt_str = format_ground_truth(gt_boxes)
        config = RewardConfig(no_box_bonus=0.2)
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_random.append(reward)
    
    ax.hist(rewards_random, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.2, color='g', linestyle='--', alpha=0.5, label='NO_BOX_BONUS')
    ax.set_xlabel('Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution (1000 Random Scenarios)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("R1 Reward Function: Distribution Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution analysis to {save_path}")
    return fig


def plot_edge_case_analysis(scenarios: Dict, save_path: str = "r1_edge_cases.png"):
    """Create detailed analysis of edge cases."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Group scenarios by edge case type
    edge_case_groups = {
        'True Negative': ['true_negative'],
        'Hallucination': ['hallucination_single', 'hallucination_multiple'],
        'Missed Detection': ['missed_detection_single', 'missed_detection_multiple'],
        'Partial Match': ['one_pred_two_gt', 'two_pred_one_gt', 'one_pred_five_gt', 'five_pred_one_gt'],
        'Multi-object': ['partial_multi_match', 'complex_multi_match'],
        'Scale': ['scale_10_boxes_perfect', 'scale_10_boxes_half', 'scale_10_boxes_overpred']
    }
    
    for idx, (group_name, scenario_names) in enumerate(edge_case_groups.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Collect rewards and labels
        rewards = []
        labels = []
        colors = []
        
        for s_name in scenario_names:
            if s_name in scenarios:
                scenario = scenarios[s_name]
                rewards.append(scenario['reward'])
                # Shorten label for display
                label = scenario['description'].split(',')[0][:20]
                labels.append(label)
                
                # Color based on reward value
                if scenario['reward'] >= 0.8:
                    colors.append('green')
                elif scenario['reward'] >= 0.5:
                    colors.append('yellow')
                elif scenario['reward'] > 0:
                    colors.append('orange')
                else:
                    colors.append('red')
        
        # Create bar plot
        bars = ax.bar(range(len(rewards)), rewards, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(rewards)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Reward')
        ax.set_title(f'{group_name} Cases')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, reward in zip(bars, rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{reward:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Add reference line for NO_BOX_BONUS
        if group_name == 'True Negative':
            ax.axhline(y=0.2, color='g', linestyle='--', alpha=0.5, label='NO_BOX_BONUS')
            ax.legend()
    
    plt.suptitle("R1 Reward Function: Edge Case Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved edge case analysis to {save_path}")
    return fig


def plot_continuity_analysis(save_path: str = "r1_continuity.png"):
    """Analyze the continuity and smoothness of the reward signal."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Continuous IoU transition
    ax = axes[0, 0]
    displacements = np.linspace(-0.3, 0.3, 100)
    rewards = []
    ious = []
    
    gt_box = [0.3, 0.3, 0.6, 0.6]
    
    for disp in displacements:
        pred_box = [0.3 + disp, 0.3 + disp, 0.6 + disp, 0.6 + disp]
        
        iou = compute_iou(pred_box, gt_box)
        ious.append(iou)
        
        pred_str = format_predictions([pred_box])
        gt_str = format_ground_truth([gt_box])
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards.append(reward)
    
    ax.plot(displacements, rewards, 'b-', label='Reward', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(displacements, ious, 'r--', label='IoU', linewidth=1, alpha=0.7)
    
    ax.set_xlabel('Box Displacement')
    ax.set_ylabel('Reward', color='b')
    ax2.set_ylabel('IoU', color='r')
    ax.set_title('Reward Continuity: Box Displacement')
    ax.grid(True, alpha=0.3)
    
    # Add threshold line
    threshold_idx = np.where(np.array(ious) >= 0.5)[0]
    if len(threshold_idx) > 0:
        ax.axvline(x=displacements[threshold_idx[0]], color='g', linestyle=':', alpha=0.5)
        ax.axvline(x=displacements[threshold_idx[-1]], color='g', linestyle=':', alpha=0.5)
    
    # 2. Scale transition (box size)
    ax = axes[0, 1]
    scales = np.linspace(0.5, 1.5, 100)
    rewards_scale = []
    
    gt_box = [0.4, 0.4, 0.6, 0.6]
    
    for scale in scales:
        center_x = (gt_box[0] + gt_box[2]) / 2
        center_y = (gt_box[1] + gt_box[3]) / 2
        half_width = (gt_box[2] - gt_box[0]) / 2 * scale
        half_height = (gt_box[3] - gt_box[1]) / 2 * scale
        
        pred_box = [
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height
        ]
        
        pred_str = format_predictions([pred_box])
        gt_str = format_ground_truth([gt_box])
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_scale.append(reward)
    
    ax.plot(scales, rewards_scale, 'g-', linewidth=2)
    ax.set_xlabel('Box Scale Factor')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Continuity: Box Scaling')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect scale')
    ax.legend()
    
    # 3. Number of predictions transition
    ax = axes[1, 0]
    n_preds_range = range(0, 11)
    rewards_npred = []
    
    gt_boxes = [[0.2, 0.2, 0.3, 0.3], [0.5, 0.5, 0.6, 0.6], [0.7, 0.7, 0.8, 0.8]]
    
    for n_pred in n_preds_range:
        if n_pred == 0:
            pred_boxes = []
        elif n_pred <= 3:
            pred_boxes = gt_boxes[:n_pred]
        else:
            # Add extra boxes (false positives)
            pred_boxes = gt_boxes.copy()
            for i in range(n_pred - 3):
                pred_boxes.append([0.85 + i*0.01, 0.1, 0.9 + i*0.01, 0.15])
        
        pred_str = format_predictions(pred_boxes)
        gt_str = format_ground_truth(gt_boxes)
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_npred.append(reward)
    
    ax.plot(n_preds_range, rewards_npred, 'purple', marker='o', linewidth=2)
    ax.set_xlabel('Number of Predictions')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs Number of Predictions (3 GT boxes)')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=3, color='r', linestyle='--', alpha=0.5, label='Optimal (3 predictions)')
    ax.legend()
    
    # 4. Gradient analysis
    ax = axes[1, 1]
    
    # Calculate numerical gradient for IoU transition
    gradients = np.gradient(rewards)
    
    ax.plot(displacements[:-1], gradients[:-1], 'b-', linewidth=2)
    ax.set_xlabel('Box Displacement')
    ax.set_ylabel('Reward Gradient')
    ax.set_title('Reward Signal Gradient (Smoothness)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Highlight discontinuities
    discontinuities = np.where(np.abs(gradients[:-1]) > 5)[0]
    if len(discontinuities) > 0:
        ax.scatter(displacements[discontinuities], gradients[discontinuities], 
                  color='red', s=50, zorder=5, label='Discontinuity')
        ax.legend()
    
    plt.suptitle("R1 Reward Function: Continuity Analysis", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved continuity analysis to {save_path}")
    return fig


def create_latex_summary(scenarios: Dict, save_path: str = "r1_analysis_report.txt"):
    """Generate LaTeX-ready summary of the analysis."""
    
    report = []
    report.append("=" * 80)
    report.append("R1 REWARD FUNCTION - COMPREHENSIVE ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Mathematical formulation
    report.append("MATHEMATICAL FORMULATION")
    report.append("-" * 40)
    report.append("LaTeX equation for thesis:")
    report.append("")
    report.append(r"\begin{equation}")
    report.append(r"\label{eq:r1_reward}")
    report.append(r"\mathcal{R}_1(\mathcal{P}, \mathcal{G}) = ")
    report.append(r"\begin{cases}")
    report.append(r"\rho_{\text{CN}} & \text{if } |\mathcal{P}| = 0 \text{ and } |\mathcal{G}| = 0 \quad \text{(Correct Negative)} \\[0.3em]")
    report.append(r"0 & \text{if } (|\mathcal{P}| > 0 \text{ and } |\mathcal{G}| = 0) \text{ or } (|\mathcal{P}| = 0 \text{ and } |\mathcal{G}| > 0) \\[0.3em]")
    report.append(r"\text{mAP}_{\tau}(\mathcal{P}, \mathcal{G}) & \text{otherwise}")
    report.append(r"\end{cases}")
    report.append(r"\end{equation}")
    report.append("")
    report.append("Where:")
    report.append("- P: Set of predicted bounding boxes")
    report.append("- G: Set of ground truth bounding boxes")
    report.append("- ρ_CN = 0.2: Correct negative bonus")
    report.append("- τ = 0.5: IoU threshold")
    report.append("- mAP: Mean Average Precision at IoU threshold τ")
    report.append("")
    
    # Scenario summary table
    report.append("SCENARIO ANALYSIS SUMMARY")
    report.append("-" * 40)
    report.append("")
    report.append("LaTeX table for thesis:")
    report.append("")
    report.append(r"\begin{table}[h]")
    report.append(r"\centering")
    report.append(r"\caption{R1 Reward Function Behavior Across Different Scenarios}")
    report.append(r"\label{tab:r1_scenarios}")
    report.append(r"\begin{tabular}{l|c|c|c|c|c|c}")
    report.append(r"\hline")
    report.append(r"Scenario & $|\mathcal{P}|$ & $|\mathcal{G}|$ & TP & FP & FN & Reward \\")
    report.append(r"\hline\hline")
    
    # Key scenarios for table
    key_scenarios_table = [
        'true_negative', 'hallucination_single', 'missed_detection_single',
        'perfect_match', 'good_overlap', 'poor_overlap',
        'one_pred_two_gt', 'partial_multi_match', 'scale_10_boxes_perfect'
    ]
    
    for s_name in key_scenarios_table:
        if s_name in scenarios:
            s = scenarios[s_name]
            m = s['metrics']
            desc = s['description'].replace('_', r'\_')[:30]
            report.append(
                f"{desc} & {m['num_predictions']} & {m['num_ground_truth']} & "
                f"{m['tp']} & {m['fp']} & {m['fn']} & {s['reward']:.3f} \\\\"
            )
    
    report.append(r"\hline")
    report.append(r"\end{tabular}")
    report.append(r"\end{table}")
    report.append("")
    
    # Properties analysis
    report.append("REWARD FUNCTION PROPERTIES")
    report.append("-" * 40)
    report.append("")
    
    # Calculate statistics
    all_rewards = [s['reward'] for s in scenarios.values()]
    edge_case_stats = {}
    
    for s in scenarios.values():
        edge_case = s['metrics']['edge_case']
        if edge_case not in edge_case_stats:
            edge_case_stats[edge_case] = []
        edge_case_stats[edge_case].append(s['reward'])
    
    report.append("1. CONTINUITY:")
    report.append("   - The reward function exhibits discontinuity at IoU = τ (0.5)")
    report.append("   - Smooth transitions within matching and non-matching regions")
    report.append("   - Gradient is well-behaved except at the threshold boundary")
    report.append("")
    
    report.append("2. RANGE AND DISTRIBUTION:")
    report.append(f"   - Reward range: [0.0, 1.0]")
    report.append(f"   - Mean reward across scenarios: {np.mean(all_rewards):.3f}")
    report.append(f"   - Std deviation: {np.std(all_rewards):.3f}")
    report.append("")
    
    report.append("3. EDGE CASE BEHAVIOR:")
    for edge_case, rewards in edge_case_stats.items():
        report.append(f"   - {edge_case}:")
        report.append(f"     Mean reward: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}")
    report.append("")
    
    report.append("4. LEARNING SIGNAL QUALITY:")
    report.append("   - Clear differentiation between correct/incorrect predictions")
    report.append("   - Proportional rewards for partial matches")
    report.append("   - Handles multi-object scenarios appropriately")
    report.append("   - Small positive reward for correct negatives encourages learning")
    report.append("")
    
    report.append("5. SCALABILITY:")
    report.append("   - Consistent behavior from 0 to 10+ boxes")
    report.append("   - Computational complexity: O(n*m) for n predictions, m ground truth")
    report.append("   - Memory efficient for typical object detection scenarios")
    report.append("")
    
    # Critical observations
    report.append("CRITICAL OBSERVATIONS")
    report.append("-" * 40)
    report.append("")
    report.append("STRENGTHS:")
    report.append("1. Simple and interpretable formulation")
    report.append("2. Aligns with standard object detection metrics (mAP@0.5)")
    report.append("3. Handles edge cases explicitly")
    report.append("4. Provides non-zero reward for correct negatives")
    report.append("5. Computationally efficient")
    report.append("")
    
    report.append("LIMITATIONS:")
    report.append("1. Hard threshold at IoU = 0.5 creates discontinuity")
    report.append("2. No partial credit for IoU < 0.5")
    report.append("3. Equal weight for all boxes (no importance weighting)")
    report.append("4. Greedy matching may not find optimal assignment")
    report.append("")
    
    report.append("RECOMMENDATIONS FOR IMPROVEMENT:")
    report.append("1. Consider soft IoU thresholding for smoother gradients")
    report.append("2. Add confidence weighting for predicted boxes")
    report.append("3. Implement Hungarian algorithm for optimal matching")
    report.append("4. Consider class-specific reward weights")
    report.append("")
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved analysis report to {save_path}")
    return report


def main():
    """Run complete R1 reward function analysis."""
    
    print("R1 Reward Function - Comprehensive Analysis")
    print("=" * 60)
    
    # Generate and calculate scenarios
    print("\n1. Generating test scenarios...")
    scenarios = generate_test_scenarios()
    scenarios = calculate_scenario_rewards(scenarios)
    print(f"   Generated {len(scenarios)} scenarios")
    
    # Create visualizations
    print("\n2. Creating visualizations...")
    
    print("   - Scenario grid visualization...")
    plot_scenario_grid(scenarios, "r1_scenarios_grid.png")
    
    print("   - Distribution analysis...")
    plot_reward_distribution_analysis("r1_distribution_analysis.png")
    
    print("   - Edge case analysis...")
    plot_edge_case_analysis(scenarios, "r1_edge_cases.png")
    
    print("   - Continuity analysis...")
    plot_continuity_analysis("r1_continuity.png")
    
    # Generate report
    print("\n3. Generating analysis report...")
    create_latex_summary(scenarios, "r1_analysis_report.txt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - r1_scenarios_grid.png: Visual grid of key scenarios")
    print("  - r1_distribution_analysis.png: Reward distribution analysis")
    print("  - r1_edge_cases.png: Edge case behavior analysis")
    print("  - r1_continuity.png: Continuity and smoothness analysis")
    print("  - r1_analysis_report.txt: Comprehensive LaTeX-ready report")
    print("\nKey Findings:")
    print("  - Reward function provides clear learning signal")
    print("  - Handles multi-object scenarios appropriately")
    print("  - Edge cases are well-defined with explicit rewards")
    print("  - Discontinuity at IoU threshold may affect gradient flow")
    print("  - Overall suitable for RL training with proper hyperparameter tuning")


if __name__ == "__main__":
    main()