"""
R1 Grounding Reward Visualization Module
=========================================
Comprehensive visualization tools for analyzing and presenting the grounding 
reward function behavior for thesis presentation.

This module provides:
1. Reward curves for different scenarios
2. mAP vs IoU comparison plots
3. Reward distribution analysis
4. Edge case visualization
5. Heatmaps and 3D surfaces for parameter sensitivity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

# Import the improved reward function
from r1_grounding_improved import (
    compute_grounding_reward,
    compute_iou,
    RewardConfig,
    extract_bounding_boxes,
    compute_map_detailed,
    classify_edge_case
)

# Set style for thesis-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_reward_curves(save_path: Optional[str] = None):
    """
    Plot reward curves for different IoU overlap scenarios.
    
    Shows how reward changes with IoU for:
    - Perfect match
    - Partial overlaps
    - Multiple predictions
    - Edge cases
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define IoU range
    iou_values = np.linspace(0, 1, 100)
    
    # -------------------------------------------------------------------------
    # Plot 1: Basic IoU vs Reward (Single Box)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    rewards_map = []
    rewards_iou_only = []
    
    for iou in iou_values:
        # Create boxes with specific IoU
        gt_box = [0.3, 0.3, 0.7, 0.7]
        
        # Calculate prediction box to achieve target IoU
        # Simplified: scale the box
        scale = np.sqrt(iou) if iou > 0 else 0
        pred_box = [
            0.3 + (1-scale)*0.2,
            0.3 + (1-scale)*0.2,
            0.7 - (1-scale)*0.2,
            0.7 - (1-scale)*0.2
        ]
        
        # Compute rewards
        pred_str = f"<answer>[{','.join(map(str, pred_box))}]</answer>"
        gt_str = f"[{','.join(map(str, gt_box))}]"
        
        config = RewardConfig(iou_threshold=0.5)
        reward = compute_grounding_reward(pred_str, gt_str, config)
        rewards_map.append(reward)
        
        # Pure IoU reward for comparison
        rewards_iou_only.append(iou)
    
    ax1.plot(iou_values, rewards_map, 'b-', linewidth=2, label='mAP@0.5 Reward')
    ax1.plot(iou_values, rewards_iou_only, 'r--', linewidth=2, alpha=0.7, label='Pure IoU Reward')
    ax1.axvline(x=0.5, color='g', linestyle=':', alpha=0.5, label='IoU Threshold')
    ax1.axhline(y=0.2, color='orange', linestyle=':', alpha=0.5, label='No-Box Bonus')
    
    ax1.set_xlabel('IoU', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Reward vs IoU: Single Box Prediction', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    
    # -------------------------------------------------------------------------
    # Plot 2: Multiple IoU Thresholds
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    colors = ['purple', 'blue', 'green', 'red']
    
    for thresh, color in zip(thresholds, colors):
        rewards = []
        for iou in iou_values:
            gt_box = [0.3, 0.3, 0.7, 0.7]
            scale = np.sqrt(iou) if iou > 0 else 0
            pred_box = [
                0.3 + (1-scale)*0.2,
                0.3 + (1-scale)*0.2,
                0.7 - (1-scale)*0.2,
                0.7 - (1-scale)*0.2
            ]
            
            pred_str = f"<answer>[{','.join(map(str, pred_box))}]</answer>"
            gt_str = f"[{','.join(map(str, gt_box))}]"
            
            config = RewardConfig(iou_threshold=thresh)
            reward = compute_grounding_reward(pred_str, gt_str, config)
            rewards.append(reward)
        
        ax2.plot(iou_values, rewards, color=color, linewidth=2, 
                label=f'τ = {thresh}')
        ax2.axvline(x=thresh, color=color, linestyle=':', alpha=0.3)
    
    ax2.set_xlabel('IoU', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Curves for Different IoU Thresholds', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])
    
    # -------------------------------------------------------------------------
    # Plot 3: Number of Boxes vs Reward
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    
    num_boxes_range = range(0, 11)
    scenarios = {
        'All Correct': [],
        'Half Correct': [],
        'All Wrong': [],
        'Hallucination': []
    }
    
    for n_boxes in num_boxes_range:
        # All correct predictions
        if n_boxes == 0:
            pred_str = "<answer></answer>"
            gt_str = ""
        else:
            boxes = [[i*0.08, i*0.08, i*0.08+0.05, i*0.08+0.05] for i in range(n_boxes)]
            pred_str = f"<answer>{','.join([str(b) for b in boxes])}</answer>"
            gt_str = ','.join([str(b) for b in boxes])
        
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, gt_str, config)
        scenarios['All Correct'].append(reward)
        
        # Half correct (for n > 0)
        if n_boxes > 0:
            correct_boxes = boxes[:n_boxes//2]
            wrong_boxes = [[b[0]+0.5, b[1]+0.5, b[2]+0.5, b[3]+0.5] for b in boxes[n_boxes//2:]]
            all_pred = correct_boxes + wrong_boxes
            pred_str = f"<answer>{','.join([str(b) for b in all_pred])}</answer>"
            gt_str = ','.join([str(b) for b in boxes])
            reward = compute_grounding_reward(pred_str, gt_str, config)
            scenarios['Half Correct'].append(reward)
        else:
            scenarios['Half Correct'].append(0.2)
        
        # All wrong
        if n_boxes > 0:
            pred_boxes = [[b[0]+0.5, b[1]+0.5, b[2]+0.5, b[3]+0.5] for b in boxes]
            pred_str = f"<answer>{','.join([str(b) for b in pred_boxes])}</answer>"
            gt_str = ','.join([str(b) for b in boxes])
            reward = compute_grounding_reward(pred_str, gt_str, config)
            scenarios['All Wrong'].append(reward)
        else:
            scenarios['All Wrong'].append(0.2)
        
        # Hallucination (predictions but no GT)
        if n_boxes > 0:
            pred_str = f"<answer>{','.join([str(b) for b in boxes])}</answer>"
            gt_str = ""
            reward = compute_grounding_reward(pred_str, gt_str, config)
            scenarios['Hallucination'].append(reward)
        else:
            scenarios['Hallucination'].append(0.2)
    
    for scenario, rewards in scenarios.items():
        ax3.plot(num_boxes_range, rewards, marker='o', linewidth=2, label=scenario)
    
    ax3.set_xlabel('Number of Boxes', fontsize=12)
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Reward vs Number of Predicted Boxes', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(num_boxes_range)
    ax3.set_ylim([0, 1.05])
    
    # -------------------------------------------------------------------------
    # Plot 4: Edge Case Comparison
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    
    edge_cases = {
        'True\nNegative': {'pred': '', 'gt': ''},
        'Perfect\nMatch': {'pred': '[0.1,0.1,0.2,0.2]', 'gt': '[0.1,0.1,0.2,0.2]'},
        'Partial\nOverlap': {'pred': '[0.12,0.12,0.22,0.22]', 'gt': '[0.1,0.1,0.2,0.2]'},
        'Hallucination': {'pred': '[0.1,0.1,0.2,0.2]', 'gt': ''},
        'Missed\nDetection': {'pred': '', 'gt': '[0.1,0.1,0.2,0.2]'},
        'One-to-Many': {'pred': '[0.1,0.1,0.3,0.3]', 'gt': '[0.1,0.1,0.2,0.2],[0.2,0.2,0.3,0.3]'},
        'Many-to-One': {'pred': '[0.1,0.1,0.2,0.2],[0.2,0.2,0.3,0.3]', 'gt': '[0.1,0.1,0.3,0.3]'}
    }
    
    case_names = []
    rewards = []
    colors_list = []
    
    for case_name, case_data in edge_cases.items():
        pred_str = f"<answer>{case_data['pred']}</answer>" if case_data['pred'] else "<answer></answer>"
        config = RewardConfig()
        reward = compute_grounding_reward(pred_str, case_data['gt'], config)
        
        case_names.append(case_name)
        rewards.append(reward)
        
        # Color based on reward level
        if reward >= 0.8:
            colors_list.append('green')
        elif reward >= 0.5:
            colors_list.append('yellow')
        elif reward > 0:
            colors_list.append('orange')
        else:
            colors_list.append('red')
    
    bars = ax4.bar(case_names, rewards, color=colors_list, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0.2, color='blue', linestyle=':', alpha=0.5, label='No-Box Bonus')
    
    # Add value labels on bars
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{reward:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylabel('Reward', fontsize=12)
    ax4.set_title('Reward for Different Edge Cases', fontsize=14, fontweight='bold')
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend()
    
    # -------------------------------------------------------------------------
    # Plot 5: Precision-Recall Trade-off
    # -------------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, :])
    
    # Simulate different confidence thresholds
    np.random.seed(42)
    n_samples = 100
    
    # Generate synthetic predictions with confidence scores
    gt_boxes_list = []
    pred_boxes_list = []
    confidences = []
    
    for i in range(n_samples):
        # Random GT box
        gt_box = [np.random.rand()*0.5, np.random.rand()*0.5, 
                  np.random.rand()*0.5 + 0.5, np.random.rand()*0.5 + 0.5]
        gt_boxes_list.append(gt_box)
        
        # Prediction with noise
        noise = np.random.randn(4) * 0.1
        pred_box = [gt_box[j] + noise[j] for j in range(4)]
        pred_boxes_list.append(pred_box)
        
        # Confidence based on IoU
        iou = compute_iou(pred_box, gt_box)
        conf = np.clip(iou + np.random.randn() * 0.1, 0, 1)
        confidences.append(conf)
    
    # Calculate precision-recall for different confidence thresholds
    conf_thresholds = np.linspace(0, 1, 50)
    precisions = []
    recalls = []
    
    for conf_thresh in conf_thresholds:
        tp = 0
        fp = 0
        fn = 0
        
        for i in range(n_samples):
            if confidences[i] >= conf_thresh:
                iou = compute_iou(pred_boxes_list[i], gt_boxes_list[i])
                if iou >= 0.5:
                    tp += 1
                else:
                    fp += 1
            else:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    ax5.plot(recalls, precisions, 'b-', linewidth=2, label='Precision-Recall Curve')
    ax5.fill_between(recalls, 0, precisions, alpha=0.2)
    
    # Add operating points
    special_points = [(0.8, 0.9), (0.9, 0.8), (0.7, 0.95)]
    for p, r in special_points:
        idx = np.argmin(np.abs(np.array(recalls) - r))
        if idx < len(precisions):
            ax5.scatter(recalls[idx], precisions[idx], s=100, c='red', zorder=5)
    
    ax5.set_xlabel('Recall', fontsize=12)
    ax5.set_ylabel('Precision', fontsize=12)
    ax5.set_title('Precision-Recall Trade-off in mAP Calculation', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1.05])
    
    # Add diagonal reference line
    ax5.plot([0, 1], [1, 0], 'k--', alpha=0.3, linewidth=1)
    
    plt.suptitle('R1 Grounding Reward Function Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_heatmap(save_path: Optional[str] = None):
    """
    Create heatmap showing reward as function of IoU and number of predictions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # -------------------------------------------------------------------------
    # Heatmap 1: IoU vs Number of Predictions
    # -------------------------------------------------------------------------
    iou_range = np.linspace(0, 1, 20)
    num_pred_range = range(1, 11)
    
    reward_matrix = np.zeros((len(num_pred_range), len(iou_range)))
    
    for i, n_pred in enumerate(num_pred_range):
        for j, target_iou in enumerate(iou_range):
            # Create scenario with n_pred predictions at target IoU
            gt_box = [0.3, 0.3, 0.7, 0.7]
            
            # Generate predictions with target IoU
            pred_boxes = []
            for k in range(n_pred):
                scale = np.sqrt(target_iou) if target_iou > 0 else 0
                offset = k * 0.01  # Small offset for multiple boxes
                pred_box = [
                    0.3 + (1-scale)*0.2 + offset,
                    0.3 + (1-scale)*0.2 + offset,
                    0.7 - (1-scale)*0.2 + offset,
                    0.7 - (1-scale)*0.2 + offset
                ]
                pred_boxes.append(pred_box)
            
            pred_str = f"<answer>{','.join([str(b) for b in pred_boxes])}</answer>"
            gt_str = f"[{','.join(map(str, gt_box))}]"
            
            config = RewardConfig()
            reward = compute_grounding_reward(pred_str, gt_str, config)
            reward_matrix[i, j] = reward
    
    im1 = axes[0].imshow(reward_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(range(0, len(iou_range), 4))
    axes[0].set_xticklabels([f'{iou_range[i]:.1f}' for i in range(0, len(iou_range), 4)])
    axes[0].set_yticks(range(len(num_pred_range)))
    axes[0].set_yticklabels(num_pred_range)
    axes[0].set_xlabel('IoU with Ground Truth', fontsize=12)
    axes[0].set_ylabel('Number of Predictions', fontsize=12)
    axes[0].set_title('Reward Heatmap: IoU vs Number of Predictions', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Reward', fontsize=10)
    
    # Add IoU threshold line
    threshold_idx = np.argmin(np.abs(iou_range - 0.5))
    axes[0].axvline(x=threshold_idx, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    axes[0].text(threshold_idx + 0.5, len(num_pred_range) - 0.5, 'τ=0.5', 
                color='blue', fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Heatmap 2: Precision vs Recall Trade-off
    # -------------------------------------------------------------------------
    precision_range = np.linspace(0, 1, 20)
    recall_range = np.linspace(0, 1, 20)
    
    f1_matrix = np.zeros((len(recall_range), len(precision_range)))
    
    for i, recall in enumerate(recall_range):
        for j, precision in enumerate(precision_range):
            # Calculate F1 score
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            f1_matrix[i, j] = f1
    
    im2 = axes[1].imshow(f1_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(range(0, len(precision_range), 4))
    axes[1].set_xticklabels([f'{precision_range[i]:.1f}' for i in range(0, len(precision_range), 4)])
    axes[1].set_yticks(range(0, len(recall_range), 4))
    axes[1].set_yticklabels([f'{recall_range[i]:.1f}' for i in range(0, len(recall_range), 4)])
    axes[1].set_xlabel('Precision', fontsize=12)
    axes[1].set_ylabel('Recall', fontsize=12)
    axes[1].set_title('F1 Score Heatmap: Precision vs Recall', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('F1 Score', fontsize=10)
    
    # Add contour lines
    CS = axes[1].contour(f1_matrix, levels=[0.5, 0.7, 0.9], colors='white', 
                        linewidths=1, alpha=0.7)
    axes[1].clabel(CS, inline=True, fontsize=8, fmt='%0.1f')
    
    plt.suptitle('Reward Function Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d_surface(save_path: Optional[str] = None):
    """
    Create 3D surface plot showing reward as function of IoU and confidence.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create mesh grid
    iou_range = np.linspace(0, 1, 30)
    conf_range = np.linspace(0, 1, 30)
    IoU, Conf = np.meshgrid(iou_range, conf_range)
    
    # Calculate rewards
    Rewards = np.zeros_like(IoU)
    
    for i in range(len(iou_range)):
        for j in range(len(conf_range)):
            iou = IoU[j, i]
            conf = Conf[j, i]
            
            # Simulate reward calculation with confidence weighting
            if iou >= 0.5:
                # Above threshold: reward influenced by confidence
                reward = min(1.0, iou * (0.5 + 0.5 * conf))
            else:
                # Below threshold: reduced reward
                reward = iou * 0.5 * conf
            
            Rewards[j, i] = reward
    
    # Create 3D surface plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(IoU, Conf, Rewards, cmap='viridis', 
                          edgecolor='none', alpha=0.8)
    
    # Add contour projections
    ax.contour(IoU, Conf, Rewards, zdir='z', offset=0, cmap='viridis', alpha=0.3)
    ax.contour(IoU, Conf, Rewards, zdir='x', offset=0, cmap='viridis', alpha=0.3)
    ax.contour(IoU, Conf, Rewards, zdir='y', offset=1, cmap='viridis', alpha=0.3)
    
    # Labels
    ax.set_xlabel('IoU', fontsize=12, labelpad=10)
    ax.set_ylabel('Confidence', fontsize=12, labelpad=10)
    ax.set_zlabel('Reward', fontsize=12, labelpad=10)
    ax.set_title('3D Reward Surface: IoU × Confidence → Reward', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Reward Value', fontsize=10)
    
    # Set viewing angle
    ax.view_init(elev=25, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_edge_case_examples(save_path: Optional[str] = None):
    """
    Visualize specific edge case examples with bounding boxes.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Define edge case scenarios with visual examples
    scenarios = [
        {
            'title': 'True Negative\n(Correct)',
            'pred_boxes': [],
            'gt_boxes': [],
            'reward': 0.2,
            'color_pred': 'green',
            'color_gt': 'blue'
        },
        {
            'title': 'Perfect Match\n(IoU = 1.0)',
            'pred_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'gt_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'reward': 1.0,
            'color_pred': 'green',
            'color_gt': 'blue'
        },
        {
            'title': 'Good Match\n(IoU > 0.5)',
            'pred_boxes': [[0.25, 0.35, 0.75, 0.85]],
            'gt_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'reward': 0.8,
            'color_pred': 'yellow',
            'color_gt': 'blue'
        },
        {
            'title': 'Poor Match\n(IoU < 0.5)',
            'pred_boxes': [[0.4, 0.5, 0.9, 0.95]],
            'gt_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'reward': 0.0,
            'color_pred': 'red',
            'color_gt': 'blue'
        },
        {
            'title': 'Hallucination\n(False Positive)',
            'pred_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'gt_boxes': [],
            'reward': 0.0,
            'color_pred': 'red',
            'color_gt': 'blue'
        },
        {
            'title': 'Missed Detection\n(False Negative)',
            'pred_boxes': [],
            'gt_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'reward': 0.0,
            'color_pred': 'green',
            'color_gt': 'red'
        },
        {
            'title': 'One-to-Many\n(Partial)',
            'pred_boxes': [[0.2, 0.3, 0.7, 0.8]],
            'gt_boxes': [[0.2, 0.3, 0.45, 0.55], [0.45, 0.55, 0.7, 0.8]],
            'reward': 0.5,
            'color_pred': 'yellow',
            'color_gt': 'blue'
        },
        {
            'title': 'Many-to-One\n(Duplicates)',
            'pred_boxes': [[0.2, 0.3, 0.45, 0.55], [0.22, 0.32, 0.47, 0.57]],
            'gt_boxes': [[0.2, 0.3, 0.45, 0.55]],
            'reward': 0.5,
            'color_pred': 'orange',
            'color_gt': 'blue'
        }
    ]
    
    for idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        # Create blank image
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # Draw ground truth boxes
        for gt_box in scenario['gt_boxes']:
            rect = patches.Rectangle(
                (gt_box[0], gt_box[1]),
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
                linewidth=2,
                edgecolor=scenario['color_gt'],
                facecolor='none',
                linestyle='--',
                label='Ground Truth' if gt_box == scenario['gt_boxes'][0] else ''
            )
            ax.add_patch(rect)
        
        # Draw predicted boxes
        for pred_box in scenario['pred_boxes']:
            rect = patches.Rectangle(
                (pred_box[0], pred_box[1]),
                pred_box[2] - pred_box[0],
                pred_box[3] - pred_box[1],
                linewidth=2,
                edgecolor=scenario['color_pred'],
                facecolor=scenario['color_pred'],
                alpha=0.3,
                label='Prediction' if pred_box == scenario['pred_boxes'][0] else ''
            )
            ax.add_patch(rect)
        
        # Add title and reward
        ax.set_title(scenario['title'], fontsize=11, fontweight='bold')
        ax.text(0.5, 0.05, f"Reward: {scenario['reward']:.2f}", 
               ha='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)
    
    # Add legend
    handles = [
        patches.Patch(color='blue', label='Ground Truth', linestyle='--', fill=False),
        patches.Patch(color='green', label='Good Prediction', alpha=0.3),
        patches.Patch(color='yellow', label='Partial Match', alpha=0.3),
        patches.Patch(color='red', label='Poor/Wrong', alpha=0.3)
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=4, frameon=True, fancybox=True, shadow=True)
    
    plt.suptitle('Edge Case Visual Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_reward_distribution(
    predictions: List[str],
    ground_truths: List[str],
    save_path: Optional[str] = None
):
    """
    Plot distribution of rewards across a dataset.
    """
    from r1_grounding_improved import analyze_reward_distribution
    
    # Analyze rewards
    analysis = analyze_reward_distribution(predictions, ground_truths)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Histogram of rewards
    ax1 = axes[0, 0]
    rewards = [m['reward'] for m in analysis['detailed_metrics']]
    ax1.hist(rewards, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(analysis['mean_reward'], color='red', linestyle='--', 
               label=f"Mean: {analysis['mean_reward']:.3f}")
    ax1.axvline(analysis['median_reward'], color='green', linestyle='--',
               label=f"Median: {analysis['median_reward']:.3f}")
    ax1.set_xlabel('Reward', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by edge case
    ax2 = axes[0, 1]
    edge_case_data = []
    edge_case_labels = []
    
    for case, stats in analysis['edge_case_distribution'].items():
        if stats['count'] > 0:
            case_rewards = [m['reward'] for m in analysis['detailed_metrics'] 
                          if m['edge_case'] == case]
            edge_case_data.append(case_rewards)
            edge_case_labels.append(f"{case}\n(n={stats['count']})")
    
    if edge_case_data:
        bp = ax2.boxplot(edge_case_data, labels=edge_case_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Reward Distribution by Edge Case', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: CDF
    ax3 = axes[1, 0]
    sorted_rewards = np.sort(rewards)
    cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    ax3.plot(sorted_rewards, cdf, linewidth=2, color='blue')
    ax3.fill_between(sorted_rewards, 0, cdf, alpha=0.2)
    ax3.set_xlabel('Reward', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Dataset Statistics:
    ─────────────────────
    Total Samples: {analysis['total_samples']}
    
    Reward Statistics:
    • Mean: {analysis['mean_reward']:.4f}
    • Std Dev: {analysis['std_reward']:.4f}
    • Min: {analysis['min_reward']:.4f}
    • Max: {analysis['max_reward']:.4f}
    
    Percentiles:
    • 25th: {analysis['percentiles']['25']:.4f}
    • 50th (Median): {analysis['percentiles']['50']:.4f}
    • 75th: {analysis['percentiles']['75']:.4f}
    • 90th: {analysis['percentiles']['90']:.4f}
    • 95th: {analysis['percentiles']['95']:.4f}
    
    Edge Case Distribution:
    """
    
    for case, stats in analysis['edge_case_distribution'].items():
        if stats['count'] > 0:
            stats_text += f"\n• {case}: {stats['count']} ({stats['count']/analysis['total_samples']*100:.1f}%)"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Reward Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return analysis


# ============================================================================
# Main Demonstration
# ============================================================================

if __name__ == "__main__":
    print("Generating R1 Grounding Reward Visualizations...")
    print("=" * 60)
    
    # Generate all visualizations
    print("\n1. Creating reward curves visualization...")
    visualize_reward_curves(save_path="r1_reward_curves.png")
    
    print("\n2. Creating heatmap visualization...")
    visualize_heatmap(save_path="r1_heatmap.png")
    
    print("\n3. Creating 3D surface plot...")
    visualize_3d_surface(save_path="r1_3d_surface.png")
    
    print("\n4. Creating edge case examples...")
    visualize_edge_case_examples(save_path="r1_edge_cases.png")
    
    # Generate sample data for distribution analysis
    print("\n5. Analyzing reward distribution on synthetic data...")
    np.random.seed(42)
    n_samples = 200
    
    # Create diverse test scenarios
    test_predictions = []
    test_ground_truths = []
    
    for i in range(n_samples):
        scenario = np.random.choice(['correct', 'partial', 'wrong', 'hallucination', 'missed'], 
                                   p=[0.3, 0.3, 0.15, 0.15, 0.1])
        
        if scenario == 'correct':
            box = f"[{np.random.rand():.2f},{np.random.rand():.2f},{np.random.rand():.2f},{np.random.rand():.2f}]"
            test_predictions.append(f"<answer>{box}</answer>")
            test_ground_truths.append(box)
        elif scenario == 'partial':
            gt_box = [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]
            pred_box = [b + np.random.randn()*0.1 for b in gt_box]
            test_predictions.append(f"<answer>[{','.join(map(str, pred_box))}]</answer>")
            test_ground_truths.append(f"[{','.join(map(str, gt_box))}]")
        elif scenario == 'wrong':
            test_predictions.append(f"<answer>[0.1,0.1,0.2,0.2]</answer>")
            test_ground_truths.append("[0.7,0.7,0.9,0.9]")
        elif scenario == 'hallucination':
            test_predictions.append(f"<answer>[0.1,0.1,0.2,0.2]</answer>")
            test_ground_truths.append("")
        else:  # missed
            test_predictions.append("<answer></answer>")
            test_ground_truths.append("[0.5,0.5,0.7,0.7]")
    
    analysis = plot_reward_distribution(test_predictions, test_ground_truths, 
                                       save_path="r1_distribution.png")
    
    print("\n" + "="*60)
    print("All visualizations completed successfully!")
    print("Generated files:")
    print("  - r1_reward_curves.png")
    print("  - r1_heatmap.png")
    print("  - r1_3d_surface.png")
    print("  - r1_edge_cases.png")
    print("  - r1_distribution.png")