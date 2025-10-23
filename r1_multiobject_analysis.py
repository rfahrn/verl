"""
R1 Multi-Object Analysis - Deep Dive into Multi-Box Scenarios
================================================================
Comprehensive analysis of R1 reward function behavior with multiple bounding boxes,
showing reward surfaces, heatmaps, and special cases for thesis presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
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


def create_multi_object_heatmap(save_path: str = "r1_heatmap.png"):
    """Create heatmap showing reward for different numbers of predictions vs GT boxes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Perfect matches heatmap
    ax = axes[0, 0]
    max_boxes = 10
    reward_matrix = np.zeros((max_boxes + 1, max_boxes + 1))
    
    for n_gt in range(max_boxes + 1):
        for n_pred in range(max_boxes + 1):
            if n_gt == 0 and n_pred == 0:
                # True negative
                reward_matrix[n_gt, n_pred] = 0.2
            elif n_gt == 0 and n_pred > 0:
                # Hallucination
                reward_matrix[n_gt, n_pred] = 0.0
            elif n_gt > 0 and n_pred == 0:
                # Missed detection
                reward_matrix[n_gt, n_pred] = 0.0
            else:
                # Calculate based on perfect matches up to min(n_pred, n_gt)
                n_matches = min(n_pred, n_gt)
                precision = n_matches / n_pred
                recall = n_matches / n_gt
                # Simplified mAP approximation
                reward_matrix[n_gt, n_pred] = precision * recall
    
    im = sns.heatmap(reward_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                     vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Reward'})
    ax.set_xlabel('Number of Predictions')
    ax.set_ylabel('Number of GT Boxes')
    ax.set_title('Reward Heatmap: Perfect Matches')
    ax.invert_yaxis()
    
    # 2. Partial matches heatmap (50% IoU overlap)
    ax = axes[0, 1]
    reward_matrix_partial = np.zeros((max_boxes + 1, max_boxes + 1))
    
    for n_gt in range(max_boxes + 1):
        for n_pred in range(max_boxes + 1):
            if n_gt == 0 and n_pred == 0:
                reward_matrix_partial[n_gt, n_pred] = 0.2
            elif n_gt == 0 or n_pred == 0:
                reward_matrix_partial[n_gt, n_pred] = 0.0
            else:
                # Assume 70% of predictions match with IoU > 0.5
                match_rate = 0.7
                n_matches = int(min(n_pred, n_gt) * match_rate)
                if n_matches > 0:
                    precision = n_matches / n_pred
                    recall = n_matches / n_gt
                    reward_matrix_partial[n_gt, n_pred] = precision * recall
                else:
                    reward_matrix_partial[n_gt, n_pred] = 0.0
    
    sns.heatmap(reward_matrix_partial, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Reward'})
    ax.set_xlabel('Number of Predictions')
    ax.set_ylabel('Number of GT Boxes')
    ax.set_title('Reward Heatmap: 70% Match Rate')
    ax.invert_yaxis()
    
    # 3. F1-score equivalent visualization
    ax = axes[1, 0]
    f1_matrix = np.zeros((max_boxes + 1, max_boxes + 1))
    
    for n_gt in range(max_boxes + 1):
        for n_pred in range(max_boxes + 1):
            if n_gt == 0 and n_pred == 0:
                f1_matrix[n_gt, n_pred] = 0.2
            elif n_gt == 0 or n_pred == 0:
                f1_matrix[n_gt, n_pred] = 0.0
            else:
                n_matches = min(n_pred, n_gt)
                precision = n_matches / n_pred
                recall = n_matches / n_gt
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_matrix[n_gt, n_pred] = f1
    
    sns.heatmap(f1_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'F1-like Score'})
    ax.set_xlabel('Number of Predictions')
    ax.set_ylabel('Number of GT Boxes')
    ax.set_title('F1-Score Equivalent (Perfect Matches)')
    ax.invert_yaxis()
    
    # 4. Difference from optimal
    ax = axes[1, 1]
    optimal_diff = np.zeros((max_boxes + 1, max_boxes + 1))
    
    for n_gt in range(max_boxes + 1):
        for n_pred in range(max_boxes + 1):
            if n_gt == 0:
                optimal_reward = 0.2 if n_pred == 0 else 0.0
            else:
                optimal_reward = 1.0 if n_pred == n_gt else 0.0
            
            actual_reward = reward_matrix[n_gt, n_pred]
            optimal_diff[n_gt, n_pred] = optimal_reward - actual_reward
    
    sns.heatmap(optimal_diff, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': 'Difference from Optimal'})
    ax.set_xlabel('Number of Predictions')
    ax.set_ylabel('Number of GT Boxes')
    ax.set_title('Difference from Optimal Reward')
    ax.invert_yaxis()
    
    plt.suptitle('R1 Multi-Object Reward Analysis: Heatmaps', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved multi-object heatmap to {save_path}")
    return fig


def analyze_special_cases(save_path: str = "r1_special_cases.png"):
    """Analyze special multi-object cases with detailed visualizations."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Special case scenarios
    special_cases = {
        "Dense Scene (5 objects)": {
            "gt": [[i*0.18, 0.2, i*0.18+0.15, 0.4] for i in range(5)],
            "variations": [
                ("Perfect", [[i*0.18, 0.2, i*0.18+0.15, 0.4] for i in range(5)]),
                ("3/5 detected", [[i*0.18, 0.2, i*0.18+0.15, 0.4] for i in [0, 2, 4]]),
                ("2 FP added", [[i*0.18, 0.2, i*0.18+0.15, 0.4] for i in range(5)] + 
                               [[0.85, 0.1, 0.95, 0.2], [0.85, 0.8, 0.95, 0.9]]),
            ]
        },
        "Overlapping Objects": {
            "gt": [[0.2, 0.3, 0.5, 0.6], [0.4, 0.4, 0.7, 0.7]],  # Overlapping boxes
            "variations": [
                ("Both detected", [[0.2, 0.3, 0.5, 0.6], [0.4, 0.4, 0.7, 0.7]]),
                ("Only larger", [[0.4, 0.4, 0.7, 0.7]]),
                ("Merged box", [[0.2, 0.3, 0.7, 0.7]]),
            ]
        },
        "Clustered vs Spread": {
            "gt": [[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.25, 0.25],  # Cluster
                   [0.7, 0.7, 0.8, 0.8], [0.75, 0.75, 0.85, 0.85]],  # Another cluster
            "variations": [
                ("All detected", [[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.25, 0.25],
                                 [0.7, 0.7, 0.8, 0.8], [0.75, 0.75, 0.85, 0.85]]),
                ("One cluster", [[0.1, 0.1, 0.2, 0.2], [0.15, 0.15, 0.25, 0.25]]),
                ("One per cluster", [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.8, 0.8]]),
            ]
        }
    }
    
    case_idx = 0
    for case_name, case_data in special_cases.items():
        gt_boxes = case_data["gt"]
        
        for var_idx, (var_name, pred_boxes) in enumerate(case_data["variations"]):
            ax = fig.add_subplot(gs[case_idx, var_idx])
            
            # Calculate reward
            pred_str = format_predictions(pred_boxes)
            gt_str = format_ground_truth(gt_boxes)
            config = RewardConfig()
            metrics = compute_grounding_reward(pred_str, gt_str, config, return_details=True)
            
            # Plot boxes
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.set_aspect('equal')
            
            # Ground truth in solid green
            for box in gt_boxes:
                rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               linewidth=2, edgecolor='green', facecolor='none')
                ax.add_patch(rect)
            
            # Predictions in dashed blue
            for box in pred_boxes:
                rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                               linewidth=2, edgecolor='blue', facecolor='none',
                               linestyle='--')
                ax.add_patch(rect)
            
            # Title and metrics
            ax.set_title(f"{case_name}\n{var_name}", fontsize=10)
            metrics_text = (f"R={metrics['reward']:.3f}\n"
                          f"P={metrics['precision']:.2f} R={metrics['recall']:.2f}\n"
                          f"TP:{metrics['tp']} FP:{metrics['fp']} FN:{metrics['fn']}")
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, alpha=0.2)
            
        case_idx += 1
    
    plt.suptitle('R1 Special Multi-Object Cases Analysis', fontsize=16)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved special cases analysis to {save_path}")
    return fig


def analyze_grounding_challenges(save_path: str = "r1_grounding_challenges.png"):
    """Analyze challenging grounding scenarios and edge cases."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    challenges = [
        {
            "name": "Small Objects",
            "gt": [[0.45, 0.45, 0.48, 0.48], [0.52, 0.52, 0.55, 0.55], 
                   [0.60, 0.60, 0.63, 0.63]],
            "pred": [[0.44, 0.44, 0.49, 0.49], [0.51, 0.51, 0.56, 0.56]]  # Missed one
        },
        {
            "name": "Large Object",
            "gt": [[0.1, 0.1, 0.9, 0.9]],
            "pred": [[0.15, 0.15, 0.85, 0.85]]  # Slightly smaller
        },
        {
            "name": "Aspect Ratio Variation",
            "gt": [[0.2, 0.4, 0.8, 0.45], [0.4, 0.2, 0.45, 0.8]],  # Thin boxes
            "pred": [[0.2, 0.38, 0.8, 0.47], [0.38, 0.2, 0.47, 0.8]]  # Slightly off
        },
        {
            "name": "Duplicate Predictions",
            "gt": [[0.3, 0.3, 0.5, 0.5]],
            "pred": [[0.3, 0.3, 0.5, 0.5], [0.31, 0.31, 0.49, 0.49]]  # Near duplicates
        },
        {
            "name": "Fragmented Detection",
            "gt": [[0.2, 0.3, 0.6, 0.7]],
            "pred": [[0.2, 0.3, 0.4, 0.5], [0.4, 0.5, 0.6, 0.7]]  # Split into two
        },
        {
            "name": "Boundary Cases",
            "gt": [[0.0, 0.0, 0.1, 0.1], [0.9, 0.9, 1.0, 1.0]],  # At edges
            "pred": [[0.0, 0.0, 0.12, 0.12], [0.88, 0.88, 1.0, 1.0]]  # Slightly off
        },
        {
            "name": "Nested Boxes",
            "gt": [[0.2, 0.2, 0.8, 0.8], [0.4, 0.4, 0.6, 0.6]],  # One inside another
            "pred": [[0.2, 0.2, 0.8, 0.8]]  # Only outer detected
        },
        {
            "name": "Grid Pattern",
            "gt": [[i*0.3, j*0.3, i*0.3+0.2, j*0.3+0.2] 
                   for i in range(3) for j in range(3)],  # 3x3 grid
            "pred": [[i*0.3, j*0.3, i*0.3+0.2, j*0.3+0.2] 
                    for i in range(3) for j in range(3) if (i+j) % 2 == 0]  # Checkerboard
        },
        {
            "name": "IoU Boundary",
            "gt": [[0.3, 0.3, 0.6, 0.6]],
            "pred": [[0.445, 0.445, 0.745, 0.745]]  # Exactly IoU ≈ 0.5
        }
    ]
    
    for idx, challenge in enumerate(challenges):
        ax = axes[idx // 3, idx % 3]
        
        # Calculate detailed metrics
        pred_str = format_predictions(challenge["pred"])
        gt_str = format_ground_truth(challenge["gt"])
        config = RewardConfig()
        metrics = compute_grounding_reward(pred_str, gt_str, config, return_details=True)
        
        # Visualize
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        
        # Plot GT boxes
        for box in challenge["gt"]:
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor='green', facecolor='green',
                           alpha=0.3)
            ax.add_patch(rect)
        
        # Plot prediction boxes
        for box in challenge["pred"]:
            rect = Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
                           linewidth=2, edgecolor='blue', facecolor='none',
                           linestyle='--')
            ax.add_patch(rect)
        
        # Calculate IoU for each match if available
        iou_text = ""
        if metrics.get('matches'):
            for pred_idx, gt_idx, iou in metrics['matches']:
                iou_text += f"IoU{pred_idx+1}: {iou:.2f}\n"
        
        # Title and annotation
        ax.set_title(challenge["name"], fontsize=11, fontweight='bold')
        
        # Metrics box
        info_text = (f"Reward: {metrics['reward']:.3f}\n"
                    f"mAP: {metrics['mAP']:.3f}\n"
                    f"P: {metrics['precision']:.2f} R: {metrics['recall']:.2f}\n"
                    f"TP:{metrics['tp']} FP:{metrics['fp']} FN:{metrics['fn']}\n"
                    f"{iou_text}")
        
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=7, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.2)
        
        # Color code background based on reward
        if metrics['reward'] >= 0.8:
            bg_color = (0.9, 1.0, 0.9, 0.2)
        elif metrics['reward'] >= 0.5:
            bg_color = (1.0, 1.0, 0.9, 0.2)
        elif metrics['reward'] > 0:
            bg_color = (1.0, 0.95, 0.9, 0.2)
        else:
            bg_color = (1.0, 0.9, 0.9, 0.2)
        
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=bg_color, zorder=-1))
    
    plt.suptitle('R1 Grounding Challenges: Edge Cases & Difficult Scenarios', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved grounding challenges to {save_path}")
    return fig


def plot_reward_surface_3d(save_path: str = "r1_surface_3d.png"):
    """Create 3D surface plot of reward function."""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create meshgrid for precision and recall
    precision = np.linspace(0, 1, 50)
    recall = np.linspace(0, 1, 50)
    P, R = np.meshgrid(precision, recall)
    
    # Calculate reward surface (simplified mAP approximation)
    Z = np.zeros_like(P)
    for i in range(len(precision)):
        for j in range(len(recall)):
            if P[i,j] == 0 or R[i,j] == 0:
                Z[i,j] = 0
            else:
                # F1-based approximation for mAP
                Z[i,j] = 2 * (P[i,j] * R[i,j]) / (P[i,j] + R[i,j])
    
    # 3D surface plot
    ax = fig.add_subplot(121, projection='3d')
    surf = ax.plot_surface(P, R, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    ax.set_zlabel('Reward (mAP approximation)')
    ax.set_title('R1 Reward Surface (Precision-Recall Space)')
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(P, R, Z, levels=20, cmap='viridis')
    ax2.contour(P, R, Z, levels=[0.2, 0.4, 0.5, 0.6, 0.8], colors='white', 
                linewidths=0.5, linestyles='--', alpha=0.5)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('R1 Reward Contours')
    fig.colorbar(contour, ax=ax2)
    
    # Add optimal line (P=R)
    ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.5, label='P=R line')
    ax2.legend()
    
    plt.suptitle('R1 Reward Function: 3D Surface Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D surface plot to {save_path}")
    return fig


def format_predictions(boxes: List) -> str:
    """Format predictions as model output string."""
    if not boxes:
        return "<answer></answer>"
    box_strs = [f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]" for b in boxes]
    return f"<answer>{', '.join(box_strs)}</answer>"


def format_ground_truth(boxes: List) -> str:
    """Format ground truth boxes as string."""
    if not boxes:
        return ""
    box_strs = [f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]" for b in boxes]
    return ', '.join(box_strs)


def main():
    """Run multi-object analysis for R1 reward function."""
    
    print("R1 Multi-Object Analysis")
    print("=" * 60)
    
    print("\n1. Creating multi-object heatmap...")
    create_multi_object_heatmap("r1_multi_object_heatmap.png")
    
    print("\n2. Analyzing special cases...")
    analyze_special_cases("r1_special_cases.png")
    
    print("\n3. Analyzing grounding challenges...")
    analyze_grounding_challenges("r1_grounding_challenges.png")
    
    print("\n4. Creating 3D reward surface...")
    plot_reward_surface_3d("r1_reward_surface_3d.png")
    
    print("\n" + "=" * 60)
    print("Multi-Object Analysis Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - r1_multi_object_heatmap.png: Heatmaps for different scenarios")
    print("  - r1_special_cases.png: Special multi-object cases")
    print("  - r1_grounding_challenges.png: Challenging grounding scenarios")
    print("  - r1_reward_surface_3d.png: 3D reward surface visualization")


if __name__ == "__main__":
    main()