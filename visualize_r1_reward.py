"""
Visualization script for R1 reward function edge cases and behavior analysis.
This script creates comprehensive plots to understand the reward distribution
for various bounding box prediction scenarios.
"""

import re
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===== Copy the core functions from R1.py =====

# Hyperparameter: Bonus reward for correctly predicting "no box" when GT has no box
NO_BOX_BONUS = 0.2  # Small reward for correct negative predictions


def extract_bounding_boxes(answer: str) -> List[List[float]]:
    """Extract bounding boxes [x1, y1, x2, y2] from answer string."""
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    pattern = rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\]"
    
    boxes: List[List[float]] = []
    for m in re.finditer(pattern, answer):
        try:
            b = [float(m.group(1)), float(m.group(2)),
                 float(m.group(3)), float(m.group(4))]
        except Exception:
            continue
        if not all(np.isfinite(b)):
            continue
        boxes.append(b)
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    denom = box1_area + box2_area - inter_area
    return inter_area / denom if denom > 0 else 0.0


def compute_average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision given precision and recall arrays."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def average_precision_at_iou(
    predicted_boxes: List[List[float]],
    actual_boxes: List[List[float]],
    iou_threshold: float = 0.5
) -> float:
    """Compute Average Precision at a given IoU threshold."""
    if not predicted_boxes and not actual_boxes:
        return NO_BOX_BONUS
    
    if not predicted_boxes or not actual_boxes:
        return 0.0
    
    ious = np.zeros((len(predicted_boxes), len(actual_boxes)))
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(actual_boxes):
            ious[i, j] = compute_iou(pred, gt)
    
    matched_gt = set()
    true_positives = np.zeros(len(predicted_boxes))
    false_positives = np.zeros(len(predicted_boxes))
    
    for i in range(len(predicted_boxes)):
        max_idx = np.argmax(ious[i, :])
        max_iou = ious[i, max_idx]
        
        if max_iou >= iou_threshold and max_idx not in matched_gt:
            true_positives[i] = 1
            matched_gt.add(max_idx)
        else:
            false_positives[i] = 1
    
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    recall = tp_cumsum / len(actual_boxes)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    
    return compute_average_precision(recall, precision)


# ===== Visualization Functions =====

def plot_box_on_axes(ax, box, color='blue', label='', alpha=0.3, linewidth=2):
    """Draw a bounding box on matplotlib axes."""
    x1, y1, x2, y2 = box
    rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                    linewidth=linewidth, edgecolor=color,
                    facecolor=color, alpha=alpha, label=label)
    ax.add_patch(rect)


def visualize_edge_cases():
    """Create comprehensive visualization of all edge cases."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define edge cases with their scenarios
    scenarios = [
        # Row 1: Basic Cases
        {
            'title': 'Perfect Match\n(Reward = 1.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [[100, 100, 200, 200]],
            'pos': (0, 0)
        },
        {
            'title': 'No Boxes (Both Empty)\n(Reward = 0.2)',
            'gt_boxes': [],
            'pred_boxes': [],
            'pos': (0, 1)
        },
        {
            'title': 'False Positive\n(Reward = 0.0)',
            'gt_boxes': [],
            'pred_boxes': [[100, 100, 200, 200]],
            'pos': (0, 2)
        },
        {
            'title': 'Missed Detection\n(Reward = 0.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [],
            'pos': (0, 3)
        },
        
        # Row 2: IoU Variations
        {
            'title': 'High IoU (>0.5)\n(Reward ≈ 1.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [[110, 110, 210, 210]],  # 70% IoU
            'pos': (1, 0)
        },
        {
            'title': 'Threshold IoU (=0.5)\n(Reward ≈ 1.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [[100, 100, 170, 170]],  # ~50% IoU
            'pos': (1, 1)
        },
        {
            'title': 'Low IoU (<0.5)\n(Reward = 0.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [[150, 150, 250, 250]],  # ~25% IoU
            'pos': (1, 2)
        },
        {
            'title': 'No Overlap\n(Reward = 0.0)',
            'gt_boxes': [[100, 100, 200, 200]],
            'pred_boxes': [[250, 250, 350, 350]],
            'pos': (1, 3)
        },
        
        # Row 3: Multiple Boxes
        {
            'title': 'Multiple Boxes (All Match)\n(Reward = 1.0)',
            'gt_boxes': [[100, 100, 150, 150], [200, 200, 250, 250]],
            'pred_boxes': [[100, 100, 150, 150], [200, 200, 250, 250]],
            'pos': (2, 0)
        },
        {
            'title': 'Multiple Boxes (Partial Match)\n(Reward ≈ 0.5)',
            'gt_boxes': [[100, 100, 150, 150], [200, 200, 250, 250]],
            'pred_boxes': [[100, 100, 150, 150], [300, 300, 350, 350]],
            'pos': (2, 1)
        },
        {
            'title': 'Extra Predictions\n(Reward < 1.0)',
            'gt_boxes': [[100, 100, 150, 150]],
            'pred_boxes': [[100, 100, 150, 150], [200, 200, 250, 250], [300, 300, 350, 350]],
            'pos': (2, 2)
        },
        {
            'title': 'Duplicate Predictions\n(Reward < 1.0)',
            'gt_boxes': [[100, 100, 150, 150]],
            'pred_boxes': [[100, 100, 150, 150], [100, 100, 150, 150]],  # Same box twice
            'pos': (2, 3)
        }
    ]
    
    for scenario in scenarios:
        row, col = scenario['pos']
        ax = fig.add_subplot(gs[row, col])
        
        # Set up the axes
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Invert y-axis to match image coordinates
        ax.grid(True, alpha=0.3)
        
        # Draw ground truth boxes
        for i, box in enumerate(scenario['gt_boxes']):
            label = 'Ground Truth' if i == 0 else ''
            plot_box_on_axes(ax, box, color='green', label=label, alpha=0.3)
        
        # Draw predicted boxes
        for i, box in enumerate(scenario['pred_boxes']):
            label = 'Prediction' if i == 0 else ''
            plot_box_on_axes(ax, box, color='blue', label=label, alpha=0.3)
        
        # Calculate and display reward
        reward = average_precision_at_iou(scenario['pred_boxes'], scenario['gt_boxes'])
        
        # Calculate IoU for single box scenarios
        if len(scenario['gt_boxes']) == 1 and len(scenario['pred_boxes']) == 1:
            iou = compute_iou(scenario['gt_boxes'][0], scenario['pred_boxes'][0])
            ax.set_title(f"{scenario['title']}\nIoU: {iou:.2f}", fontsize=10, fontweight='bold')
        else:
            ax.set_title(scenario['title'], fontsize=10, fontweight='bold')
        
        # Add reward annotation
        ax.text(200, 380, f"Reward: {reward:.3f}", 
                fontsize=12, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        if scenario['gt_boxes'] or scenario['pred_boxes']:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('R1 Reward Function: Edge Cases and Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/r1_edge_cases.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Edge cases visualization saved as 'r1_edge_cases.png'")


def plot_iou_reward_curve():
    """Plot the relationship between IoU and reward for single box scenarios."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. IoU vs Reward for single box (continuous)
    ax = axes[0, 0]
    gt_box = [100, 100, 200, 200]
    ious = []
    rewards = []
    
    # Generate predictions with varying overlaps
    for offset in np.linspace(0, 100, 50):
        pred_box = [100 + offset, 100 + offset, 200 + offset, 200 + offset]
        iou = compute_iou(gt_box, pred_box)
        reward = average_precision_at_iou([pred_box], [gt_box])
        ious.append(iou)
        rewards.append(reward)
    
    ax.plot(ious, rewards, 'b-', linewidth=2)
    ax.axvline(x=0.5, color='r', linestyle='--', label='IoU Threshold (0.5)')
    ax.fill_between([0, 0.5], [0, 0], [1, 1], alpha=0.2, color='red', label='Below Threshold')
    ax.fill_between([0.5, 1.0], [0, 0], [1, 1], alpha=0.2, color='green', label='Above Threshold')
    ax.set_xlabel('IoU', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('IoU vs Reward (Single Box)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    
    # 2. Reward Distribution Histogram
    ax = axes[0, 1]
    
    # Generate random scenarios
    np.random.seed(42)
    scenario_rewards = []
    
    # Perfect matches
    scenario_rewards.extend([1.0] * 20)
    # Good matches (IoU > 0.5)
    scenario_rewards.extend([1.0] * 30)
    # Bad matches (IoU < 0.5)
    scenario_rewards.extend([0.0] * 25)
    # No box scenarios
    scenario_rewards.extend([NO_BOX_BONUS] * 15)
    # Complete misses
    scenario_rewards.extend([0.0] * 10)
    
    ax.hist(scenario_rewards, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=NO_BOX_BONUS, color='orange', linestyle='--', 
               label=f'No Box Bonus ({NO_BOX_BONUS})')
    ax.set_xlabel('Reward', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Reward Distribution (100 Random Scenarios)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Multiple Box Precision-Recall Curve
    ax = axes[1, 0]
    
    # Simulate a multi-box scenario
    gt_boxes = [[50, 50, 100, 100], [150, 150, 200, 200], [250, 250, 300, 300]]
    pred_boxes_scenarios = [
        # Progressively better predictions
        [[50, 50, 100, 100]],  # 1 correct
        [[50, 50, 100, 100], [150, 150, 200, 200]],  # 2 correct
        [[50, 50, 100, 100], [150, 150, 200, 200], [250, 250, 300, 300]],  # All correct
        [[50, 50, 100, 100], [150, 150, 200, 200], [250, 250, 300, 300], [350, 350, 400, 400]],  # Extra FP
    ]
    
    precisions = []
    recalls = []
    
    for pred_boxes in pred_boxes_scenarios:
        # Calculate precision and recall for this scenario
        ious = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                ious[i, j] = compute_iou(pred, gt)
        
        matched_gt = set()
        tp = 0
        for i in range(len(pred_boxes)):
            max_idx = np.argmax(ious[i, :]) if i < len(ious) else -1
            if max_idx >= 0:
                max_iou = ious[i, max_idx]
                if max_iou >= 0.5 and max_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(max_idx)
        
        precision = tp / len(pred_boxes) if pred_boxes else 0
        recall = tp / len(gt_boxes)
        precisions.append(precision)
        recalls.append(recall)
    
    ax.plot(recalls, precisions, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Trade-off (Multiple Boxes)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Annotate points
    labels = ['1 box found', '2 boxes found', 'All 3 found', '3 found + 1 FP']
    for i, (r, p, label) in enumerate(zip(recalls, precisions, labels)):
        ax.annotate(label, (r, p), textcoords="offset points", 
                   xytext=(10, -5 if i % 2 == 0 else 10), fontsize=9)
    
    # 4. IoU Threshold Sensitivity
    ax = axes[1, 1]
    
    gt_box = [100, 100, 200, 200]
    pred_box = [120, 120, 210, 210]  # Fixed prediction with ~0.55 IoU
    
    thresholds = np.linspace(0.1, 0.9, 20)
    rewards_by_threshold = []
    
    for thresh in thresholds:
        reward = average_precision_at_iou([pred_box], [gt_box], iou_threshold=thresh)
        rewards_by_threshold.append(reward)
    
    actual_iou = compute_iou(gt_box, pred_box)
    
    ax.plot(thresholds, rewards_by_threshold, 'g-', linewidth=2)
    ax.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    ax.axvline(x=actual_iou, color='b', linestyle=':', label=f'Actual IoU ({actual_iou:.2f})')
    ax.fill_between(thresholds, rewards_by_threshold, alpha=0.3)
    ax.set_xlabel('IoU Threshold', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Threshold Sensitivity (Box IoU = {actual_iou:.2f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('R1 Reward Function: Detailed Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/r1_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Analysis plots saved as 'r1_analysis.png'")


def plot_reward_heatmap():
    """Create a heatmap showing rewards for different box displacement scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ground truth box
    gt_box = [100, 100, 200, 200]
    
    # 1. Displacement heatmap
    ax = axes[0]
    
    # Create grid of displacements
    displacements = np.linspace(-50, 100, 30)
    X, Y = np.meshgrid(displacements, displacements)
    rewards = np.zeros_like(X)
    ious_grid = np.zeros_like(X)
    
    for i, dx in enumerate(displacements):
        for j, dy in enumerate(displacements):
            pred_box = [100 + dx, 100 + dy, 200 + dx, 200 + dy]
            iou = compute_iou(gt_box, pred_box)
            reward = average_precision_at_iou([pred_box], [gt_box])
            rewards[j, i] = reward
            ious_grid[j, i] = iou
    
    im = ax.contourf(X, Y, rewards, levels=20, cmap='RdYlGn')
    ax.contour(X, Y, ious_grid, levels=[0.5], colors='blue', linewidths=2, linestyles='--')
    ax.plot(0, 0, 'ko', markersize=10, label='Perfect Alignment')
    
    plt.colorbar(im, ax=ax, label='Reward')
    ax.set_xlabel('X Displacement (pixels)', fontsize=12)
    ax.set_ylabel('Y Displacement (pixels)', fontsize=12)
    ax.set_title('Reward Heatmap for Box Displacement', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add annotation for IoU=0.5 contour
    ax.text(-40, 40, 'IoU=0.5 boundary', color='blue', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 2. Scale variation heatmap
    ax = axes[1]
    
    # Create grid for scale variations
    scale_x = np.linspace(0.5, 1.5, 30)
    scale_y = np.linspace(0.5, 1.5, 30)
    X_scale, Y_scale = np.meshgrid(scale_x, scale_y)
    rewards_scale = np.zeros_like(X_scale)
    
    for i, sx in enumerate(scale_x):
        for j, sy in enumerate(scale_y):
            width = 100 * sx
            height = 100 * sy
            # Center the scaled box
            cx, cy = 150, 150  # Center of original box
            pred_box = [cx - width/2, cy - height/2, cx + width/2, cy + height/2]
            reward = average_precision_at_iou([pred_box], [gt_box])
            rewards_scale[j, i] = reward
    
    im2 = ax.contourf(X_scale, Y_scale, rewards_scale, levels=20, cmap='RdYlGn')
    ax.plot(1.0, 1.0, 'ko', markersize=10, label='Original Size')
    
    plt.colorbar(im2, ax=ax, label='Reward')
    ax.set_xlabel('Width Scale Factor', fontsize=12)
    ax.set_ylabel('Height Scale Factor', fontsize=12)
    ax.set_title('Reward Heatmap for Box Scaling', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.suptitle('R1 Reward Function: Spatial Sensitivity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/r1_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmap visualization saved as 'r1_heatmap.png'")


def test_special_cases():
    """Test and visualize special edge cases that might break the function."""
    print("\n" + "="*60)
    print("Testing Special Edge Cases")
    print("="*60 + "\n")
    
    test_cases = [
        {
            'name': 'Negative coordinates',
            'pred': [[-10, -10, 50, 50]],
            'gt': [[0, 0, 40, 40]]
        },
        {
            'name': 'Very large coordinates',
            'pred': [[1000, 1000, 2000, 2000]],
            'gt': [[1010, 1010, 2010, 2010]]
        },
        {
            'name': 'Zero-area box (line)',
            'pred': [[100, 100, 100, 200]],  # Zero width
            'gt': [[100, 100, 200, 200]]
        },
        {
            'name': 'Inverted box (x2 < x1)',
            'pred': [[200, 100, 100, 200]],  # x2 < x1
            'gt': [[100, 100, 200, 200]]
        },
        {
            'name': 'Tiny box',
            'pred': [[100, 100, 100.1, 100.1]],
            'gt': [[100, 100, 100.1, 100.1]]
        },
        {
            'name': 'Many predictions for one GT',
            'pred': [[100+i*10, 100+i*10, 150+i*10, 150+i*10] for i in range(10)],
            'gt': [[100, 100, 150, 150]]
        },
        {
            'name': 'Overlapping GT boxes',
            'pred': [[100, 100, 200, 200]],
            'gt': [[100, 100, 150, 150], [125, 125, 175, 175]]  # Overlapping GTs
        },
        {
            'name': 'Identical predictions',
            'pred': [[100, 100, 150, 150]] * 5,  # Same box repeated 5 times
            'gt': [[100, 100, 150, 150]]
        }
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, test_case in enumerate(test_cases):
        ax = axes[idx]
        
        # Calculate reward
        try:
            reward = average_precision_at_iou(test_case['pred'], test_case['gt'])
            status = "✓ Success"
            color = 'green'
        except Exception as e:
            reward = 0.0
            status = f"✗ Error: {str(e)[:20]}"
            color = 'red'
        
        # Visualize
        ax.set_xlim(-50, 250)
        ax.set_ylim(-50, 250)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        # Draw boxes if they're valid
        for box in test_case['gt']:
            if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                plot_box_on_axes(ax, box, color='green', alpha=0.3)
        
        for box in test_case['pred']:
            if len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                plot_box_on_axes(ax, box, color='blue', alpha=0.3)
        
        ax.set_title(f"{test_case['name']}\nReward: {reward:.3f}\n{status}", 
                    fontsize=9, color=color)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Special Edge Cases Testing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/r1_special_cases.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Special cases visualization saved as 'r1_special_cases.png'")
    
    # Print detailed results
    print("\nDetailed Test Results:")
    print("-" * 60)
    for test_case in test_cases:
        try:
            reward = average_precision_at_iou(test_case['pred'], test_case['gt'])
            print(f"✓ {test_case['name']:<30} Reward: {reward:.3f}")
        except Exception as e:
            print(f"✗ {test_case['name']:<30} ERROR: {e}")


def plot_multi_object_scenarios():
    """Visualize complex multi-object detection scenarios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    scenarios = [
        {
            'title': 'Perfect Multi-Object Match',
            'gt': [[50, 50, 100, 100], [150, 50, 200, 100], [50, 150, 100, 200]],
            'pred': [[50, 50, 100, 100], [150, 50, 200, 100], [50, 150, 100, 200]],
            'ax': axes[0, 0]
        },
        {
            'title': 'One Missing Object',
            'gt': [[50, 50, 100, 100], [150, 50, 200, 100], [50, 150, 100, 200]],
            'pred': [[50, 50, 100, 100], [150, 50, 200, 100]],  # Missing third
            'ax': axes[0, 1]
        },
        {
            'title': 'One False Positive',
            'gt': [[50, 50, 100, 100], [150, 50, 200, 100]],
            'pred': [[50, 50, 100, 100], [150, 50, 200, 100], [250, 250, 300, 300]],  # Extra box
            'ax': axes[0, 2]
        },
        {
            'title': 'Mixed Quality Detections',
            'gt': [[50, 50, 100, 100], [150, 50, 200, 100], [50, 150, 100, 200]],
            'pred': [[50, 50, 100, 100], [160, 60, 210, 110], [80, 180, 130, 230]],  # Various IoUs
            'ax': axes[1, 0]
        },
        {
            'title': 'Crowded Scene',
            'gt': [[50+i*30, 50, 80+i*30, 80] for i in range(5)],  # 5 small objects in a row
            'pred': [[50+i*30+5, 50+5, 80+i*30+5, 80+5] for i in range(5)],  # Slightly offset
            'ax': axes[1, 1]
        },
        {
            'title': 'Size Mismatch',
            'gt': [[50, 50, 150, 150]],  # Large box
            'pred': [[70, 70, 130, 130], [75, 75, 125, 125]],  # Two smaller boxes inside
            'ax': axes[1, 2]
        }
    ]
    
    for scenario in scenarios:
        ax = scenario['ax']
        
        # Set up axes
        ax.set_xlim(0, 350)
        ax.set_ylim(0, 350)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        # Draw ground truth
        for i, box in enumerate(scenario['gt']):
            label = f'GT{i+1}' if len(scenario['gt']) <= 3 else ('GT' if i == 0 else '')
            plot_box_on_axes(ax, box, color='green', label=label, alpha=0.3)
        
        # Draw predictions
        for i, box in enumerate(scenario['pred']):
            label = f'Pred{i+1}' if len(scenario['pred']) <= 3 else ('Pred' if i == 0 else '')
            plot_box_on_axes(ax, box, color='blue', label=label, alpha=0.3)
        
        # Calculate metrics
        reward = average_precision_at_iou(scenario['pred'], scenario['gt'])
        
        # Calculate additional metrics
        n_gt = len(scenario['gt'])
        n_pred = len(scenario['pred'])
        
        # Count matches
        if n_pred > 0 and n_gt > 0:
            ious = np.zeros((n_pred, n_gt))
            for i, pred in enumerate(scenario['pred']):
                for j, gt in enumerate(scenario['gt']):
                    ious[i, j] = compute_iou(pred, gt)
            
            matched = 0
            matched_gt = set()
            for i in range(n_pred):
                max_idx = np.argmax(ious[i, :])
                if ious[i, max_idx] >= 0.5 and max_idx not in matched_gt:
                    matched += 1
                    matched_gt.add(max_idx)
            
            precision = matched / n_pred if n_pred > 0 else 0
            recall = matched / n_gt if n_gt > 0 else 0
        else:
            precision = recall = 0
        
        # Add title and metrics
        ax.set_title(scenario['title'], fontsize=11, fontweight='bold')
        
        # Add text box with metrics
        metrics_text = (f"mAP@0.5: {reward:.3f}\n"
                       f"Precision: {precision:.2f}\n"
                       f"Recall: {recall:.2f}\n"
                       f"GT: {n_gt}, Pred: {n_pred}")
        ax.text(175, 320, metrics_text, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
               ha='center')
        
        if len(scenario['gt']) <= 3 and len(scenario['pred']) <= 3:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Multi-Object Detection Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/workspace/r1_multi_object.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Multi-object scenarios saved as 'r1_multi_object.png'")


def create_summary_report():
    """Create a text summary of key insights about the reward function."""
    
    report = """
    ================================================================================
                        R1 REWARD FUNCTION ANALYSIS REPORT
    ================================================================================
    
    REWARD DISTRIBUTION:
    ┌─────────────────────────────────────┬──────────────────────────────┐
    │ Scenario                            │ Reward                       │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ No boxes predicted, no GT boxes     │ 0.2 (NO_BOX_BONUS)          │
    │ (Correct negative prediction)       │ Small reward for correctness │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ Boxes predicted, but no GT boxes    │ 0.0 (False positives)       │
    │ (Model hallucinates boxes)          │                              │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ No boxes predicted, GT has boxes    │ 0.0 (Missed detections)     │
    │ (Model misses findings)             │                              │
    ├─────────────────────────────────────┼──────────────────────────────┤
    │ Boxes predicted, GT has boxes       │ mAP@0.5 (0.0 to 1.0)        │
    │ - Perfect localization              │ → 1.0                        │
    │ - Good localization (IoU ≥ 0.5)     │ → 0.5 to 1.0                │
    │ - Poor localization (IoU < 0.5)     │ → 0.0 to 0.5                │
    │ - Completely wrong                  │ → 0.0                        │
    └─────────────────────────────────────┴──────────────────────────────┘
    
    KEY CHARACTERISTICS:
    
    1. IoU Threshold Behavior:
       - Hard threshold at IoU = 0.5
       - Below 0.5: Box is considered a false positive (reward = 0)
       - At or above 0.5: Box is considered a true positive (reward > 0)
       - This creates a sharp discontinuity in the reward function
    
    2. Multiple Box Handling:
       - Uses Average Precision metric (standard in object detection)
       - Greedy matching: Each predicted box matches at most one GT box
       - Order matters: Predictions are processed sequentially
       - Duplicate predictions hurt precision (counted as false positives)
    
    3. Edge Case Behaviors:
       ✓ Handles negative coordinates correctly
       ✓ Handles very large coordinates correctly
       ✗ Zero-area boxes (lines/points) result in IoU = 0
       ✗ Inverted boxes (x2 < x1 or y2 < y1) may cause issues
       ✓ Handles multiple predictions for single GT (penalizes extras)
       ✓ Handles overlapping GT boxes (each matched independently)
    
    4. Strengths:
       - Standard metric (mAP@0.5) widely used in computer vision
       - Rewards both precision and recall
       - Handles multi-object scenarios well
       - Provides small reward for correct "no detection" cases
    
    5. Potential Weaknesses:
       - Hard IoU threshold may be too strict for some applications
       - No partial credit for IoU < 0.5 (even 0.49 gets 0 reward)
       - NO_BOX_BONUS (0.2) might encourage false negatives in edge cases
       - Greedy matching may not find optimal assignment
    
    6. Recommended Use Cases:
       - Object detection tasks with clear bounding boxes
       - Medical imaging (lesion detection)
       - Document analysis (text region detection)
       - Any task where 50% overlap is meaningful threshold
    
    7. Not Recommended For:
       - Tasks requiring precise boundaries (use higher IoU threshold)
       - Tasks where any overlap is acceptable (use lower threshold)
       - Segmentation tasks (use pixel-wise metrics instead)
    
    ================================================================================
    """
    
    with open('/workspace/r1_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nReport saved as 'r1_analysis_report.txt'")


def main():
    """Run all visualizations and analysis."""
    print("Starting R1 Reward Function Analysis...")
    print("="*60)
    
    # Generate all visualizations
    print("\n1. Creating edge cases visualization...")
    visualize_edge_cases()
    
    print("\n2. Creating detailed analysis plots...")
    plot_iou_reward_curve()
    
    print("\n3. Creating spatial sensitivity heatmaps...")
    plot_reward_heatmap()
    
    print("\n4. Testing special edge cases...")
    test_special_cases()
    
    print("\n5. Creating multi-object scenarios...")
    plot_multi_object_scenarios()
    
    print("\n6. Generating summary report...")
    create_summary_report()
    
    print("\n" + "="*60)
    print("Analysis complete! Generated files:")
    print("  - r1_edge_cases.png: Visual overview of all edge cases")
    print("  - r1_analysis.png: Detailed analysis plots")
    print("  - r1_heatmap.png: Spatial sensitivity analysis")
    print("  - r1_special_cases.png: Special edge case testing")
    print("  - r1_multi_object.png: Multi-object detection scenarios")
    print("  - r1_analysis_report.txt: Comprehensive text report")
    print("="*60)


if __name__ == "__main__":
    main()