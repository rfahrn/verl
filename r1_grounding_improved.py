"""
R1 Grounding Reward Function - Enhanced Version
================================================
Advanced grounding reward function based on mAP@0.5 with improved formulation,
edge case handling, and comprehensive documentation for thesis presentation.

Mathematical Formulation:
------------------------
The reward function R is defined as:

R(B̂, B) = {
    NO_BOX_BONUS,     if |B̂| = 0 and |B| = 0  (Correct negative)
    0,                if |B̂| > 0 and |B| = 0  (False positive/Hallucination)
    0,                if |B̂| = 0 and |B| > 0  (False negative/Missed detection)
    mAP(B̂, B; τ),     if |B̂| > 0 and |B| > 0  (Detection with quality assessment)
}

Where:
- B̂: Set of predicted bounding boxes
- B: Set of ground truth bounding boxes
- τ: IoU threshold (default: 0.5)
- NO_BOX_BONUS: Small reward for correct negative predictions (default: 0.2)
- mAP: Mean Average Precision at IoU threshold τ

mAP Calculation:
---------------
mAP(B̂, B; τ) = AP@τ = ∫₀¹ P(r) dr

Where P(r) is precision at recall r, computed via:
1. For each prediction, find best matching GT box by IoU
2. Match is valid if IoU ≥ τ and GT box not already matched
3. Compute precision-recall curve
4. Calculate area under curve (AUC)

Author: Enhanced for Master Thesis
Date: 2025
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RewardConfig:
    """Configuration for reward function behavior."""
    no_box_bonus: float = 0.2  # Reward for correct negative predictions
    iou_threshold: float = 0.5  # Default IoU threshold for mAP
    min_box_area: float = 1e-6  # Minimum valid box area
    max_boxes: int = 100  # Maximum number of boxes to consider
    confidence_weight: bool = False  # Whether to weight by confidence scores
    normalize_coordinates: bool = True  # Normalize coordinates to [0, 1]
    
    # Edge case penalties/rewards
    hallucination_penalty: float = 0.0  # Additional penalty for hallucinations
    missed_detection_penalty: float = 0.0  # Additional penalty for missed detections
    
    # Advanced options
    use_soft_iou: bool = False  # Use soft IoU for partial credit
    iou_power: float = 1.0  # Power to apply to IoU scores (higher = stricter)


# ============================================================================
# Core Functions
# ============================================================================

def extract_bounding_boxes(
    answer: str, 
    normalize: bool = True,
    return_confidence: bool = False
) -> Tuple[List[List[float]], Optional[List[float]]]:
    """
    Extract bounding boxes from answer string with enhanced parsing.
    
    Supports formats:
    - [x1, y1, x2, y2]
    - [x1, y1, x2, y2, confidence]
    - <box>x1,y1,x2,y2</box>
    
    Args:
        answer: String containing bounding boxes
        normalize: Whether to normalize coordinates to [0, 1]
        return_confidence: Whether to return confidence scores
        
    Returns:
        Tuple of (boxes, confidences) where confidences may be None
    """
    # Pattern for various float formats including scientific notation
    NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    
    # Try multiple patterns
    patterns = [
        # Standard [x1, y1, x2, y2] or [x1, y1, x2, y2, conf]
        rf"\[\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*(?:,\s*({NUM}))?\s*\]",
        # XML-style <box>x1,y1,x2,y2</box>
        rf"<box>\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*</box>",
        # Parentheses format (x1, y1, x2, y2)
        rf"\(\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*,\s*({NUM})\s*\)"
    ]
    
    boxes: List[List[float]] = []
    confidences: List[float] = []
    
    for pattern in patterns:
        for match in re.finditer(pattern, answer, re.IGNORECASE):
            try:
                groups = match.groups()
                box = [float(groups[i]) if groups[i] else 0.0 for i in range(4)]
                
                # Validate box
                if not all(np.isfinite(box)):
                    continue
                    
                # Ensure x2 > x1 and y2 > y1
                if box[2] <= box[0] or box[3] <= box[1]:
                    # Try to fix by swapping
                    box = [min(box[0], box[2]), min(box[1], box[3]), 
                           max(box[0], box[2]), max(box[1], box[3])]
                    
                # Normalize if requested
                if normalize and all(b <= 1000 for b in box):  # Assume pixel coords if > 1
                    # Already normalized or normalize to [0, 1]
                    if max(box) > 1:
                        # Assume image size for normalization (this is a heuristic)
                        box = [b / 1000.0 for b in box]
                
                boxes.append(box)
                
                # Extract confidence if available
                if len(groups) > 4 and groups[4]:
                    confidences.append(float(groups[4]))
                else:
                    confidences.append(1.0)  # Default confidence
                    
            except (ValueError, TypeError):
                continue
    
    if return_confidence and confidences:
        return boxes, confidences
    return boxes, None


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union with numerical stability.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score in [0, 1]
    """
    # Intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU with numerical stability
    iou = inter_area / (union_area + 1e-10)
    return np.clip(iou, 0.0, 1.0)


def compute_soft_iou(box1: List[float], box2: List[float], sigma: float = 0.5) -> float:
    """
    Compute soft IoU that gives partial credit for near-misses.
    
    Uses a Gaussian kernel to weight the IoU based on distance.
    """
    iou = compute_iou(box1, box2)
    
    # Compute center distance
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    dist = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Apply Gaussian weighting
    soft_factor = np.exp(-dist**2 / (2 * sigma**2))
    
    return iou * soft_factor


def compute_average_precision(
    recall: np.ndarray, 
    precision: np.ndarray,
    interpolation: str = '11point'
) -> float:
    """
    Compute Average Precision with different interpolation methods.
    
    Args:
        recall: Recall values
        precision: Precision values
        interpolation: '11point' or 'all' interpolation
        
    Returns:
        Average Precision score
    """
    if interpolation == '11point':
        # 11-point interpolation (PASCAL VOC style)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        return ap
    else:
        # All-point interpolation (COCO style)
        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))
        
        # Make precision monotonically decreasing
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = np.maximum(precision[i - 1], precision[i])
        
        # Calculate area under curve
        indices = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
        return ap


def compute_map_detailed(
    predicted_boxes: List[List[float]],
    ground_truth_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    config: Optional[RewardConfig] = None
) -> Dict[str, Any]:
    """
    Compute mAP with detailed metrics for analysis.
    
    Returns dictionary with:
    - mAP: Mean Average Precision
    - precision: Final precision
    - recall: Final recall
    - tp: Number of true positives
    - fp: Number of false positives
    - fn: Number of false negatives
    - iou_matrix: IoU values between predictions and GT
    - matches: List of (pred_idx, gt_idx, iou) for matched boxes
    """
    if config is None:
        config = RewardConfig()
    
    result = {
        'mAP': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'iou_matrix': None,
        'matches': []
    }
    
    # Handle edge cases
    if not predicted_boxes and not ground_truth_boxes:
        # Both empty: perfect match (true negative)
        result['mAP'] = config.no_box_bonus
        result['precision'] = 1.0
        result['recall'] = 1.0
        return result
    
    if not predicted_boxes:
        # No predictions but GT exists: all false negatives
        result['fn'] = len(ground_truth_boxes)
        return result
    
    if not ground_truth_boxes:
        # Predictions but no GT: all false positives (hallucination)
        result['fp'] = len(predicted_boxes)
        return result
    
    # Compute IoU matrix
    n_pred = len(predicted_boxes)
    n_gt = len(ground_truth_boxes)
    iou_matrix = np.zeros((n_pred, n_gt))
    
    for i, pred_box in enumerate(predicted_boxes):
        for j, gt_box in enumerate(ground_truth_boxes):
            if config.use_soft_iou:
                iou_matrix[i, j] = compute_soft_iou(pred_box, gt_box)
            else:
                iou_matrix[i, j] = compute_iou(pred_box, gt_box)
            
            # Apply IoU power for stricter/looser matching
            if config.iou_power != 1.0:
                iou_matrix[i, j] = iou_matrix[i, j] ** config.iou_power
    
    result['iou_matrix'] = iou_matrix
    
    # Greedy matching: assign each prediction to best available GT
    matched_gt = set()
    true_positives = np.zeros(n_pred)
    false_positives = np.zeros(n_pred)
    
    # Sort predictions by maximum IoU (process best matches first)
    max_ious = np.max(iou_matrix, axis=1)
    sorted_indices = np.argsort(-max_ious)
    
    for idx in sorted_indices:
        i = idx
        if iou_matrix[i].max() < iou_threshold:
            false_positives[i] = 1
            continue
            
        # Find best unmatched GT box
        available_gt = [j for j in range(n_gt) if j not in matched_gt]
        if not available_gt:
            false_positives[i] = 1
            continue
            
        best_j = max(available_gt, key=lambda j: iou_matrix[i, j])
        
        if iou_matrix[i, best_j] >= iou_threshold:
            true_positives[i] = 1
            matched_gt.add(best_j)
            result['matches'].append((i, best_j, iou_matrix[i, best_j]))
        else:
            false_positives[i] = 1
    
    # Compute metrics
    result['tp'] = int(np.sum(true_positives))
    result['fp'] = int(np.sum(false_positives))
    result['fn'] = n_gt - len(matched_gt)
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    recall = tp_cumsum / n_gt if n_gt > 0 else np.zeros_like(tp_cumsum)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    
    # Final metrics
    result['precision'] = float(precision[-1]) if len(precision) > 0 else 0.0
    result['recall'] = float(recall[-1]) if len(recall) > 0 else 0.0
    
    # Compute mAP
    result['mAP'] = compute_average_precision(recall, precision, interpolation='all')
    
    return result


# ============================================================================
# Main Reward Function
# ============================================================================

def compute_grounding_reward(
    solution_str: str,
    ground_truth: str,
    config: Optional[RewardConfig] = None,
    return_details: bool = False
) -> float | Dict[str, Any]:
    """
    Compute grounding reward with comprehensive edge case handling.
    
    Edge Cases Handled:
    1. True Negative: No predictions, no GT → NO_BOX_BONUS
    2. False Positive (Hallucination): Predictions but no GT → 0.0
    3. False Negative (Missed): No predictions but GT exists → 0.0
    4. One-to-Many: One GT, multiple predictions → Partial credit
    5. Many-to-One: Multiple GT, one prediction → Partial credit
    6. Many-to-Many: Multiple GT, multiple predictions → Full mAP
    
    Args:
        solution_str: Model output containing predictions
        ground_truth: Ground truth bounding boxes
        config: Reward configuration
        return_details: If True, return detailed metrics
        
    Returns:
        Reward score or detailed metrics dictionary
    """
    if config is None:
        config = RewardConfig()
    
    # Extract predictions from solution
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.I | re.S)
    answer_content = answer_match.group(1) if answer_match else solution_str
    
    predicted_boxes, pred_conf = extract_bounding_boxes(
        answer_content, 
        normalize=config.normalize_coordinates,
        return_confidence=config.confidence_weight
    )
    
    # Extract ground truth boxes
    gt_boxes, _ = extract_bounding_boxes(
        ground_truth,
        normalize=config.normalize_coordinates,
        return_confidence=False
    )
    
    # Limit number of boxes
    if len(predicted_boxes) > config.max_boxes:
        # Keep highest confidence boxes if available
        if pred_conf:
            indices = np.argsort(pred_conf)[-config.max_boxes:]
            predicted_boxes = [predicted_boxes[i] for i in indices]
            pred_conf = [pred_conf[i] for i in indices]
        else:
            predicted_boxes = predicted_boxes[:config.max_boxes]
    
    # Compute detailed metrics
    metrics = compute_map_detailed(predicted_boxes, gt_boxes, config.iou_threshold, config)
    
    # Apply edge case adjustments
    reward = metrics['mAP']
    
    # Case 1: True negative (already handled in compute_map_detailed)
    # Case 2: Hallucination penalty
    if gt_boxes and not predicted_boxes:
        reward -= config.missed_detection_penalty
    # Case 3: Missed detection penalty  
    elif predicted_boxes and not gt_boxes:
        reward -= config.hallucination_penalty
    
    # Ensure reward is in valid range
    reward = np.clip(reward, 0.0, 1.0)
    
    if return_details:
        metrics['reward'] = reward
        metrics['num_predictions'] = len(predicted_boxes)
        metrics['num_ground_truth'] = len(gt_boxes)
        metrics['edge_case'] = classify_edge_case(predicted_boxes, gt_boxes)
        return metrics
    
    return reward


def classify_edge_case(pred_boxes: List, gt_boxes: List) -> str:
    """Classify the scenario for analysis."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    
    if n_pred == 0 and n_gt == 0:
        return "true_negative"
    elif n_pred > 0 and n_gt == 0:
        return "hallucination"
    elif n_pred == 0 and n_gt > 0:
        return "missed_detection"
    elif n_pred == 1 and n_gt == 1:
        return "one_to_one"
    elif n_pred > 1 and n_gt == 1:
        return "many_to_one"
    elif n_pred == 1 and n_gt > 1:
        return "one_to_many"
    else:
        return "many_to_many"


# ============================================================================
# Compatibility Function
# ============================================================================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[Any] = None
) -> float:
    """
    Compatibility wrapper matching original interface.
    
    This is the main entry point for veRL integration.
    """
    config = RewardConfig()
    return compute_grounding_reward(solution_str, ground_truth, config, return_details=False)


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_reward_distribution(
    predictions: List[str],
    ground_truths: List[str],
    config: Optional[RewardConfig] = None
) -> Dict[str, Any]:
    """
    Analyze reward distribution across a dataset.
    
    Returns statistics about reward distribution and edge cases.
    """
    if config is None:
        config = RewardConfig()
    
    rewards = []
    edge_cases = {}
    detailed_metrics = []
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = compute_grounding_reward(pred, gt, config, return_details=True)
        rewards.append(metrics['reward'])
        detailed_metrics.append(metrics)
        
        edge_case = metrics['edge_case']
        if edge_case not in edge_cases:
            edge_cases[edge_case] = []
        edge_cases[edge_case].append(metrics['reward'])
    
    # Compute statistics
    rewards_array = np.array(rewards)
    
    analysis = {
        'mean_reward': float(np.mean(rewards_array)),
        'std_reward': float(np.std(rewards_array)),
        'min_reward': float(np.min(rewards_array)),
        'max_reward': float(np.max(rewards_array)),
        'median_reward': float(np.median(rewards_array)),
        'percentiles': {
            '25': float(np.percentile(rewards_array, 25)),
            '50': float(np.percentile(rewards_array, 50)),
            '75': float(np.percentile(rewards_array, 75)),
            '90': float(np.percentile(rewards_array, 90)),
            '95': float(np.percentile(rewards_array, 95))
        },
        'edge_case_distribution': {
            case: {
                'count': len(rewards),
                'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
                'std_reward': float(np.std(rewards)) if len(rewards) > 1 else 0.0
            }
            for case, rewards in edge_cases.items()
        },
        'total_samples': len(rewards_array),
        'detailed_metrics': detailed_metrics
    }
    
    return analysis


if __name__ == "__main__":
    # Example usage and testing
    print("R1 Grounding Reward Function - Enhanced Version")
    print("=" * 50)
    
    # Test cases demonstrating various scenarios
    test_cases = [
        {
            'name': 'Perfect Match',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'True Negative',
            'prediction': '<answer></answer>',
            'ground_truth': ''
        },
        {
            'name': 'Hallucination',
            'prediction': '<answer>[0.1, 0.2, 0.3, 0.4]</answer>',
            'ground_truth': ''
        },
        {
            'name': 'Missed Detection',
            'prediction': '<answer></answer>',
            'ground_truth': '[0.5, 0.5, 0.7, 0.7]'
        },
        {
            'name': 'Partial Overlap',
            'prediction': '<answer>[0.15, 0.25, 0.35, 0.45]</answer>',
            'ground_truth': '[0.1, 0.2, 0.3, 0.4]'
        },
        {
            'name': 'Multiple Boxes',
            'prediction': '<answer>[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]</answer>',
            'ground_truth': '[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6]'
        }
    ]
    
    for test in test_cases:
        result = compute_grounding_reward(
            test['prediction'],
            test['ground_truth'],
            return_details=True
        )
        print(f"\n{test['name']}:")
        print(f"  Reward: {result['reward']:.3f}")
        print(f"  Edge Case: {result['edge_case']}")
        print(f"  TP/FP/FN: {result['tp']}/{result['fp']}/{result['fn']}")