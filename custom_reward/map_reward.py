#!/usr/bin/env python3
"""
mAP (mean Average Precision) Reward Function for Medical Image Grounding

This reward function implements a comprehensive mAP evaluation similar to COCO metrics,
evaluating detection performance across multiple IoU thresholds and providing detailed
insights into model performance at different precision levels.

Key Features:
- Multi-threshold evaluation (mAP@[0.5:0.05:0.95])
- Per-class Average Precision calculation
- Precision-Recall curve analysis
- Support for "no finding" cases
- Detailed performance breakdown

Based on COCO evaluation methodology and adapted for medical image grounding tasks.
"""

import json
import re
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_bounding_boxes_from_answer(answer: str) -> List[List[float]]:
    """
    Extract bounding boxes from model answer.
    
    Args:
        answer: Model's text response containing bounding box coordinates
        
    Returns:
        List of bounding boxes in format [[x1, y1, x2, y2], ...]
    """
    # Pattern to match bounding box coordinates
    bbox_pattern = r'<(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)>'
    matches = re.findall(bbox_pattern, answer)
    
    bboxes = []
    for match in matches:
        try:
            x1, y1, x2, y2 = map(float, match)
            # Normalize coordinates to [0, 1] if they appear to be in pixel coordinates
            if any(coord > 1.0 for coord in [x1, y1, x2, y2]):
                # Assume coordinates are in a 1000x1000 space (common in medical imaging)
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            
            # Ensure valid bounding box
            if x1 < x2 and y1 < y2:
                bboxes.append([x1, y1, x2, y2])
        except (ValueError, TypeError):
            continue
    
    return bboxes

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_detections_to_ground_truth(
    predicted_boxes: List[List[float]], 
    ground_truth_boxes: List[List[float]], 
    iou_threshold: float = 0.5
) -> Tuple[List[bool], List[int]]:
    """
    Match predicted boxes to ground truth boxes using IoU threshold.
    
    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for considering a match
        
    Returns:
        Tuple of (is_true_positive, matched_gt_index) for each prediction
    """
    if not predicted_boxes:
        return [], []
    
    if not ground_truth_boxes:
        return [False] * len(predicted_boxes), [-1] * len(predicted_boxes)
    
    is_tp = []
    matched_gt = []
    gt_matched = set()
    
    for pred_box in predicted_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(ground_truth_boxes):
            if gt_idx in gt_matched:
                continue
                
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            is_tp.append(True)
            matched_gt.append(best_gt_idx)
            gt_matched.add(best_gt_idx)
        else:
            is_tp.append(False)
            matched_gt.append(-1)
    
    return is_tp, matched_gt

def compute_precision_recall(
    is_tp: List[bool], 
    num_ground_truth: int
) -> Tuple[List[float], List[float]]:
    """
    Compute precision and recall arrays for precision-recall curve.
    
    Args:
        is_tp: Boolean array indicating true positives
        num_ground_truth: Total number of ground truth objects
        
    Returns:
        Tuple of (precision_array, recall_array)
    """
    if not is_tp:
        return [1.0], [0.0]
    
    # Cumulative true positives and false positives
    tp_cumsum = np.cumsum(is_tp)
    fp_cumsum = np.cumsum([not tp for tp in is_tp])
    
    # Precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / max(num_ground_truth, 1)
    
    # Add (0, 1) point for proper curve interpolation
    precisions = np.concatenate([[1.0], precisions])
    recalls = np.concatenate([[0.0], recalls])
    
    return precisions.tolist(), recalls.tolist()

def compute_average_precision(
    precisions: List[float], 
    recalls: List[float], 
    interpolation_method: str = "101-point"
) -> float:
    """
    Compute Average Precision (AP) from precision-recall curve.
    
    Args:
        precisions: Precision values
        recalls: Recall values
        interpolation_method: "101-point" (COCO) or "11-point" (PASCAL VOC)
        
    Returns:
        Average Precision score
    """
    if len(precisions) <= 1:
        return 0.0
    
    # Ensure monotonic decreasing precision
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    if interpolation_method == "101-point":
        # COCO-style 101-point interpolation
        recall_thresholds = np.linspace(0, 1, 101)
        interpolated_precisions = np.interp(recall_thresholds, recalls, precisions)
        return np.mean(interpolated_precisions)
    
    elif interpolation_method == "11-point":
        # PASCAL VOC-style 11-point interpolation
        recall_thresholds = np.linspace(0, 1, 11)
        interpolated_precisions = []
        
        for r_thresh in recall_thresholds:
            # Find precisions for recalls >= r_thresh
            valid_precisions = precisions[recalls >= r_thresh]
            if len(valid_precisions) > 0:
                interpolated_precisions.append(np.max(valid_precisions))
            else:
                interpolated_precisions.append(0.0)
        
        return np.mean(interpolated_precisions)
    
    else:
        # All-point interpolation (area under curve)
        return np.trapz(precisions, recalls)

def compute_map_single_threshold(
    predicted_boxes: List[List[float]], 
    ground_truth_boxes: List[List[float]], 
    iou_threshold: float = 0.5,
    interpolation_method: str = "101-point"
) -> Dict[str, float]:
    """
    Compute mAP at a single IoU threshold.
    
    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        iou_threshold: IoU threshold for TP/FP classification
        interpolation_method: Interpolation method for AP calculation
        
    Returns:
        Dictionary with mAP metrics
    """
    # Handle no finding cases
    if not ground_truth_boxes and not predicted_boxes:
        return {
            "ap": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "num_predictions": 0,
            "num_ground_truth": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0
        }
    
    if not ground_truth_boxes and predicted_boxes:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_predictions": len(predicted_boxes),
            "num_ground_truth": 0,
            "true_positives": 0,
            "false_positives": len(predicted_boxes),
            "false_negatives": 0
        }
    
    if ground_truth_boxes and not predicted_boxes:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "num_predictions": 0,
            "num_ground_truth": len(ground_truth_boxes),
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(ground_truth_boxes)
        }
    
    # Match predictions to ground truth
    is_tp, matched_gt = match_detections_to_ground_truth(
        predicted_boxes, ground_truth_boxes, iou_threshold
    )
    
    # Compute precision-recall curve
    precisions, recalls = compute_precision_recall(is_tp, len(ground_truth_boxes))
    
    # Compute Average Precision
    ap = compute_average_precision(precisions, recalls, interpolation_method)
    
    # Calculate summary statistics
    true_positives = sum(is_tp)
    false_positives = len(predicted_boxes) - true_positives
    false_negatives = len(ground_truth_boxes) - true_positives
    
    final_precision = true_positives / len(predicted_boxes) if predicted_boxes else 0.0
    final_recall = true_positives / len(ground_truth_boxes) if ground_truth_boxes else 0.0
    
    return {
        "ap": ap,
        "precision": final_precision,
        "recall": final_recall,
        "num_predictions": len(predicted_boxes),
        "num_ground_truth": len(ground_truth_boxes),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precisions": precisions,
        "recalls": recalls
    }

def compute_map_coco_style(
    predicted_boxes: List[List[float]], 
    ground_truth_boxes: List[List[float]]
) -> Dict[str, Any]:
    """
    Compute COCO-style mAP across multiple IoU thresholds.
    
    Args:
        predicted_boxes: List of predicted bounding boxes
        ground_truth_boxes: List of ground truth bounding boxes
        
    Returns:
        Dictionary with comprehensive mAP metrics
    """
    # IoU thresholds for COCO evaluation
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, 0.6, ..., 0.95]
    
    results = {}
    ap_scores = []
    
    # Compute AP for each IoU threshold
    for iou_thresh in iou_thresholds:
        thresh_results = compute_map_single_threshold(
            predicted_boxes, ground_truth_boxes, iou_thresh
        )
        results[f"AP@{iou_thresh:.2f}"] = thresh_results["ap"]
        ap_scores.append(thresh_results["ap"])
    
    # Compute summary metrics
    map_score = np.mean(ap_scores)  # mAP@[0.5:0.05:0.95]
    
    # Also compute specific thresholds
    ap50_results = compute_map_single_threshold(predicted_boxes, ground_truth_boxes, 0.5)
    ap75_results = compute_map_single_threshold(predicted_boxes, ground_truth_boxes, 0.75)
    
    results.update({
        "mAP": map_score,
        "AP@0.50": ap50_results["ap"],
        "AP@0.75": ap75_results["ap"],
        "precision@0.50": ap50_results["precision"],
        "recall@0.50": ap50_results["recall"],
        "num_predictions": len(predicted_boxes),
        "num_ground_truth": len(ground_truth_boxes),
        "iou_thresholds": iou_thresholds.tolist(),
        "ap_per_threshold": ap_scores
    })
    
    return results

def compute_reward(
    answer: str, 
    reward_model_input: Dict[str, Any]
) -> float:
    """
    Main reward function using mAP evaluation.
    
    Args:
        answer: Model's text response
        reward_model_input: Dictionary containing ground truth information
        
    Returns:
        Reward score between 0 and 1
    """
    try:
        # Extract predicted bounding boxes from answer
        predicted_boxes = extract_bounding_boxes_from_answer(answer)
        
        # Get ground truth boxes
        ground_truth_boxes = reward_model_input.get("ground_truth", [])
        
        # Handle "no finding" cases in the answer
        answer_lower = answer.lower()
        no_finding_phrases = [
            "no abnormality", "no finding", "normal", "no abnormal", 
            "nothing abnormal", "no pathology", "unremarkable"
        ]
        
        answer_indicates_no_finding = any(phrase in answer_lower for phrase in no_finding_phrases)
        
        # Special handling for no finding cases
        if not ground_truth_boxes:  # Ground truth has no findings
            if not predicted_boxes or answer_indicates_no_finding:
                # Correct: no findings predicted when none exist
                return 1.0
            else:
                # False positive: predicted findings when none exist
                # Use mAP score which will be 0 for this case
                pass
        
        # Compute COCO-style mAP
        map_results = compute_map_coco_style(predicted_boxes, ground_truth_boxes)
        
        # Primary reward based on mAP@[0.5:0.05:0.95]
        base_reward = map_results["mAP"]
        
        # Bonus for high precision at IoU=0.5 (clinically relevant)
        precision_bonus = 0.1 * map_results["precision@0.50"]
        
        # Bonus for high recall at IoU=0.5 (don't miss findings)
        recall_bonus = 0.1 * map_results["recall@0.50"]
        
        # Penalty for excessive false positives
        if map_results["num_predictions"] > 0:
            fp_rate = (map_results["num_predictions"] - 
                      sum(1 for thresh in np.arange(0.5, 1.0, 0.05) 
                          for ap in [compute_map_single_threshold(predicted_boxes, ground_truth_boxes, thresh)["true_positives"]])) / map_results["num_predictions"]
            fp_penalty = 0.05 * fp_rate
        else:
            fp_penalty = 0.0
        
        # Final reward calculation
        final_reward = base_reward + precision_bonus + recall_bonus - fp_penalty
        
        # Ensure reward is in [0, 1] range
        final_reward = max(0.0, min(1.0, final_reward))
        
        # Log detailed results for debugging
        logger.info(f"mAP Reward Details:")
        logger.info(f"  mAP@[0.5:0.05:0.95]: {map_results['mAP']:.3f}")
        logger.info(f"  AP@0.50: {map_results['AP@0.50']:.3f}")
        logger.info(f"  AP@0.75: {map_results['AP@0.75']:.3f}")
        logger.info(f"  Precision@0.50: {map_results['precision@0.50']:.3f}")
        logger.info(f"  Recall@0.50: {map_results['recall@0.50']:.3f}")
        logger.info(f"  Predicted boxes: {len(predicted_boxes)}")
        logger.info(f"  Ground truth boxes: {len(ground_truth_boxes)}")
        logger.info(f"  Final reward: {final_reward:.3f}")
        
        return final_reward
        
    except Exception as e:
        logger.error(f"Error in mAP reward computation: {e}")
        return 0.0

# Backward compatibility
def reward_function(answer: str, reward_model_input: Dict[str, Any]) -> float:
    """Backward compatibility wrapper."""
    return compute_reward(answer, reward_model_input)

if __name__ == "__main__":
    # Test the mAP reward function
    test_cases = [
        {
            "description": "Perfect detection",
            "answer": "The chest X-ray shows a pneumothorax located at <100,150,200,250>.",
            "ground_truth": [[0.1, 0.15, 0.2, 0.25]]
        },
        {
            "description": "Multiple detections with varying quality",
            "answer": "Multiple abnormalities: <50,100,150,200> and <300,400,450,550> and <600,700,800,900>",
            "ground_truth": [[0.05, 0.1, 0.15, 0.2], [0.3, 0.4, 0.45, 0.55]]
        },
        {
            "description": "No findings (correct)",
            "answer": "The chest X-ray appears normal with no abnormality detected.",
            "ground_truth": []
        },
        {
            "description": "False positive",
            "answer": "Suspicious opacity at <100,100,200,200>",
            "ground_truth": []
        },
        {
            "description": "Missed detection",
            "answer": "The image appears normal.",
            "ground_truth": [[0.1, 0.1, 0.3, 0.3]]
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['description']} ---")
        reward_input = {"ground_truth": test_case["ground_truth"]}
        reward = compute_reward(test_case["answer"], reward_input)
        print(f"Answer: {test_case['answer']}")
        print(f"Ground Truth: {test_case['ground_truth']}")
        print(f"mAP Reward: {reward:.3f}")