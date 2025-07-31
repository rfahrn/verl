# Copyright 2025 – Apache-2.0
"""
Corrected reward function for VinDR-CXR medical grounding task.
Follows VERL's RewardManager contract exactly and handles all 28 VinDR findings.
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# All 28 VinDR-CXR findings
LOCAL_LABELS = [
    "Aortic enlargement", "Atelectasis", "Cardiomegaly", "Calcification", 
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA",
    "Interstitial lung disease", "Infiltration", "Lung cavity", "Lung cyst",
    "Lung opacity", "Mediastinal shift", "Nodule/Mass", "Pulmonary fibrosis",
    "Pneumothorax", "Pleural thickening", "Pleural effusion", "Rib fracture",
    "Other lesion"
]

GLOBAL_LABELS = [
    "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", 
    "Chronic obstructive pulmonary disease", "No finding"
]

def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def extract_predictions_from_response(response: str) -> Dict[str, Any]:
    """Extract predicted findings and coordinates from model response."""
    # Extract thinking and answer sections
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    
    if answer_match:
        answer_text = answer_match.group(1)
    else:
        # Fallback to full response if no answer tags
        answer_text = response
    
    # Extract coordinates
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, answer_text)
    pred_boxes = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in coordinates]
    
    # Extract local findings (should have coordinates)
    pred_local = []
    answer_lower = answer_text.lower()
    
    for finding in LOCAL_LABELS:
        if finding.lower() in answer_lower:
            pred_local.append(finding)
    
    # Extract global findings (diagnostic impressions)
    pred_global = []
    for finding in GLOBAL_LABELS:
        if finding.lower() in answer_lower:
            pred_global.append(finding)
    
    # Remove duplicates while preserving order
    pred_local = list(dict.fromkeys(pred_local))
    pred_global = list(dict.fromkeys(pred_global))
    
    return {
        "coordinates": pred_boxes,
        "local_findings": pred_local,
        "global_findings": pred_global,
        "has_thinking": think_match is not None,
        "has_answer_tags": answer_match is not None,
        "raw_answer": answer_text.strip()
    }

def calculate_localization_map(pred_boxes: List[Tuple[float, float, float, float]],
                              gt_boxes: List[Tuple[float, float, float, float]],
                              iou_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate mAP for localization task."""
    if not gt_boxes and not pred_boxes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "map": 1.0}
    
    if not gt_boxes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "map": 0.0}
    
    if not pred_boxes:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "map": 0.0}
    
    # IoU-based matching
    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    total_matches = 0
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if matched_gt[j]:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = j
        
        if best_gt_idx >= 0:
            matched_gt[best_gt_idx] = True
            matched_pred[i] = True
            total_matches += 1
    
    precision = total_matches / len(pred_boxes) if pred_boxes else 0.0
    recall = total_matches / len(gt_boxes) if gt_boxes else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1, "map": f1}

def calculate_classification_f1(pred_findings: List[str], 
                               gt_findings: List[str]) -> Dict[str, float]:
    """Calculate F1 score for classification task."""
    if not gt_findings and not pred_findings:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    pred_set = set(pred_findings)
    gt_set = set(gt_findings)
    
    if not gt_set:
        return {"precision": 0.0 if pred_set else 1.0, "recall": 1.0, "f1": 0.0 if pred_set else 1.0}
    
    if not pred_set:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    
    intersection = len(pred_set & gt_set)
    precision = intersection / len(pred_set)
    recall = intersection / len(gt_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """
    VERL-compliant reward function for VinDR-CXR medical grounding.
    
    Args:
        data_source: Dataset identifier (string) - should be "vindr_grpo"
        solution_str: Model's generated response (string)
        ground_truth: Ground truth data (dict) from parquet file
        extra_info: Additional information (optional, unused)
    
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    # Validate data source
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        # Handle different ground_truth formats
        if isinstance(ground_truth, str):
            # Fallback: if ground_truth is just a string, minimal scoring
            return 0.1 if solution_str.strip() else 0.0
        
        if not isinstance(ground_truth, dict):
            return 0.0
        
        # Extract predictions from model response
        predictions = extract_predictions_from_response(solution_str)
        
        # Get ground truth data
        gt_coordinates = ground_truth.get("coordinates", [])
        gt_local_findings = ground_truth.get("local_findings", [])
        gt_global_findings = ground_truth.get("global_findings", [])
        has_no_finding = ground_truth.get("has_no_finding", False)
        
        # Handle "No finding" cases specially
        if has_no_finding:
            # Perfect score if model correctly identifies no findings
            if ("no finding" in solution_str.lower() or 
                "no abnormalities" in solution_str.lower() or
                "clear" in solution_str.lower()):
                format_bonus = 0.1 if predictions["has_thinking"] and predictions["has_answer_tags"] else 0.0
                return min(1.0, 0.9 + format_bonus)
            # Penalty for false positives on normal cases
            elif predictions["coordinates"] or predictions["local_findings"] or predictions["global_findings"]:
                return 0.1
            else:
                return 0.7  # Partial credit for not hallucinating findings
        
        # Calculate localization performance (mAP for bounding boxes)
        localization_metrics = calculate_localization_map(
            pred_boxes=predictions["coordinates"],
            gt_boxes=gt_coordinates
        )
        
        # Calculate local findings classification (F1 for findings that need localization)
        local_classification_metrics = calculate_classification_f1(
            pred_findings=predictions["local_findings"],
            gt_findings=gt_local_findings
        )
        
        # Calculate global findings classification (F1 for diagnostic impressions)
        global_classification_metrics = calculate_classification_f1(
            pred_findings=predictions["global_findings"],
            gt_findings=gt_global_findings
        )
        
        # Weighted combination of scores
        # Localization is most important for medical grounding
        localization_score = localization_metrics["map"]
        local_classification_score = local_classification_metrics["f1"]
        global_classification_score = global_classification_metrics["f1"]
        
        # Combined score with emphasis on localization
        base_score = (0.5 * localization_score + 
                     0.3 * local_classification_score + 
                     0.2 * global_classification_score)
        
        # Format bonuses
        format_bonus = 0.0
        if predictions["has_thinking"]:
            format_bonus += 0.05
        if predictions["has_answer_tags"]:
            format_bonus += 0.05
        
        # Consistency penalty
        consistency_penalty = 0.0
        # Penalize if local findings mentioned but no coordinates provided
        if predictions["local_findings"] and not predictions["coordinates"]:
            consistency_penalty += 0.1
        # Penalize if coordinates provided but no local findings mentioned
        if predictions["coordinates"] and not predictions["local_findings"]:
            consistency_penalty += 0.05
        
        final_score = min(1.0, max(0.0, base_score + format_bonus - consistency_penalty))
        
        return final_score
        
    except Exception as e:
        # Return minimal score for any errors
        print(f"Error in VinDR reward calculation: {e}")
        return 0.0

# Alternative reward functions for different training phases

def compute_score_localization_only(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """Reward function focused only on localization accuracy."""
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        predictions = extract_predictions_from_response(solution_str)
        gt_coordinates = ground_truth.get("coordinates", [])
        
        if ground_truth.get("has_no_finding", False):
            return 1.0 if not predictions["coordinates"] else 0.0
        
        metrics = calculate_localization_map(predictions["coordinates"], gt_coordinates)
        return metrics["map"]
        
    except Exception:
        return 0.0

def compute_score_classification_only(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """Reward function focused only on classification accuracy."""
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        predictions = extract_predictions_from_response(solution_str)
        gt_local = ground_truth.get("local_findings", [])
        gt_global = ground_truth.get("global_findings", [])
        
        local_f1 = calculate_classification_f1(predictions["local_findings"], gt_local)["f1"]
        global_f1 = calculate_classification_f1(predictions["global_findings"], gt_global)["f1"]
        
        return 0.7 * local_f1 + 0.3 * global_f1
        
    except Exception:
        return 0.0