# Copyright 2025 – Apache-2.0
"""
Custom reward function for VinDR-CXR medical grounding task.
Calculates mAP (mean Average Precision) for lesion detection and localization.
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any

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
    """Extract predicted bounding boxes and lesion types from model response."""
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    if not answer_match:
        answer_text = response
    else:
        answer_text = answer_match.group(1)
    
    # Pattern to match coordinates like [0.48, 0.34, 0.59, 0.43]
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, answer_text)
    
    # Convert to float tuples
    pred_boxes = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in coordinates]
    
    # Extract lesion types
    pred_lesions = []
    lesion_keywords = {
        "aortic enlargement": "Aortic enlargement",
        "cardiomegaly": "Cardiomegaly", 
        "pleural effusion": "Pleural effusion",
        "pleural thickening": "Pleural thickening",
        "other lesion": "Other lesion",
        "no finding": "No finding",
        "no abnormalities": "No finding",
        "clear": "No finding"
    }
    
    answer_lower = answer_text.lower()
    for keyword, lesion_type in lesion_keywords.items():
        if keyword in answer_lower:
            pred_lesions.append(lesion_type)
    
    # Remove duplicates while preserving order
    pred_lesions = list(dict.fromkeys(pred_lesions))
    
    return {
        "coordinates": pred_boxes,
        "lesion_types": pred_lesions,
        "has_thinking": "<think>" in response.lower(),
        "has_answer_tags": answer_match is not None,
        "raw_answer": answer_text.strip()
    }

def calculate_map_score(pred_boxes, gt_boxes, pred_lesions, gt_lesions, iou_threshold=0.5):
    """Calculate mAP score for lesion detection and localization."""
    if not gt_boxes and not pred_boxes:
        # Both empty - perfect match for "No finding" cases
        if "No finding" in gt_lesions and "No finding" in pred_lesions:
            return {"map_score": 1.0, "localization_score": 1.0, "classification_score": 1.0}
        elif "No finding" in gt_lesions and not pred_lesions:
            return {"map_score": 1.0, "localization_score": 1.0, "classification_score": 1.0}
        else:
            return {"map_score": 0.0, "localization_score": 0.0, "classification_score": 0.0}
    
    if not gt_boxes:
        # Ground truth has no findings but model predicted some
        if "No finding" in gt_lesions:
            return {"map_score": 0.0, "localization_score": 0.0, "classification_score": 0.0}
    
    if not pred_boxes:
        # Model predicted no findings but ground truth has some
        if gt_boxes:
            return {"map_score": 0.0, "localization_score": 0.0, "classification_score": 0.0}
    
    # Calculate localization score (IoU-based matching)
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
    
    # Localization metrics
    precision = total_matches / len(pred_boxes) if pred_boxes else 0.0
    recall = total_matches / len(gt_boxes) if gt_boxes else 0.0
    localization_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Classification score (lesion type matching)
    gt_lesion_set = set(gt_lesions) - {"No finding"}
    pred_lesion_set = set(pred_lesions) - {"No finding"}
    
    if not gt_lesion_set and not pred_lesion_set:
        classification_score = 1.0
    elif not gt_lesion_set or not pred_lesion_set:
        classification_score = 0.0
    else:
        intersection = len(gt_lesion_set & pred_lesion_set)
        union = len(gt_lesion_set | pred_lesion_set)
        classification_score = intersection / union if union > 0 else 0.0
    
    # Combined mAP score (weighted combination)
    map_score = 0.7 * localization_score + 0.3 * classification_score
    
    return {
        "map_score": map_score,
        "localization_score": localization_score,
        "classification_score": classification_score
    }

def compute_score(data_source: str, solution_str: str, ground_truth, extra_info=None) -> float:
    """Main reward function for VinDR-CXR medical grounding task."""
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        # Extract predictions from model response
        predictions = extract_predictions_from_response(solution_str)
        
        # Get ground truth data
        gt_coordinates = ground_truth.get("parsed_coordinates", [])
        gt_lesions = ground_truth.get("lesion_types", [])
        
        # Calculate mAP score
        metrics = calculate_map_score(
            pred_boxes=predictions["coordinates"],
            gt_boxes=gt_coordinates,
            pred_lesions=predictions["lesion_types"],
            gt_lesions=gt_lesions
        )
        
        base_score = metrics["map_score"]
        
        # Bonus for proper formatting (reasoning structure)
        format_bonus = 0.0
        if predictions["has_thinking"]:
            format_bonus += 0.05  # Bonus for using <think> tags
        if predictions["has_answer_tags"]:
            format_bonus += 0.05  # Bonus for using <answer> tags
        
        # Penalty for completely wrong format
        if not predictions["coordinates"] and not predictions["lesion_types"] and gt_coordinates:
            format_penalty = 0.1
        else:
            format_penalty = 0.0
        
        final_score = min(1.0, max(0.0, base_score + format_bonus - format_penalty))
        
        return final_score
        
    except Exception as e:
        # Return minimal score for parsing errors
        print(f"Error in reward calculation: {e}")
        return 0.0
