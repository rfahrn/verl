# Copyright 2025 – Apache-2.0
"""
Simple reward function for VinDR-CXR localization grounding.
Follows VERL RewardManager contract exactly.
Priority: Localization accuracy (coordinates) > Classification accuracy
"""

import re
import numpy as np

def calculate_iou(box1, box2):
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

def extract_coordinates_from_response(response_str):
    """Extract predicted coordinates from model response."""
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1)
    else:
        answer_text = response_str
    
    # Pattern to match coordinates
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, answer_text)
    
    # Convert to float lists
    coord_list = []
    for x1, y1, x2, y2 in coordinates:
        coord_list.append([float(x1), float(y1), float(x2), float(y2)])
    
    return coord_list

def calculate_localization_score(pred_coords, gt_coords, iou_threshold=0.5):
    """
    Calculate localization score based on IoU matching.
    Priority: Localization accuracy for grounding task.
    """
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect match for "No finding" cases
    
    if not gt_coords:
        return 0.0  # False positives
    
    if not pred_coords:
        return 0.0  # False negatives
    
    # IoU-based matching (greedy assignment)
    matched_gt = [False] * len(gt_coords)
    matched_pred = [False] * len(pred_coords)
    total_matches = 0
    
    for i, pred_box in enumerate(pred_coords):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_coords):
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
    
    # Calculate precision and recall
    precision = total_matches / len(pred_coords) if pred_coords else 0.0
    recall = total_matches / len(gt_coords) if gt_coords else 0.0
    
    # F1 score as localization metric
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function for VinDR-CXR localization grounding.
    
    Args:
        data_source: Dataset identifier (string) - should be "vindr_grpo"
        solution_str: Model's generated response (string)
        ground_truth: Ground truth data from parquet file (list of coordinates or string)
        extra_info: Additional information (optional, unused)
    
    Returns:
        float: Reward score between 0.0 and 1.0
    """
    # Validate data source
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        # Extract predicted coordinates from model response
        pred_coords = extract_coordinates_from_response(solution_str)
        
        # Handle different ground truth formats
        if ground_truth == "NO_FINDING":
            # Perfect score if model correctly identifies no findings
            if not pred_coords and ("no finding" in solution_str.lower() or 
                                   "no abnormalities" in solution_str.lower() or
                                   "clear" in solution_str.lower()):
                format_bonus = 0.1 if "<think>" in solution_str and "<answer>" in solution_str else 0.0
                return min(1.0, 0.9 + format_bonus)
            # Penalty for false positives
            elif pred_coords:
                return 0.1
            else:
                return 0.7  # Partial credit for not hallucinating
        
        elif ground_truth == "PARSING_ERROR":
            # Minimal score for parsing errors in ground truth
            return 0.1 if solution_str.strip() else 0.0
        
        elif isinstance(ground_truth, list):
            # Main case: ground truth is list of coordinates
            gt_coords = ground_truth
        else:
            # Fallback: try to parse ground truth as string
            return 0.1
        
        # Calculate localization score (PRIORITY: this is the main metric)
        localization_score = calculate_localization_score(pred_coords, gt_coords)
        
        # Format bonuses for proper reasoning structure
        format_bonus = 0.0
        if "<think>" in solution_str.lower():
            format_bonus += 0.05
        if "<answer>" in solution_str.lower():
            format_bonus += 0.05
        
        # Consistency check: if model mentions findings, should provide coordinates
        consistency_penalty = 0.0
        has_findings_mentioned = any(finding in solution_str.lower() for finding in 
                                   ["enlargement", "cardiomegaly", "effusion", "thickening", 
                                    "lesion", "abnormality", "opacity", "consolidation"])
        
        if has_findings_mentioned and not pred_coords and gt_coords:
            consistency_penalty = 0.1  # Mentioned findings but no coordinates
        
        # Final score: prioritize localization
        final_score = min(1.0, max(0.0, localization_score + format_bonus - consistency_penalty))
        
        return final_score
        
    except Exception as e:
        # Return minimal score for any errors
        print(f"Error in VinDR reward calculation: {e}")
        return 0.0

# Alternative: Even simpler version focused purely on coordinate matching
def compute_score_coords_only(data_source, solution_str, ground_truth, extra_info=None):
    """Pure coordinate matching reward function."""
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        pred_coords = extract_coordinates_from_response(solution_str)
        
        if ground_truth == "NO_FINDING":
            return 1.0 if not pred_coords else 0.0
        elif isinstance(ground_truth, list):
            return calculate_localization_score(pred_coords, ground_truth)
        else:
            return 0.0
    except Exception:
        return 0.0