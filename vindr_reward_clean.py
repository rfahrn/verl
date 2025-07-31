# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy mAP reward function for multi-box grounding.
Uses continuous IoU-based scoring instead of binary thresholds.

Fuzzy Scoring:
- IoU 0.1 → score 0.1 (minimal match)
- IoU 0.5 → score 0.6 (good match) 
- IoU 1.0 → score 1.0 (perfect match)
"""
import re

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_inter, y1_inter = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_inter, y2_inter = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def extract_predicted_coords(response_str):
    """Extract coordinates from model response."""
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL | re.IGNORECASE)
    text = answer_match.group(1) if answer_match else response_str
    
    # Find all coordinate patterns
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, text)
    
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]

def calculate_fuzzy_map(pred_coords, gt_coords, min_iou=0.1):
    """Calculate fuzzy mAP with continuous IoU-based scoring."""
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect "No finding" case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth
    
    # Match predictions to ground truth with fuzzy scoring
    matches = []
    for pred_box in pred_coords:
        best_iou = max((calculate_iou(pred_box, gt_box) for gt_box in gt_coords), default=0.0)
        best_gt_idx = next((i for i, gt_box in enumerate(gt_coords) 
                           if calculate_iou(pred_box, gt_box) == best_iou), -1)
        
        # Fuzzy match quality based on IoU
        if best_iou >= min_iou:
            # Continuous scoring: IoU 0.1->0.1, IoU 0.5->0.6, IoU 1.0->1.0
            fuzzy_score = (best_iou - min_iou) / (1.0 - min_iou) * 0.9 + 0.1
            matches.append((best_iou, best_gt_idx, fuzzy_score))
        else:
            matches.append((best_iou, -1, 0.0))  # No match
    
    # Sort by IoU confidence
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate fuzzy precision/recall curve
    weighted_tp, fp = 0.0, 0
    matched_gt = set()
    precisions, recalls = [], []
    
    for iou_score, gt_idx, fuzzy_score in matches:
        if fuzzy_score > 0 and gt_idx not in matched_gt and gt_idx >= 0:
            weighted_tp += fuzzy_score  # Add fuzzy match weight
            matched_gt.add(gt_idx)
        else:
            fp += 1
        
        # Fuzzy precision: weighted true positives / total predictions
        precision = weighted_tp / (weighted_tp + fp) if (weighted_tp + fp) > 0 else 0.0
        # Fuzzy recall: weighted matches / total ground truth
        recall = weighted_tp / len(gt_coords) if gt_coords else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
    
    if not precisions:
        return 0.0
    
    # Calculate area under fuzzy precision-recall curve
    # Simple trapezoidal integration
    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] > recalls[i-1]:  # Only count increasing recall
            ap += (recalls[i] - recalls[i-1]) * (precisions[i] + precisions[i-1]) / 2
    
    return min(1.0, ap)  # Cap at 1.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """VERL-compliant reward function for VinDR-CXR multi-box grounding."""
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        pred_coords = extract_predicted_coords(solution_str)
        
        # Handle special ground truth cases
        if ground_truth == "NO_FINDING":
            no_finding_mentioned = any(phrase in solution_str.lower() 
                                     for phrase in ["no finding", "no abnormalities", "clear"])
            if not pred_coords and no_finding_mentioned:
                return 1.0  # Perfect "no finding" case
            return 0.1 if pred_coords else 0.7  # Penalty for false positives
        
        if ground_truth == "PARSING_ERROR":
            return 0.1 if solution_str.strip() else 0.0
        
        if not isinstance(ground_truth, list):
            return 0.1  # Fallback for unexpected format
        
        # Calculate fuzzy mAP score (main metric)
        map_score = calculate_fuzzy_map(pred_coords, ground_truth)
        
        # Format bonuses
        format_bonus = 0.0
        if "<think>" in solution_str.lower():
            format_bonus += 0.05
        if "<answer>" in solution_str.lower():
            format_bonus += 0.05
        
        return min(1.0, max(0.0, map_score + format_bonus))
        
    except Exception as e:
        print(f"VinDR reward error: {e}")
        return 0.0