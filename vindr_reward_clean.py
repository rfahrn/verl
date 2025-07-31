# Copyright 2025 – Apache-2.0
"""
VinDR-CXR mAP reward function for multi-box grounding.
Clean, focused implementation following VERL patterns.
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

def calculate_map(pred_coords, gt_coords, iou_threshold=0.5):
    """Calculate mAP for multi-box detection."""
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect "No finding" case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth
    
    # Match predictions to ground truth
    matches = []
    for pred_box in pred_coords:
        best_iou = max((calculate_iou(pred_box, gt_box) for gt_box in gt_coords), default=0.0)
        best_gt_idx = next((i for i, gt_box in enumerate(gt_coords) 
                           if calculate_iou(pred_box, gt_box) == best_iou), -1)
        
        is_match = best_iou >= iou_threshold
        matches.append((best_iou, best_gt_idx, is_match))
    
    # Sort by confidence (IoU)
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate precision/recall curve
    tp, fp = 0, 0
    matched_gt = set()
    precisions, recalls = [], []
    
    for iou_score, gt_idx, is_match in matches:
        if is_match and gt_idx not in matched_gt and gt_idx >= 0:
            tp += 1
            matched_gt.add(gt_idx)
        else:
            fp += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / len(gt_coords)
        precisions.append(precision)
        recalls.append(recall)
    
    if not precisions:
        return 0.0
    
    # Interpolated Average Precision
    precisions = [0.0] + precisions + [0.0]
    recalls = [0.0] + recalls + [1.0]
    
    # Smooth precision curve
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Calculate area under curve
    ap = sum((recalls[i] - recalls[i-1]) * precisions[i] for i in range(1, len(recalls)))
    return ap

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
        
        # Calculate mAP score (main metric)
        map_score = calculate_map(pred_coords, ground_truth)
        
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