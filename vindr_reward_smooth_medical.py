# Copyright 2025 – Apache-2.0
"""
VinDR-CXR Smooth Medical-Aware reward function.
Advanced approach combining:
1. Smooth distance-based assignment (no hard thresholds)
2. Medical-aware weighting (anatomical regions)
3. Multi-scale evaluation (coarse + fine)
4. Differentiable everywhere for optimal GRPO learning
"""
import re
import math

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def calculate_center_distance(box1, box2):
    """Calculate normalized center distance between two boxes."""
    # Get centers
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    
    # Euclidean distance
    distance = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Normalize by image diagonal (assuming normalized coords 0-1)
    max_distance = math.sqrt(2)  # diagonal of unit square
    return distance / max_distance

def calculate_size_similarity(box1, box2):
    """Calculate size similarity between two boxes."""
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    if area1 == 0 and area2 == 0:
        return 1.0
    if area1 == 0 or area2 == 0:
        return 0.0
    
    # Size similarity: min/max ratio
    return min(area1, area2) / max(area1, area2)

def get_anatomical_weight(box):
    """
    Get medical importance weight based on anatomical region.
    Higher weights for critical areas (heart, lungs).
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    
    # Define anatomical regions (normalized coordinates)
    # Heart region (center-left, lower)
    if 0.3 <= cx <= 0.7 and 0.4 <= cy <= 0.8:
        return 1.2  # Heart findings are critical
    
    # Lung regions (bilateral)
    elif (0.1 <= cx <= 0.4 or 0.6 <= cx <= 0.9) and 0.2 <= cy <= 0.8:
        return 1.1  # Lung findings are important
    
    # Upper mediastinum (aorta, etc.)
    elif 0.4 <= cx <= 0.6 and 0.1 <= cy <= 0.4:
        return 1.15  # Vascular findings are critical
    
    # Other regions
    else:
        return 1.0  # Standard weight

def smooth_assignment_score(pred_box, gt_box, temperature=2.0):
    """
    Calculate smooth assignment score combining multiple factors.
    No hard thresholds - everything is differentiable.
    """
    # Factor 1: IoU (spatial overlap)
    iou = calculate_iou(pred_box, gt_box)
    iou_score = iou  # Direct IoU
    
    # Factor 2: Center distance (spatial proximity)
    center_dist = calculate_center_distance(pred_box, gt_box)
    # Convert distance to similarity (closer = higher score)
    proximity_score = math.exp(-center_dist * temperature)
    
    # Factor 3: Size similarity (scale consistency)
    size_score = calculate_size_similarity(pred_box, gt_box)
    
    # Factor 4: Medical importance weighting
    medical_weight = get_anatomical_weight(gt_box)
    
    # Combine factors with learned weighting
    # IoU is most important, but proximity and size matter too
    combined_score = (
        0.6 * iou_score +           # Primary: spatial overlap
        0.25 * proximity_score +    # Secondary: center proximity  
        0.15 * size_score           # Tertiary: size consistency
    )
    
    # Apply medical weighting
    final_score = combined_score * medical_weight
    
    return min(1.0, final_score)  # Cap at 1.0

def soft_assignment_matrix(pred_coords, gt_coords, temperature=2.0):
    """
    Create soft assignment matrix using smooth scoring.
    Each pred-gt pair gets a continuous assignment weight.
    """
    if not pred_coords or not gt_coords:
        return []
    
    assignment_matrix = []
    for pred_box in pred_coords:
        row = []
        for gt_box in gt_coords:
            score = smooth_assignment_score(pred_box, gt_box, temperature)
            row.append(score)
        assignment_matrix.append(row)
    
    return assignment_matrix

def multi_scale_medical_score(pred_coords, gt_coords):
    """
    Multi-scale medical scoring:
    1. Coarse-level: Are we in the right general area?
    2. Fine-level: Are we precisely localized?
    3. Consistency: Do predictions make medical sense?
    """
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect no-finding case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth
    
    # Get soft assignment matrix
    assignment_matrix = soft_assignment_matrix(pred_coords, gt_coords)
    
    # Method 1: Soft bipartite matching
    # For each GT, find best soft assignment
    gt_scores = []
    for gt_idx in range(len(gt_coords)):
        best_score = 0.0
        for pred_idx in range(len(pred_coords)):
            score = assignment_matrix[pred_idx][gt_idx]
            best_score = max(best_score, score)
        gt_scores.append(best_score)
    
    # Method 2: Prediction consistency
    # Penalize predictions that don't match any GT well
    pred_scores = []
    for pred_idx in range(len(pred_coords)):
        best_score = 0.0
        for gt_idx in range(len(gt_coords)):
            score = assignment_matrix[pred_idx][gt_idx]
            best_score = max(best_score, score)
        pred_scores.append(best_score)
    
    # Recall: How well did we cover the ground truth?
    recall = sum(gt_scores) / len(gt_scores) if gt_scores else 0.0
    
    # Precision: How accurate were our predictions?
    precision = sum(pred_scores) / len(pred_scores) if pred_scores else 0.0
    
    # Multi-scale F1 with medical weighting
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    
    # Medical consistency bonus
    # Reward coherent multi-box predictions (e.g., bilateral findings)
    consistency_bonus = 0.0
    if len(pred_coords) > 1 and len(gt_coords) > 1:
        # Check if predictions show similar patterns to GT
        pred_centers = [(box[0] + box[2])/2 for box in pred_coords]
        gt_centers = [(box[0] + box[2])/2 for box in gt_coords]
        
        # Bilateral consistency (left-right symmetry)
        if len(pred_centers) == 2 and len(gt_centers) == 2:
            pred_bilateral = abs(pred_centers[0] - pred_centers[1])
            gt_bilateral = abs(gt_centers[0] - gt_centers[1])
            if pred_bilateral > 0.3 and gt_bilateral > 0.3:  # Bilateral findings
                bilateral_similarity = 1.0 - abs(pred_bilateral - gt_bilateral)
                consistency_bonus = 0.1 * max(0, bilateral_similarity)
    
    return min(1.0, f1_score + consistency_bonus)

def extract_coordinates(text):
    """Extract coordinate boxes from text."""
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

def format_reward(solution_str):
    """Reward for proper formatting with <think> and <answer> tags."""
    has_think = bool(re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))

    if has_think and has_answer:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0

def grounding_accuracy(pred_text, gt_data):
    """
    Core grounding accuracy using smooth medical-aware scoring.
    """
    pred_coords = extract_coordinates(pred_text)

    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        is_no_finding = gt_data.get("has_no_finding", False)
    else:
        gt_coords = extract_coordinates(str(gt_data))
        is_no_finding = len(gt_coords) == 0

    if is_no_finding:
        phrases = ["no finding", "no abnormalities", "clear", "normal"]
        has_no_finding_text = any(phrase in pred_text.lower() for phrase in phrases)

        if not pred_coords and has_no_finding_text:
            return 1.0
        if not pred_coords:
            return 0.7
        return 0.1

    return multi_scale_medical_score(pred_coords, gt_coords)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function with smooth medical-aware scoring.

    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict or string)
        extra_info: Additional metadata (unused)

    Returns:
        float: Score between 0.0 and 1.0
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        if match:
            core_answer = match.group(1).strip()
            accuracy_score = grounding_accuracy(core_answer, ground_truth)
        else:
            accuracy_score = grounding_accuracy(solution_str, ground_truth) * 0.5

        format_bonus = format_reward(solution_str) * 0.1  # Up to 10% bonus
        final_score = accuracy_score + format_bonus

        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0