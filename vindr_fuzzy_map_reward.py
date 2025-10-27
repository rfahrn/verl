# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy mAP reward function for medical image grounding.
Clean implementation following GSM8K/RadGraph pattern.
"""
import re

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
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

def extract_coordinates(text):
    """Extract coordinate boxes from text."""
    # Find all [x1, y1, x2, y2] patterns
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

def fuzzy_map_score(pred_coords, gt_coords):
    """
    Calculate fuzzy mAP score between predicted and ground truth boxes.
    Returns score between 0.0 and 1.0.
    """
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect no-finding case
    
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth
    
    # Create IoU matrix between all pred/gt pairs
    iou_matrix = []
    for pred_box in pred_coords:
        row = [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        iou_matrix.append(row)
    
    # Greedy assignment: match each prediction to best available GT
    assigned_gt = set()
    assigned_pred = set()
    total_score = 0.0
    
    # Get all valid matches above threshold (IoU >= 0.1)
    matches = []
    for pred_idx, row in enumerate(iou_matrix):
        for gt_idx, iou in enumerate(row):
            if iou >= 0.1:  # Minimum IoU threshold
                matches.append((iou, pred_idx, gt_idx))
    
    # Sort by IoU (best matches first)
    matches.sort(reverse=True)
    
    # Assign matches greedily
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            # Convert IoU to fuzzy score: 0.1→0.1, 1.0→1.0
            fuzzy_score = (iou - 0.1) / 0.9 * 0.9 + 0.1
            total_score += fuzzy_score
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)
    
    # Calculate fuzzy precision and recall
    n_pred = len(pred_coords)
    n_gt = len(gt_coords)
    
    precision = total_score / n_pred if n_pred > 0 else 0.0
    recall = total_score / n_gt if n_gt > 0 else 0.0
    
    # Fuzzy F1 score
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score

def format_reward(solution_str):
    """Reward for proper formatting with <think> and <answer> tags."""
    has_think = bool(re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))
    
    if has_think and has_answer:
        return 1.0
    elif has_answer:
        return 0.5
    else:
        return 0.0

def grounding_accuracy(pred_text, gt_data):
    """
    Core grounding accuracy function.
    Similar to radgraph_partial_score but for coordinate grounding.
    """
    # Extract coordinates from prediction
    pred_coords = extract_coordinates(pred_text)
    
    # Handle ground truth format
    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        is_no_finding = gt_data.get("has_no_finding", False)
    else:
        # Fallback: assume it's the raw answer string
        gt_coords = extract_coordinates(str(gt_data))
        is_no_finding = len(gt_coords) == 0
    
    # Handle no-finding cases
    if is_no_finding:
        no_finding_phrases = ["no finding", "no abnormalities", "clear", "normal"]
        has_no_finding_text = any(phrase in pred_text.lower() for phrase in no_finding_phrases)
        
        if not pred_coords and has_no_finding_text:
            return 1.0  # Perfect no-finding prediction
        elif not pred_coords:
            return 0.7   # Correct but missing explicit statement
        else:
            return 0.1   # False positive
    
    # Calculate fuzzy mAP score
    return fuzzy_map_score(pred_coords, gt_coords)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function following GSM8K/RadGraph pattern.
    
    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict or string)
        extra_info: Additional metadata (unused)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    # Filter by data source
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        # Extract core answer from <answer> tags (like RadGraph example)
        answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
        if answer_match:
            core_answer = answer_match.group(1).strip()
            # Calculate main grounding accuracy
            accuracy_score = grounding_accuracy(core_answer, ground_truth)
        else:
            # No answer tags found - fallback to full text but penalize
            accuracy_score = grounding_accuracy(solution_str, ground_truth) * 0.5
        
        # Add small format bonus (like GSM8K example)
        format_score = format_reward(solution_str)
        format_bonus = format_score * 0.1  # Max 10% bonus
        
        # Combine scores
        final_score = accuracy_score + format_bonus
        
        return min(1.0, max(0.0, final_score))
        
    except Exception as e:
        # Graceful fallback on errors
        return 0.0