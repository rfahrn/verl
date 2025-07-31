# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy multi-box reward function for grounding.
Handles multiple predicted boxes vs multiple ground truth boxes.
Uses optimal assignment with continuous IoU-based scoring.

Multi-box Fuzzy Scoring:
- Creates IoU matrix between all pred/gt box pairs
- Greedy assignment (each box matched at most once)
- IoU 0.1 → score 0.1, IoU 0.5 → score 0.6, IoU 1.0 → score 1.0
- Final score: Fuzzy F1 with penalties for unmatched boxes
"""
import re

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    try:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    except ValueError as e:
        print(f"IoU unpacking error - box1: {box1}, box2: {box2}")
        return 0.0

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

    try:
        result = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]
        return result
    except ValueError as e:
        print(f"Coordinate parsing error - coordinates: {coordinates}")
        return []

def calculate_fuzzy_multibox_score(pred_coords, gt_coords, min_iou=0.1):
    """Calculate fuzzy multi-box score with optimal assignment."""
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect "No finding" case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth

    # Create IoU matrix: pred_boxes x gt_boxes
    iou_matrix = []
    for pred_box in pred_coords:
        iou_row = [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        iou_matrix.append(iou_row)

    # Simple greedy assignment (can be replaced with Hungarian algorithm for optimal)
    assigned_gt = set()
    assigned_pred = set()
    total_fuzzy_score = 0.0
    
    # Sort all possible matches by IoU score (greedy approximation)
    all_matches = []
    for pred_idx, iou_row in enumerate(iou_matrix):
        for gt_idx, iou_score in enumerate(iou_row):
            if iou_score >= min_iou:
                all_matches.append((iou_score, pred_idx, gt_idx))

    # Sort by IoU descending (best matches first)
    all_matches.sort(key=lambda x: x[0], reverse=True)

    # Assign matches greedily (each pred/gt can only be matched once)
    matched_pairs = []
    for iou_score, pred_idx, gt_idx in all_matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            # Convert IoU to fuzzy score
            fuzzy_score = (iou_score - min_iou) / (1.0 - min_iou) * 0.9 + 0.1
            matched_pairs.append((pred_idx, gt_idx, fuzzy_score))
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)
            total_fuzzy_score += fuzzy_score

    # Calculate multi-box metrics
    n_pred = len(pred_coords)
    n_gt = len(gt_coords)
    n_matched = len(matched_pairs)
    
    # Fuzzy precision: sum of fuzzy scores / total predictions
    fuzzy_precision = total_fuzzy_score / n_pred if n_pred > 0 else 0.0

    # Fuzzy recall: sum of fuzzy scores / total ground truth
    fuzzy_recall = total_fuzzy_score / n_gt if n_gt > 0 else 0.0

    # Fuzzy F1 score (harmonic mean)
    if fuzzy_precision + fuzzy_recall > 0:
        fuzzy_f1 = (2 * fuzzy_precision * fuzzy_recall) / (fuzzy_precision + fuzzy_recall)
    else:
        fuzzy_f1 = 0.0

    # Penalty for unmatched boxes
    unmatched_pred_penalty = (n_pred - n_matched) * 0.1  # Small penalty per false positive
    unmatched_gt_penalty = (n_gt - n_matched) * 0.1     # Small penalty per false negative

    # Final score: F1 with penalties
    final_score = fuzzy_f1 - (unmatched_pred_penalty + unmatched_gt_penalty) / max(n_pred, n_gt)

    return max(0.0, min(1.0, final_score))  # Clamp to [0, 1]

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """VERL-compliant reward function for VinDR-CXR multi-box grounding."""
    if data_source != "vindr_grpo":
        return 0.0

    try:
        pred_coords = extract_predicted_coords(solution_str)

        # Handle ground truth as dict (new format from preprocessing)
        if isinstance(ground_truth, dict):
            gt_coords = ground_truth.get("coordinates", [])
            is_no_finding = ground_truth.get("has_no_finding", False)
            print(f"DEBUG: gt_coords = {gt_coords}, type = {type(gt_coords)}")
            if gt_coords:
                print(f"DEBUG: First gt coord = {gt_coords[0]}, type = {type(gt_coords[0])}")
        # Fallback: handle ground truth as list (old format)
        elif isinstance(ground_truth, list):
            gt_coords = ground_truth
            is_no_finding = len(gt_coords) == 0
            print(f"DEBUG: gt_coords (list) = {gt_coords}")
        else:
            print(f"DEBUG: Unexpected ground_truth type: {type(ground_truth)}, value: {ground_truth}")
            return 0.1  # Fallback for unexpected format

        # Handle "No finding" cases
        if is_no_finding:
            no_finding_mentioned = any(phrase in solution_str.lower()
                                     for phrase in ["no finding", "no abnormalities", "clear"])
            if not pred_coords and no_finding_mentioned:
                return 1.0  # Perfect "no finding" case
            return 0.1 if pred_coords else 0.7  # Penalty for false positives

        # Handle empty coordinates (parsing errors)
        if len(gt_coords) == 0:
            return 0.1 if solution_str.strip() else 0.0

        # Calculate fuzzy multi-box score (main metric)
        multibox_score = calculate_fuzzy_multibox_score(pred_coords, gt_coords)

        # Format bonuses
        format_bonus = 0.0
        if "<think>" in solution_str.lower():
            format_bonus += 0.05
        if "<answer>" in solution_str.lower():
            format_bonus += 0.05

        return min(1.0, max(0.0, multibox_score + format_bonus))

    except Exception as e:
        print(f"VinDR reward error: {e}")
        return 0.0