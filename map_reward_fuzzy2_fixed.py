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
        print(f"DEBUG: IoU unpacking error - box1: {box1} (type: {type(box1)}), box2: {box2} (type: {type(box2)})")
        return 0.0

    # Ensure coordinates are in correct order (x1 < x2, y1 < y2)
    x1_1, x2_1 = min(x1_1, x2_1), max(x1_1, x2_1)
    y1_1, y2_1 = min(y1_1, y2_1), max(y1_1, y2_1)
    x1_2, x2_2 = min(x1_2, x2_2), max(x1_2, x2_2)
    y1_2, y2_2 = min(y1_2, y2_2), max(y1_2, y2_2)

    # Calculate intersection
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
    """Extract coordinates from model response, similar to dataset creation."""
    # First try to extract from <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL | re.IGNORECASE)
    text = answer_match.group(1).strip() if answer_match else response_str.strip()
    
    # More robust coordinate pattern - matches the one used in dataset creation
    coord_patterns = [
        r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]',  # [0.34, 0.53, 0.81, 0.66]
        r'\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)',  # (0.34, 0.53, 0.81, 0.66)
        r'([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)',      # 0.34, 0.53, 0.81, 0.66
    ]
    
    coordinates = []
    for pattern in coord_patterns:
        matches = re.findall(pattern, text)
        if matches:
            coordinates.extend(matches)
            break  # Use first successful pattern
    
    try:
        result = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]
        print(f"DEBUG: Extracted {len(result)} coordinate boxes from response")
        return result
    except (ValueError, TypeError) as e:
        print(f"DEBUG: Coordinate parsing error - raw matches: {coordinates}, error: {e}")
        return []

def calculate_fuzzy_multibox_score(pred_coords, gt_coords, min_iou=0.1):
    """Calculate fuzzy multi-box score with optimal assignment."""
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect "No finding" case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth

    print(f"DEBUG: Scoring {len(pred_coords)} pred vs {len(gt_coords)} gt boxes")

    # Create IoU matrix: pred_boxes x gt_boxes
    iou_matrix = []
    for i, pred_box in enumerate(pred_coords):
        iou_row = []
        for j, gt_box in enumerate(gt_coords):
            iou = calculate_iou(pred_box, gt_box)
            iou_row.append(iou)
        iou_matrix.append(iou_row)
        print(f"DEBUG: Pred box {i} IoUs: {[f'{iou:.4f}' for iou in iou_row]}")

    # Greedy assignment (can be replaced with Hungarian algorithm for optimal)
    assigned_gt = set()
    assigned_pred = set()
    total_fuzzy_score = 0.0

    # Gather all matches above threshold
    all_matches = []
    for pred_idx, iou_row in enumerate(iou_matrix):
        for gt_idx, iou_score in enumerate(iou_row):
            if iou_score >= min_iou:
                all_matches.append((iou_score, pred_idx, gt_idx))

    # Sort by IoU descending (best matches first)
    all_matches.sort(key=lambda x: x[0], reverse=True)

    # Assign matches greedily
    matched_pairs = []
    for iou_score, pred_idx, gt_idx in all_matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            # Convert IoU to fuzzy score: IoU 0.1→0.1, IoU 1.0→1.0
            fuzzy_score = (iou_score - min_iou) / (1.0 - min_iou) * 0.9 + 0.1
            matched_pairs.append((pred_idx, gt_idx, fuzzy_score))
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)
            total_fuzzy_score += fuzzy_score
            print(f"DEBUG: Matched pred[{pred_idx}] ↔ gt[{gt_idx}], IoU={iou_score:.4f}, fuzzy={fuzzy_score:.4f}")

    # Calculate metrics
    n_pred = len(pred_coords)
    n_gt = len(gt_coords)
    n_matched = len(matched_pairs)

    fuzzy_precision = total_fuzzy_score / n_pred if n_pred > 0 else 0.0
    fuzzy_recall = total_fuzzy_score / n_gt if n_gt > 0 else 0.0

    if fuzzy_precision + fuzzy_recall > 0:
        fuzzy_f1 = (2 * fuzzy_precision * fuzzy_recall) / (fuzzy_precision + fuzzy_recall)
    else:
        fuzzy_f1 = 0.0

    # Penalties for unmatched boxes (less aggressive than before)
    unmatched_pred_penalty = (n_pred - n_matched) * 0.05
    unmatched_gt_penalty = (n_gt - n_matched) * 0.05

    # Final score: F1 with small penalties, clamped to [0, 1]
    penalty_factor = (unmatched_pred_penalty + unmatched_gt_penalty) / max(n_pred, n_gt, 1)
    final_score = fuzzy_f1 - penalty_factor
    
    print(f"DEBUG: F1={fuzzy_f1:.4f}, penalty_factor={penalty_factor:.4f}, final={final_score:.4f}")
    
    return max(0.0, min(1.0, final_score))

def format_reward(solution_str: str) -> float:
    """Reward for proper formatting with <think> and <answer> tags."""
    has_think = bool(re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))
    
    if has_think and has_answer:
        return 1.0
    elif has_answer:
        return 0.7
    else:
        return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function for VinDR-CXR multi-box grounding.
    
    Args:
        data_source: Dataset identifier (should be "vindr_grpo")
        solution_str: Model's response string
        ground_truth: Dict with 'coordinates' and 'has_no_finding' keys
        extra_info: Additional metadata (unused)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        # Extract predicted coordinates from model response
        pred_coords = extract_predicted_coords(solution_str)
        
        # Handle ground truth format (should be dict from dataset creation)
        if isinstance(ground_truth, dict):
            gt_coords = ground_truth.get("coordinates", [])
            is_no_finding = ground_truth.get("has_no_finding", False)
        elif isinstance(ground_truth, list):
            # Fallback for old format
            gt_coords = ground_truth
            is_no_finding = len(gt_coords) == 0
        else:
            print(f"DEBUG: Unexpected ground_truth format: {type(ground_truth)}")
            return 0.1

        print(f"DEBUG: Pred coords: {pred_coords}")
        print(f"DEBUG: GT coords: {gt_coords}")
        print(f"DEBUG: No finding: {is_no_finding}")

        # Handle "No finding" cases
        if is_no_finding:
            no_finding_phrases = ["no finding", "no abnormalities", "clear", "normal"]
            no_finding_mentioned = any(phrase in solution_str.lower() for phrase in no_finding_phrases)
            
            if not pred_coords and no_finding_mentioned:
                return 1.0  # Perfect no-finding case
            elif not pred_coords:
                return 0.7  # Correct prediction but missing explicit statement
            else:
                return 0.1  # False positive - predicted boxes when there are none

        # Main grounding accuracy
        if not gt_coords:
            return 0.1 if solution_str.strip() else 0.0
            
        grounding_score = calculate_fuzzy_multibox_score(pred_coords, gt_coords)
        
        # Format bonus (smaller than main score)
        format_score = format_reward(solution_str)
        format_bonus = format_score * 0.1  # Max 0.1 bonus
        
        # Combine scores
        total_score = grounding_score + format_bonus
        final_score = min(1.0, max(0.0, total_score))
        
        print(f"DEBUG: Grounding={grounding_score:.4f}, Format bonus={format_bonus:.4f}, Final={final_score:.4f}")
        
        return final_score

    except Exception as e:
        print(f"DEBUG: VinDR reward error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0