import re
import numpy as np
from typing import List, Tuple, Union, Any

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info=None) -> float:
    """
    Compute GIoU (Generalized Intersection over Union) reward for grounding tasks.
    
    GIoU addresses the main weakness of IoU: when bounding boxes don't overlap,
    IoU is always 0 and provides no gradient. GIoU provides meaningful scores
    even for non-overlapping boxes by considering the smallest enclosing box.
    
    Params follow veRL's RewardManager contract:
    - solution_str: detokenized LLM output for one sample
    - ground_truth: list of ground truth bounding boxes in format [[x1, y1, x2, y2], ...]
                   or empty list [] for "no finding" cases
    - extra_info: additional information (not used currently)
    
    Returns:
    - float: GIoU score between -1 and 1 (higher is better)
    """
    # Extract bounding boxes from the answer section
    predicted_boxes = extract_bounding_boxes_from_answer(solution_str)
    
    # Handle ground truth validation
    if not isinstance(ground_truth, list):
        return -1.0  # Worst possible score for invalid ground truth
    
    # Case 1: No ground truth boxes (e.g., "No finding" cases)
    if len(ground_truth) == 0:
        # If model correctly predicts no boxes, give high reward
        if len(predicted_boxes) == 0:
            # Check if the model explicitly says "no finding" or similar
            answer_content = extract_answer_content(solution_str)
            if answer_content and any(phrase in answer_content.lower() for phrase in [
                'no finding', 'no abnormalities', 'no lesions', 'clear', 'normal', 
                'no detectable', 'no visible', 'clean bill', 'unremarkable'
            ]):
                return 0.9  # High reward for correctly identifying no findings
            else:
                return 0.7  # Good reward for predicting no boxes without explicit statement
        else:
            # False positive: predicted boxes when there should be none
            # Use GIoU to penalize based on how far the predicted boxes are from "nowhere"
            # For false positives, we return a negative score proportional to the predicted box sizes
            total_predicted_area = sum(box_area(box) for box in predicted_boxes)
            # Normalize by image area (assuming normalized coordinates [0,1])
            normalized_penalty = min(0.5, total_predicted_area)  # Cap penalty at 0.5
            return -normalized_penalty
    
    # Case 2: Ground truth has boxes but no predictions (false negative)
    if len(predicted_boxes) == 0:
        # Penalty for missing detections - use negative score proportional to missed area
        total_gt_area = sum(box_area(box) for box in ground_truth)
        normalized_penalty = min(0.8, total_gt_area)  # Cap penalty at 0.8
        return -normalized_penalty
    
    # Case 3: Both ground truth and predictions have boxes - compute GIoU
    # For multiple boxes, we use the best GIoU score (matching strategy)
    max_giou = -1.0  # Start with worst possible GIoU
    
    for pred_box in predicted_boxes:
        for gt_box in ground_truth:
            giou = compute_giou(pred_box, gt_box)
            max_giou = max(max_giou, giou)
    
    return max_giou


def extract_answer_content(solution_str: str) -> str:
    """Extract content inside <answer> tags."""
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def extract_bounding_boxes_from_answer(solution_str: str) -> List[List[float]]:
    """
    Extract bounding boxes from the model's answer.
    Looks for patterns like [x1, y1, x2, y2] in the <answer> section.
    """
    # First extract the answer section
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
    if not answer_match:
        return []
    
    answer_content = answer_match.group(1)
    
    # Look for bounding box patterns like [x1, y1, x2, y2]
    box_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    matches = re.findall(box_pattern, answer_content)
    
    boxes = []
    for match in matches:
        try:
            x1, y1, x2, y2 = [float(coord) for coord in match]
            # Ensure valid box format (x1 < x2, y1 < y2)
            if x1 < x2 and y1 < y2:
                boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
    
    return boxes


def compute_giou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Generalized Intersection over Union (GIoU) of two bounding boxes.
    
    GIoU = IoU - |C \ (A ∪ B)| / |C|
    
    Where C is the smallest enclosing box that contains both A and B.
    
    Args:
    - box1, box2: [x1, y1, x2, y2] format where (x1,y1) is top-left, (x2,y2) is bottom-right
    
    Returns:
    - float: GIoU score between -1 and 1
    """
    if len(box1) != 4 or len(box2) != 4:
        return -1.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Ensure boxes are valid (x1 < x2, y1 < y2)
    if x1_1 >= x2_1 or y1_1 >= y2_1 or x1_2 >= x2_2 or y1_2 >= y2_2:
        return -1.0
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Calculate intersection area
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        intersection_area = 0.0
    else:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate individual box areas
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union_area = area1 + area2 - intersection_area
    
    # Calculate IoU
    if union_area <= 0:
        iou = 0.0
    else:
        iou = intersection_area / union_area
    
    # Calculate smallest enclosing box (C)
    x1_c = min(x1_1, x1_2)
    y1_c = min(y1_1, y1_2)
    x2_c = max(x2_1, x2_2)
    y2_c = max(y2_1, y2_2)
    
    # Calculate enclosing box area
    c_area = (x2_c - x1_c) * (y2_c - y1_c)
    
    # Calculate GIoU
    if c_area <= 0:
        return -1.0
    
    # GIoU = IoU - |C \ (A ∪ B)| / |C|
    # where |C \ (A ∪ B)| = c_area - union_area
    giou = iou - (c_area - union_area) / c_area
    
    return giou


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute standard Intersection over Union (IoU) of two bounding boxes.
    Kept for comparison purposes.
    
    Args:
    - box1, box2: [x1, y1, x2, y2] format where (x1,y1) is top-left, (x2,y2) is bottom-right
    
    Returns:
    - float: IoU score between 0 and 1
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Ensure boxes are valid
    if x1_1 >= x2_1 or y1_1 >= y2_1 or x1_2 >= x2_2 or y1_2 >= y2_2:
        return 0.0
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def box_area(box: List[float]) -> float:
    """Calculate the area of a bounding box."""
    if len(box) != 4:
        return 0.0
    
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    return (x2 - x1) * (y2 - y1)