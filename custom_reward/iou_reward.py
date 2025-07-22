import re
import json
import numpy as np
from typing import List, Tuple, Union, Any

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info=None) -> float:
    """
    Compute IOU reward for grounding tasks.
    
    Params follow veRL's RewardManager contract:
    - solution_str: detokenized LLM output for one sample
    - ground_truth: list of ground truth bounding boxes in format [[x1, y1, x2, y2], ...]
    - extra_info: additional information (not used currently)
    
    Returns:
    - float: IOU score between 0 and 1
    """
    # Extract bounding boxes from the answer section
    predicted_boxes = extract_bounding_boxes_from_answer(solution_str)
    
    if not predicted_boxes:
        return 0.0
    
    if not ground_truth or not isinstance(ground_truth, list):
        return 0.0
    
    # Compute maximum IOU between any predicted box and any ground truth box
    max_iou = 0.0
    for pred_box in predicted_boxes:
        for gt_box in ground_truth:
            iou = compute_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
    
    return max_iou

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
    # This regex matches patterns like [0.1, 0.2, 0.3, 0.4] or [0.1,0.2,0.3,0.4]
    box_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    matches = re.findall(box_pattern, answer_content)
    
    boxes = []
    for match in matches:
        try:
            x1, y1, x2, y2 = [float(coord) for coord in match]
            boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
    
    return boxes

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    
    Args:
    - box1, box2: [x1, y1, x2, y2] format where (x1,y1) is top-left, (x2,y2) is bottom-right
    
    Returns:
    - float: IoU score between 0 and 1
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Ensure boxes are valid (x1 < x2, y1 < y2)
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
    
    # Calculate intersection area
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    # Avoid division by zero
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area