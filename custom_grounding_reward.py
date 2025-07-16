import re
import json
import numpy as np
from typing import List, Dict, Any

def extract_bounding_boxes(response: str) -> List[Dict[str, float]]:
    """Extract bounding boxes from model response"""
    # Pattern to match <click>x, y, width, height</click>
    pattern = r'<click>(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)</click>'
    matches = re.findall(pattern, response)
    
    boxes = []
    for match in matches:
        x, y, w, h = map(float, match)
        boxes.append({
            'x': x,
            'y': y, 
            'width': w,
            'height': h
        })
    
    return boxes

def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """Calculate IoU between two bounding boxes"""
    # Convert to x1, y1, x2, y2 format
    x1_1, y1_1 = box1['x'], box1['y']
    x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
    
    x1_2, y1_2 = box2['x'], box2['y']
    x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
    
    # Calculate intersection
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def grounding_reward_function(ground_truth: List[Dict[str, float]], 
                            generated_response: str) -> float:
    """Calculate IoU-based reward for grounding tasks"""
    predicted_boxes = extract_bounding_boxes(generated_response)
    
    if not predicted_boxes:
        return 0.0
    
    # Calculate max IoU for each predicted box
    max_iou = 0.0
    for pred_box in predicted_boxes:
        for gt_box in ground_truth:
            iou = calculate_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
    
    return max_iou

# Register reward function with VeRL
def get_reward_function():
    return grounding_reward_function