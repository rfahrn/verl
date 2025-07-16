# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
import numpy as np
from typing import List, Dict, Any, Optional


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes in XYXY format.
    
    Args:
        box1: [x1, y1, x2, y2] - First bounding box
        box2: [x1, y1, x2, y2] - Second bounding box
    
    Returns:
        IoU value between 0 and 1
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # No intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Calculate areas
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def parse_bbox_from_response(response: str) -> List[List[float]]:
    """
    Extract bounding box coordinates from model response.
    
    Supports multiple formats:
    - [x1, y1, x2, y2]
    - <bbox>x1,y1,x2,y2</bbox>
    - coordinates: x1, y1, x2, y2
    
    Args:
        response: Model response string
    
    Returns:
        List of bounding boxes [[x1, y1, x2, y2], ...]
    """
    boxes = []
    
    # Pattern 1: [x1, y1, x2, y2] format
    pattern1 = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
    matches = re.findall(pattern1, response)
    for match in matches:
        box = [float(x) for x in match]
        boxes.append(box)
    
    # Pattern 2: <bbox>x1,y1,x2,y2</bbox> format
    pattern2 = r'<bbox>(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)</bbox>'
    matches = re.findall(pattern2, response)
    for match in matches:
        box = [float(x) for x in match]
        boxes.append(box)
    
    # Pattern 3: coordinates: x1, y1, x2, y2 format
    pattern3 = r'coordinates?\s*:?\s*(\d+(?:\.\d+)?),?\s*(\d+(?:\.\d+)?),?\s*(\d+(?:\.\d+)?),?\s*(\d+(?:\.\d+)?)'
    matches = re.findall(pattern3, response, re.IGNORECASE)
    for match in matches:
        box = [float(x) for x in match]
        boxes.append(box)
    
    return boxes


def normalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Normalize bounding box coordinates to [0, 1] range.
    
    Args:
        bbox: [x1, y1, x2, y2] in absolute coordinates
        image_width: Image width
        image_height: Image height
    
    Returns:
        Normalized bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return [
        x1 / image_width,
        y1 / image_height,
        x2 / image_width,
        y2 / image_height
    ]


def denormalize_bbox(bbox: List[float], image_width: int, image_height: int) -> List[float]:
    """
    Denormalize bounding box coordinates from [0, 1] range to absolute coordinates.
    
    Args:
        bbox: [x1, y1, x2, y2] in normalized coordinates
        image_width: Image width
        image_height: Image height
    
    Returns:
        Absolute bounding box [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    return [
        x1 * image_width,
        y1 * image_height,
        x2 * image_width,
        y2 * image_height
    ]


def compute_grounding_reward(
    prediction: str,
    ground_truth: Dict[str, Any],
    iou_threshold: float = 0.5,
    reward_scaling: str = "linear",
    class_specific: bool = False,
    normalize_coords: bool = True
) -> float:
    """
    Compute reward based on IoU between predicted and ground truth boxes.
    
    Args:
        prediction: Model prediction string containing bounding boxes
        ground_truth: Ground truth data containing bboxes and image info
        iou_threshold: IoU threshold for positive reward
        reward_scaling: Reward scaling method ('linear', 'quadratic', 'exponential')
        class_specific: Whether to use class-specific matching
        normalize_coords: Whether to normalize coordinates before comparison
    
    Returns:
        Reward value (typically between -1 and 1)
    """
    # Parse predicted boxes from response
    predicted_boxes = parse_bbox_from_response(prediction)
    
    # Extract ground truth information
    ground_truth_boxes = [box["bbox"] for box in ground_truth["bboxes"]]
    image_info = ground_truth.get("image_info", {})
    image_width = image_info.get("width", 1)
    image_height = image_info.get("height", 1)
    
    # Handle no detection case
    if not predicted_boxes:
        return -1.0 if ground_truth_boxes else 0.0
    
    # Handle no ground truth case (negative samples)
    if not ground_truth_boxes:
        return -0.5  # Penalty for false positive
    
    # Normalize coordinates if requested
    if normalize_coords:
        predicted_boxes = [normalize_bbox(box, image_width, image_height) for box in predicted_boxes]
        ground_truth_boxes = [normalize_bbox(box, image_width, image_height) for box in ground_truth_boxes]
    
    # Compute IoU for each ground truth box
    max_ious = []
    matched_predictions = set()
    
    for gt_idx, gt_box in enumerate(ground_truth_boxes):
        max_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx, pred_box in enumerate(predicted_boxes):
            if pred_idx in matched_predictions:
                continue
                
            # Class-specific matching if enabled
            if class_specific and len(ground_truth["bboxes"]) > gt_idx:
                gt_class = ground_truth["bboxes"][gt_idx].get("class_name", "object")
                # In practice, you'd need to extract class from prediction
                # For now, assume generic matching
            
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                best_pred_idx = pred_idx
        
        max_ious.append(max_iou)
        if best_pred_idx >= 0:
            matched_predictions.add(best_pred_idx)
    
    # Compute reward based on IoU values
    total_reward = 0.0
    
    for iou in max_ious:
        if iou >= iou_threshold:
            # Positive reward for correct detection
            if reward_scaling == "linear":
                reward = 1.0
            elif reward_scaling == "quadratic":
                reward = iou ** 2
            elif reward_scaling == "exponential":
                reward = np.exp(iou) - 1
            else:
                reward = 1.0
        else:
            # Scaled reward/penalty based on IoU
            if reward_scaling == "linear":
                reward = 2 * iou - 1  # Scale to [-1, 1]
            elif reward_scaling == "quadratic":
                reward = 2 * (iou ** 2) - 1
            elif reward_scaling == "exponential":
                reward = np.exp(iou) - np.exp(0.5)
            else:
                reward = 2 * iou - 1
        
        total_reward += reward
    
    # Normalize by number of ground truth boxes
    normalized_reward = total_reward / len(ground_truth_boxes)
    
    # Penalty for false positives (unmatched predictions)
    num_false_positives = len(predicted_boxes) - len(matched_predictions)
    if num_false_positives > 0:
        fp_penalty = 0.1 * num_false_positives
        normalized_reward -= fp_penalty
    
    # Clamp reward to reasonable range
    return max(-1.0, min(1.0, normalized_reward))


def compute_map_reward(
    prediction: str,
    ground_truth: Dict[str, Any],
    iou_thresholds: List[float] = [0.5, 0.75, 0.9]
) -> float:
    """
    Compute reward based on mAP-style evaluation.
    
    Args:
        prediction: Model prediction string
        ground_truth: Ground truth data
        iou_thresholds: List of IoU thresholds for evaluation
    
    Returns:
        mAP-based reward value
    """
    predicted_boxes = parse_bbox_from_response(prediction)
    ground_truth_boxes = [box["bbox"] for box in ground_truth["bboxes"]]
    
    if not predicted_boxes or not ground_truth_boxes:
        return 0.0
    
    # Compute precision at different IoU thresholds
    precisions = []
    
    for iou_threshold in iou_thresholds:
        correct_detections = 0
        
        for gt_box in ground_truth_boxes:
            max_iou = 0.0
            for pred_box in predicted_boxes:
                iou = compute_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            
            if max_iou >= iou_threshold:
                correct_detections += 1
        
        precision = correct_detections / len(ground_truth_boxes)
        precisions.append(precision)
    
    # Return average precision across thresholds
    return np.mean(precisions)


def grounding_reward_function(
    predict_str: str,
    ground_truth: str,
    reward_type: str = "iou",
    iou_threshold: float = 0.5,
    **kwargs
) -> float:
    """
    VERL-compatible reward function for grounding tasks.
    
    Args:
        predict_str: Model prediction string
        ground_truth: Ground truth data as JSON string
        reward_type: Type of reward computation ('iou', 'map')
        iou_threshold: IoU threshold for positive reward
        **kwargs: Additional parameters
    
    Returns:
        Reward value for the prediction
    """
    try:
        # Parse ground truth
        if isinstance(ground_truth, str):
            gt_data = json.loads(ground_truth)
        else:
            gt_data = ground_truth
        
        # Compute reward based on type
        if reward_type == "iou":
            return compute_grounding_reward(
                predict_str,
                gt_data,
                iou_threshold=iou_threshold,
                **kwargs
            )
        elif reward_type == "map":
            return compute_map_reward(
                predict_str,
                gt_data,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    except Exception as e:
        print(f"Error in grounding reward function: {e}")
        return -1.0  # Return penalty for errors


# Alias for backward compatibility
compute_score = grounding_reward_function