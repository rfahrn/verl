# Adapting Grounding Datasets for VERL Integration

## Overview

This guide provides comprehensive instructions for adapting your grounding task datasets (train/test) to work with VERL (Vision-Enhanced Reinforcement Learning), focusing on RadVLM data generation, IoU-based reward functions, and mAP evaluation.

## Table of Contents

1. [Understanding VERL's Vision-Language Architecture](#understanding-verls-vision-language-architecture)
2. [RadVLM Data Generation Pipeline](#radvlm-data-generation-pipeline)
3. [Dataset Format Requirements](#dataset-format-requirements)
4. [IoU-Based Reward Function Implementation](#iou-based-reward-function-implementation)
5. [mAP Evaluation Setup](#map-evaluation-setup)
6. [Integration Steps](#integration-steps)
7. [Best Practices](#best-practices)

## Understanding VERL's Vision-Language Architecture

VERL supports vision-language models through:
- **Vision Processing**: Handles images/videos via `verl.utils.dataset.vision_utils`
- **Multi-modal Data**: Processes image-text pairs for training
- **Reward Functions**: Supports custom reward computation for RL training
- **Evaluation Metrics**: Flexible metric computation framework

### Key Components:
- `verl.utils.dataset.vision_utils.process_image()` - Image preprocessing
- `verl.utils.dataset.vision_utils.process_video()` - Video preprocessing  
- `verl.utils.reward_score.*` - Reward function implementations
- `verl.utils.metric.utils` - Evaluation metrics

## RadVLM Data Generation Pipeline

### 1. Data Format Requirements

Your grounding dataset should follow this structure:

```python
# Example grounding data point
{
    "image": {
        "type": "image",
        "image": "path/to/image.jpg"  # or PIL Image object
    },
    "text_query": "Find the red car in the parking lot",
    "ground_truth_boxes": [
        {
            "bbox": [x1, y1, x2, y2],  # Coordinates in XYXY format
            "confidence": 1.0,
            "class_name": "car",
            "attributes": ["red", "sedan"]
        }
    ],
    "image_info": {
        "width": 640,
        "height": 480,
        "id": "image_001"
    }
}
```

### 2. Data Generation for RadVLM

Create a data preprocessing script:

```python
# data_preprocessing.py
import json
from PIL import Image
from verl.utils.dataset.vision_utils import process_image

def convert_to_verl_format(grounding_dataset):
    """Convert grounding dataset to VERL format"""
    converted_data = []
    
    for item in grounding_dataset:
        # Process image
        image = process_image(item["image"])
        
        # Create instruction following format
        instruction = f"Locate and describe the following in the image: {item['text_query']}"
        
        # Format ground truth for reward computation
        ground_truth = {
            "bboxes": item["ground_truth_boxes"],
            "image_info": item["image_info"]
        }
        
        converted_item = {
            "image": image,
            "instruction": instruction,
            "ground_truth": ground_truth,
            "id": item["image_info"]["id"]
        }
        
        converted_data.append(converted_item)
    
    return converted_data
```

## Dataset Format Requirements

### Training Dataset Structure

```python
# train_dataset.json
{
    "data": [
        {
            "id": "train_001",
            "image": "/path/to/image1.jpg",
            "conversations": [
                {
                    "from": "human",
                    "value": "Find the red car in the parking lot"
                },
                {
                    "from": "assistant", 
                    "value": "The red car is located at coordinates [x1, y1, x2, y2] in the parking lot."
                }
            ],
            "ground_truth_boxes": [
                {
                    "bbox": [100, 150, 300, 400],
                    "confidence": 1.0,
                    "class_name": "car",
                    "attributes": ["red"]
                }
            ],
            "image_info": {
                "width": 640,
                "height": 480
            }
        }
    ]
}
```

### Test Dataset Structure

```python
# test_dataset.json
{
    "data": [
        {
            "id": "test_001", 
            "image": "/path/to/test_image1.jpg",
            "query": "Locate the blue bicycle",
            "ground_truth_boxes": [
                {
                    "bbox": [200, 100, 400, 300],
                    "confidence": 1.0,
                    "class_name": "bicycle",
                    "attributes": ["blue"]
                }
            ],
            "image_info": {
                "width": 800,
                "height": 600
            }
        }
    ]
}
```

## IoU-Based Reward Function Implementation

### 1. Create Custom Reward Function

```python
# grounding_reward_function.py
import torch
import numpy as np
from typing import List, Dict, Any

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two bounding boxes in XYXY format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def parse_bbox_from_response(response: str) -> List[List[float]]:
    """Extract bounding box coordinates from model response"""
    import re
    
    # Pattern to match bbox coordinates [x1, y1, x2, y2]
    pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
    matches = re.findall(pattern, response)
    
    boxes = []
    for match in matches:
        box = [float(x) for x in match]
        boxes.append(box)
    
    return boxes

def compute_grounding_reward(
    prediction: str,
    ground_truth: Dict[str, Any],
    iou_threshold: float = 0.5
) -> float:
    """Compute reward based on IoU between predicted and ground truth boxes"""
    
    # Parse predicted boxes from response
    predicted_boxes = parse_bbox_from_response(prediction)
    ground_truth_boxes = [box["bbox"] for box in ground_truth["bboxes"]]
    
    if not predicted_boxes:
        return -1.0  # Penalty for no detection
    
    # Compute maximum IoU for each ground truth box
    max_ious = []
    for gt_box in ground_truth_boxes:
        max_iou = 0.0
        for pred_box in predicted_boxes:
            iou = compute_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
        max_ious.append(max_iou)
    
    # Compute reward based on IoU
    reward = 0.0
    for iou in max_ious:
        if iou >= iou_threshold:
            reward += 1.0  # Positive reward for correct detection
        else:
            reward += iou - 0.5  # Scaled reward based on IoU
    
    # Normalize by number of ground truth boxes
    reward = reward / len(ground_truth_boxes)
    
    return reward

# Integration with VERL reward system
def grounding_reward_function(predict_str: str, ground_truth: str, **kwargs) -> float:
    """VERL-compatible reward function for grounding tasks"""
    import json
    
    # Parse ground truth
    gt_data = json.loads(ground_truth)
    
    # Compute reward
    reward = compute_grounding_reward(predict_str, gt_data)
    
    return reward
```

### 2. Register Reward Function

```python
# Add to verl/utils/reward_score/__init__.py
from .grounding_reward import grounding_reward_function

def compute_score(predict_str: str, ground_truth: str, data_source: str = "grounding", **kwargs):
    if data_source == "grounding":
        return grounding_reward_function(predict_str, ground_truth, **kwargs)
    # ... other data sources
```

## mAP Evaluation Setup

### 1. mAP Computation Implementation

```python
# map_evaluation.py
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation"""
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(precisions[recalls >= t]) if np.sum(recalls >= t) > 0 else 0.0
        ap += p / 11.0
    return ap

def compute_map(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    class_names: List[str] = None
) -> Dict[str, float]:
    """Compute mAP for grounding task"""
    
    if class_names is None:
        # Extract all unique class names
        class_names = set()
        for gt in ground_truths:
            for box in gt["bboxes"]:
                class_names.add(box["class_name"])
        class_names = sorted(list(class_names))
    
    # Compute AP for each IoU threshold and class
    aps = defaultdict(dict)
    
    for iou_thresh in iou_thresholds:
        for class_name in class_names:
            # Filter predictions and ground truths for this class
            class_predictions = []
            class_ground_truths = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                # Filter predicted boxes for this class
                pred_boxes = [box for box in pred.get("bboxes", []) 
                             if box["class_name"] == class_name]
                
                # Filter ground truth boxes for this class
                gt_boxes = [box for box in gt["bboxes"] 
                           if box["class_name"] == class_name]
                
                if pred_boxes:
                    class_predictions.extend([
                        {
                            "image_id": i,
                            "bbox": box["bbox"],
                            "confidence": box["confidence"],
                            "class_name": class_name
                        }
                        for box in pred_boxes
                    ])
                
                if gt_boxes:
                    class_ground_truths.extend([
                        {
                            "image_id": i,
                            "bbox": box["bbox"],
                            "class_name": class_name
                        }
                        for box in gt_boxes
                    ])
            
            # Compute precision-recall curve
            if not class_predictions or not class_ground_truths:
                aps[iou_thresh][class_name] = 0.0
                continue
            
            # Sort predictions by confidence
            class_predictions.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Compute matches
            tp = np.zeros(len(class_predictions))
            fp = np.zeros(len(class_predictions))
            
            # Group ground truths by image
            gt_by_image = defaultdict(list)
            for gt in class_ground_truths:
                gt_by_image[gt["image_id"]].append(gt)
            
            for i, pred in enumerate(class_predictions):
                image_id = pred["image_id"]
                pred_box = pred["bbox"]
                
                # Find best matching ground truth
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt in enumerate(gt_by_image[image_id]):
                    iou = compute_iou(pred_box, gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                # Check if match is good enough
                if best_iou >= iou_thresh:
                    tp[i] = 1
                    # Remove matched ground truth to avoid double counting
                    gt_by_image[image_id].pop(best_gt_idx)
                else:
                    fp[i] = 1
            
            # Compute precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(class_ground_truths)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP
            ap = compute_ap(recalls, precisions)
            aps[iou_thresh][class_name] = ap
    
    # Compute mAP across classes and IoU thresholds
    results = {}
    
    # mAP for each IoU threshold
    for iou_thresh in iou_thresholds:
        class_aps = list(aps[iou_thresh].values())
        results[f"mAP@{iou_thresh}"] = np.mean(class_aps) if class_aps else 0.0
    
    # Overall mAP (average across IoU thresholds)
    all_aps = []
    for iou_thresh in iou_thresholds:
        class_aps = list(aps[iou_thresh].values())
        if class_aps:
            all_aps.extend(class_aps)
    
    results["mAP"] = np.mean(all_aps) if all_aps else 0.0
    
    return results
```

### 2. Integration with VERL Evaluation

```python
# Add to verl/utils/metric/utils.py
def compute_grounding_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """Compute mAP and other grounding metrics"""
    import json
    
    # Parse predictions and ground truths
    parsed_predictions = []
    parsed_ground_truths = []
    
    for pred_str, gt_str in zip(predictions, ground_truths):
        # Parse prediction
        pred_boxes = parse_bbox_from_response(pred_str)
        parsed_pred = {
            "bboxes": [
                {
                    "bbox": box,
                    "confidence": 1.0,  # Default confidence
                    "class_name": "object"  # Default class
                }
                for box in pred_boxes
            ]
        }
        parsed_predictions.append(parsed_pred)
        
        # Parse ground truth
        gt_data = json.loads(gt_str)
        parsed_ground_truths.append(gt_data)
    
    # Compute mAP
    map_results = compute_map(parsed_predictions, parsed_ground_truths)
    
    # Compute additional metrics
    total_iou = 0.0
    total_detections = 0
    
    for pred, gt in zip(parsed_predictions, parsed_ground_truths):
        pred_boxes = [box["bbox"] for box in pred["bboxes"]]
        gt_boxes = [box["bbox"] for box in gt["bboxes"]]
        
        for gt_box in gt_boxes:
            max_iou = 0.0
            for pred_box in pred_boxes:
                iou = compute_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
            total_iou += max_iou
            total_detections += 1
    
    # Add average IoU
    map_results["avg_iou"] = total_iou / total_detections if total_detections > 0 else 0.0
    
    return map_results
```

## Integration Steps

### 1. Dataset Preparation

```python
# prepare_grounding_dataset.py
import json
from pathlib import Path

def prepare_grounding_dataset(
    input_data_path: str,
    output_train_path: str,
    output_test_path: str,
    train_split: float = 0.8
):
    """Prepare grounding dataset for VERL training"""
    
    # Load original dataset
    with open(input_data_path, 'r') as f:
        data = json.load(f)
    
    # Convert to VERL format
    converted_data = convert_to_verl_format(data)
    
    # Split train/test
    split_idx = int(len(converted_data) * train_split)
    train_data = converted_data[:split_idx]
    test_data = converted_data[split_idx:]
    
    # Save processed datasets
    with open(output_train_path, 'w') as f:
        json.dump({"data": train_data}, f, indent=2)
    
    with open(output_test_path, 'w') as f:
        json.dump({"data": test_data}, f, indent=2)
    
    print(f"Prepared {len(train_data)} training samples")
    print(f"Prepared {len(test_data)} test samples")

if __name__ == "__main__":
    prepare_grounding_dataset(
        input_data_path="path/to/your/grounding_dataset.json",
        output_train_path="data/grounding_train.json",
        output_test_path="data/grounding_test.json"
    )
```

### 2. Training Configuration

```python
# grounding_config.py
from verl.base_config import BaseConfig

class GroundingConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        
        # Model configuration
        self.model_name = "qwen2.5-vl-7b"
        self.vision_model_name = "qwen2.5-vl-7b"
        
        # Dataset configuration
        self.train_data_path = "data/grounding_train.json"
        self.test_data_path = "data/grounding_test.json"
        
        # Reward function configuration
        self.reward_function = "grounding"
        self.iou_threshold = 0.5
        
        # Training configuration
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.num_epochs = 10
        self.gradient_accumulation_steps = 4
        
        # Evaluation configuration
        self.eval_steps = 500
        self.eval_metrics = ["mAP", "avg_iou"]
        
        # RL configuration
        self.rl_algorithm = "ppo"
        self.value_loss_coeff = 0.5
        self.entropy_coeff = 0.01
```

### 3. Training Script

```python
# train_grounding_model.py
import torch
from verl.trainer.ppo.ray_trainer import PPOTrainer
from verl.utils.dataset.rl_dataset import RLDataset
from grounding_config import GroundingConfig

def main():
    # Load configuration
    config = GroundingConfig()
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Load datasets
    train_dataset = RLDataset(config.train_data_path)
    test_dataset = RLDataset(config.test_data_path)
    
    # Start training
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        num_epochs=config.num_epochs
    )

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Data Quality
- Ensure accurate bounding box annotations
- Validate coordinate formats (XYXY vs XYWH)
- Include diverse object scales and scenes
- Balance positive and negative examples

### 2. Reward Function Design
- Use IoU thresholds appropriate for your task
- Consider class-specific rewards
- Implement progressive reward scaling
- Add penalties for hallucination

### 3. Evaluation Strategy
- Use multiple IoU thresholds for mAP
- Report class-specific performance
- Include qualitative evaluation samples
- Monitor overfitting with validation set

### 4. Model Training
- Start with smaller learning rates
- Use gradient clipping for stability
- Implement early stopping
- Save regular checkpoints

### 5. Debugging Tips
- Visualize predicted vs ground truth boxes
- Check reward function behavior
- Monitor training curves
- Test with simple examples first

## Example Usage

```bash
# 1. Prepare dataset
python prepare_grounding_dataset.py

# 2. Train model
python train_grounding_model.py

# 3. Evaluate model
python evaluate_grounding_model.py --model_path checkpoints/best_model.pt
```

This guide provides a comprehensive framework for adapting your grounding datasets for VERL integration. The key is to properly format your data, implement robust IoU-based reward functions, and set up comprehensive mAP evaluation metrics.