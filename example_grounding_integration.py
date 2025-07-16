#!/usr/bin/env python3
"""
Example integration of grounding tasks with VERL framework.
This demonstrates the complete workflow from data preparation to model training and evaluation.
"""

import json
import numpy as np
from typing import Dict, List, Any
from pathlib import Path

# Import VERL components
from verl.utils.dataset.vision_utils import process_image
from verl.utils.reward_score.grounding_reward import grounding_reward_function, compute_iou
from verl.utils.metric.utils import compute_metrics


def create_sample_grounding_data():
    """Create sample grounding dataset for demonstration."""
    
    sample_data = [
        {
            "id": "sample_001",
            "image_path": "images/parking_lot.jpg",
            "text_query": "Find the red car in the parking lot",
            "ground_truth_boxes": [
                {
                    "bbox": [100, 150, 300, 400],
                    "confidence": 1.0,
                    "class_name": "car",
                    "attributes": ["red", "sedan"]
                }
            ],
            "image_info": {
                "width": 640,
                "height": 480,
                "id": "sample_001"
            }
        },
        {
            "id": "sample_002", 
            "image_path": "images/street_scene.jpg",
            "text_query": "Locate the blue bicycle on the sidewalk",
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
                "height": 600,
                "id": "sample_002"
            }
        },
        {
            "id": "sample_003",
            "image_path": "images/multiple_objects.jpg", 
            "text_query": "Find all the people in the image",
            "ground_truth_boxes": [
                {
                    "bbox": [50, 100, 150, 400],
                    "confidence": 1.0,
                    "class_name": "person",
                    "attributes": ["adult", "standing"]
                },
                {
                    "bbox": [300, 120, 380, 350],
                    "confidence": 1.0,
                    "class_name": "person",
                    "attributes": ["adult", "walking"]
                }
            ],
            "image_info": {
                "width": 640,
                "height": 480,
                "id": "sample_003"
            }
        }
    ]
    
    return sample_data


def convert_to_verl_format(grounding_data: List[Dict]) -> List[Dict]:
    """Convert grounding dataset to VERL conversation format."""
    
    verl_data = []
    
    for item in grounding_data:
        # Format as conversation
        conversation = [
            {
                "from": "human",
                "value": f"<image>\n{item['text_query']}"
            },
            {
                "from": "assistant",
                "value": f"I can see the objects you're looking for. Here are the locations:\n"
            }
        ]
        
        # Add bounding box information to assistant response
        boxes_description = []
        for i, box in enumerate(item["ground_truth_boxes"]):
            bbox = box["bbox"]
            class_name = box["class_name"]
            attributes = box.get("attributes", [])
            
            attr_str = ", ".join(attributes) if attributes else ""
            desc = f"{class_name}"
            if attr_str:
                desc += f" ({attr_str})"
            
            desc += f" at coordinates [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
            boxes_description.append(desc)
        
        conversation[1]["value"] += "\n".join(boxes_description)
        
        verl_item = {
            "id": item["id"],
            "image": item["image_path"],
            "conversations": conversation,
            "ground_truth": {
                "bboxes": item["ground_truth_boxes"],
                "image_info": item["image_info"]
            }
        }
        
        verl_data.append(verl_item)
    
    return verl_data


def test_reward_function():
    """Test the grounding reward function with sample predictions."""
    
    # Sample ground truth
    ground_truth = {
        "bboxes": [
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
    
    # Test cases with different prediction quality
    test_cases = [
        {
            "name": "Perfect prediction",
            "prediction": "I can see a red car at coordinates [100, 150, 300, 400]",
            "expected_reward": 1.0
        },
        {
            "name": "Good prediction (high IoU)",
            "prediction": "The red car is located at [95, 145, 305, 405]",
            "expected_reward": 0.8  # High IoU
        },
        {
            "name": "Moderate prediction",
            "prediction": "I found a car at [120, 170, 280, 380]",
            "expected_reward": 0.3  # Moderate IoU
        },
        {
            "name": "Poor prediction",
            "prediction": "There's a car at [200, 300, 400, 500]",
            "expected_reward": -0.5  # Low IoU
        },
        {
            "name": "No detection",
            "prediction": "I cannot see any cars in the image",
            "expected_reward": -1.0
        },
        {
            "name": "Multiple detections",
            "prediction": "I see cars at [100, 150, 300, 400] and [500, 200, 600, 300]",
            "expected_reward": 0.9  # One correct, one false positive
        }
    ]
    
    print("Testing Grounding Reward Function:")
    print("=" * 50)
    
    for test_case in test_cases:
        reward = grounding_reward_function(
            test_case["prediction"],
            json.dumps(ground_truth),
            reward_type="iou",
            iou_threshold=0.5
        )
        
        print(f"Test: {test_case['name']}")
        print(f"Prediction: {test_case['prediction']}")
        print(f"Reward: {reward:.3f}")
        print(f"Expected: ~{test_case['expected_reward']}")
        print("-" * 30)


def compute_map_example():
    """Demonstrate mAP computation for grounding evaluation."""
    
    # Sample predictions and ground truths
    predictions = [
        "Car at [100, 150, 300, 400]",  # Perfect match
        "Bicycle at [190, 95, 410, 305]",  # Good match 
        "Person at [45, 95, 155, 405] and person at [295, 115, 385, 355]"  # Perfect match
    ]
    
    ground_truths = [
        {
            "bboxes": [{"bbox": [100, 150, 300, 400], "class_name": "car"}],
            "image_info": {"width": 640, "height": 480}
        },
        {
            "bboxes": [{"bbox": [200, 100, 400, 300], "class_name": "bicycle"}],
            "image_info": {"width": 800, "height": 600}
        },
        {
            "bboxes": [
                {"bbox": [50, 100, 150, 400], "class_name": "person"},
                {"bbox": [300, 120, 380, 350], "class_name": "person"}
            ],
            "image_info": {"width": 640, "height": 480}
        }
    ]
    
    print("\nTesting mAP Computation:")
    print("=" * 50)
    
    # Compute individual IoU scores
    total_reward = 0.0
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        reward = grounding_reward_function(
            pred,
            json.dumps(gt),
            reward_type="iou",
            iou_threshold=0.5
        )
        total_reward += reward
        print(f"Sample {i+1}: {reward:.3f}")
    
    avg_reward = total_reward / len(predictions)
    print(f"Average IoU-based reward: {avg_reward:.3f}")
    
    # Compute mAP-style rewards
    print("\nmAP-style evaluation:")
    map_rewards = []
    for pred, gt in zip(predictions, ground_truths):
        map_reward = grounding_reward_function(
            pred,
            json.dumps(gt),
            reward_type="map",
            iou_thresholds=[0.5, 0.75, 0.9]
        )
        map_rewards.append(map_reward)
        print(f"mAP reward: {map_reward:.3f}")
    
    avg_map_reward = np.mean(map_rewards)
    print(f"Average mAP reward: {avg_map_reward:.3f}")


def create_training_config():
    """Create training configuration for grounding tasks."""
    
    config = {
        "model": {
            "name": "qwen2.5-vl-7b",
            "vision_model": "qwen2.5-vl-7b",
            "max_image_size": 1024,
            "max_sequence_length": 2048
        },
        "dataset": {
            "train_path": "data/grounding_train.json",
            "val_path": "data/grounding_val.json",
            "test_path": "data/grounding_test.json",
            "image_folder": "data/images/",
            "conversation_template": "qwen"
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-5,
            "num_epochs": 10,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "reward": {
            "function": "grounding",
            "reward_type": "iou",
            "iou_threshold": 0.5,
            "reward_scaling": "linear",
            "normalize_coords": True
        },
        "evaluation": {
            "eval_steps": 500,
            "eval_batch_size": 8,
            "metrics": ["mAP@0.5", "mAP@0.75", "mAP", "avg_iou"],
            "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        },
        "rl": {
            "algorithm": "ppo",
            "value_loss_coeff": 0.5,
            "entropy_coeff": 0.01,
            "clip_ratio": 0.2,
            "target_kl": 0.01,
            "gae_lambda": 0.95,
            "discount_factor": 0.99
        }
    }
    
    return config


def integration_example():
    """Complete integration example."""
    
    print("VERL Grounding Integration Example")
    print("=" * 50)
    
    # 1. Create sample data
    print("1. Creating sample grounding data...")
    sample_data = create_sample_grounding_data()
    print(f"Created {len(sample_data)} samples")
    
    # 2. Convert to VERL format
    print("\n2. Converting to VERL format...")
    verl_data = convert_to_verl_format(sample_data)
    
    # Save to file
    output_path = Path("grounding_dataset_verl.json")
    with open(output_path, 'w') as f:
        json.dump({"data": verl_data}, f, indent=2)
    print(f"Saved VERL dataset to {output_path}")
    
    # 3. Test reward function
    print("\n3. Testing reward function...")
    test_reward_function()
    
    # 4. Compute mAP example
    compute_map_example()
    
    # 5. Create training configuration
    print("\n5. Creating training configuration...")
    config = create_training_config()
    config_path = Path("grounding_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config to {config_path}")
    
    # 6. Show integration with VERL reward system
    print("\n6. Integration with VERL reward system:")
    print("Add to verl/utils/reward_score/__init__.py:")
    print("""
from .grounding_reward import grounding_reward_function

def compute_score(predict_str: str, ground_truth: str, data_source: str = "grounding", **kwargs):
    if data_source == "grounding":
        return grounding_reward_function(predict_str, ground_truth, **kwargs)
    # ... other data sources
    """)
    
    print("\n7. Usage in training script:")
    print("""
# In your training script:
from verl.utils.reward_score import compute_score

# Compute reward for a prediction
reward = compute_score(
    predict_str=model_output,
    ground_truth=json.dumps(ground_truth_data),
    data_source="grounding",
    reward_type="iou",
    iou_threshold=0.5
)
    """)


if __name__ == "__main__":
    integration_example()