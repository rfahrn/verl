# Copyright 2025 – Apache-2.0
"""
Preprocess the VinDR-CXR dataset to parquet format for GRPO training
Follows VERL patterns exactly - simple and focused on localization grounding
"""

import argparse
import os
import re
import json
import datasets

from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    """
    Extract coordinates from the VinDR ground truth response.
    Focus on localization - coordinates are the key for grounding task.
    """
    # Pattern to match coordinates like [0.48, 0.34, 0.59, 0.43]
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, solution_str)
    
    if not coordinates:
        # If no coordinates found, check for "No finding" cases
        if "no abnormalities" in solution_str.lower() or "no finding" in solution_str.lower():
            return "NO_FINDING"
        else:
            return "PARSING_ERROR"
    
    # Convert to standardized format for reward function
    coord_list = []
    for x1, y1, x2, y2 in coordinates:
        coord_list.append([float(x1), float(y1), float(x2), float(y2)])
    
    return coord_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to VinDR-CXR JSON file")
    parser.add_argument("--local_dir", default="~/data/vindr_grpo")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "vindr_grpo"

    # Load VinDR JSON data
    with open(args.json_path) as f:
        rows = json.load(f)

    # Convert to HuggingFace dataset format
    dataset = datasets.Dataset.from_list(rows)
    
    # Simple train/test split (you can adjust the ratio)
    dataset = dataset.train_test_split(test_size=0.001, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Your tutor's exact prompt suffix
    instruction_following = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # Skip if image doesn't exist
            img_path = example.get("image", "")
            if not img_path or not os.path.exists(img_path):
                return None
            
            # Get the original question from conversations
            question_raw = example["conversations"][0]["value"]
            
            # Add instruction following suffix (your tutor's requirement)
            question = question_raw + instruction_following
            
            # Get the ground truth answer
            answer_raw = example["conversations"][1]["value"]
            
            # Extract solution (coordinates) for reward function
            solution = extract_solution(answer_raw)
            
            # Image entry for multimodal
            images = [{
                "image": f"file://{img_path}",
                "resized_height": 512,
                "resized_width": 512,
                "image_processor_type": "qwen2_vl"
            }]
            
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "images": images,
                "ability": "medical_grounding",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "labels": example.get("labels", []),
                    "image_path": img_path,
                    "id": example.get("id", f"{split}_{idx}")
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    # Filter out None entries (missing images)
    train_dataset = train_dataset.filter(lambda x: x is not None)
    test_dataset = test_dataset.filter(lambda x: x is not None)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print(f"✅ VinDR preprocessing completed!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Focus: Localization grounding with coordinates extraction")