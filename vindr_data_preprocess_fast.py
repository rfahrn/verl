# Copyright 2025 – Apache-2.0
"""
Fast VinDR data preprocessing for development.
Creates smaller images and limits dataset size for rapid iteration.
"""
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "vindr_grpo"
ABILITY = "medical_grounding"
PROMPT_SUFFIX = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

# FAST DEVELOPMENT SETTINGS
TARGET_TOKENS = 256  # Much smaller for speed
MAX_PIXELS = TARGET_TOKENS * 16 * 16  # 256x256 images (~65k pixels)
MAX_SAMPLES = 1000  # Limit dataset size

def extract_coordinates(solution_str):
    """Extract coordinates from VinDR ground truth for reward function."""
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, solution_str)
    if not coordinates:
        if "no abnormalities" in solution_str.lower() or "no finding" in solution_str.lower():
            return []
        return []
    
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]

def make_map_fn(split):
    def proc(example, idx):
        # Skip if image doesn't exist
        if not example.get("image") or not os.path.exists(example["image"]):
            return None
        
        # FAST: Smaller images for development
        img_entry = {
            "image": f"file://{example['image']}",
            "resized_height": 256,  # Much smaller: 16×16px → 256 tokens
            "resized_width": 256,
            "image_processor_type": "qwen2_vl"
        }
        
        # User prompt
        prompt_msg = {
            "role": "user",
            "content": example["conversations"][0]["value"] + PROMPT_SUFFIX
        }
        
        # Extract data for reward function
        raw_answer = example["conversations"][1]["value"]
        ground_truth_coords = extract_coordinates(raw_answer)
        labels = example.get("labels") or []
        
        # Ground truth structure
        has_no_finding = ("no abnormalities" in raw_answer.lower() or 
                         "no finding" in raw_answer.lower() or 
                         "No finding" in labels)
        
        ground_truth_data = {
            "coordinates": ground_truth_coords,
            "has_no_finding": has_no_finding,
            "raw_answer": raw_answer,
            "labels": labels  # Add labels for reward function
        }
        
        return {
            "data_source": DATA_SOURCE,
            "prompt": [prompt_msg],
            "images": [img_entry],
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_data
            },
            "extra_info": {
                "id": example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "labels": labels,
                "coord_count": len(ground_truth_coords)
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.8):
    with open(json_path) as f:
        rows = json.load(f)
    
    print(f"📊 Original dataset: {len(rows)} samples")
    
    # FAST: Limit dataset size for development
    if len(rows) > MAX_SAMPLES:
        print(f"⚡ FAST MODE: Limiting to {MAX_SAMPLES} samples for development")
        rows = rows[:MAX_SAMPLES]
    
    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n = int(len(ds) * train_ratio)
    
    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    val_ds = ds.select(range(n, len(ds))).map(make_map_fn("val"), with_indices=True, num_proc=8)
    
    # Filter out None entries (missing images)
    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds = val_ds.filter(lambda x: x is not None)
    
    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(local_dir, "val.parquet"))
    
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
    
    print(f"⚡ FAST VinDR preprocessing completed!")
    print(f"   📊 Train: {len(train_ds)} samples")
    print(f"   📊 Val: {len(val_ds)} samples")
    print(f"   🖼️  Image size: 256x256 (fast)")
    print(f"   🎯 Tokens per image: ~256 (vs ~16k before)")
    print(f"   ⚡ Expected speedup: ~60x faster")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--local_dir", default="~/data/vindr_grpo_fast")
    parser.add_argument("--hdfs_dir", default=None)
    main(**vars(parser.parse_args()))