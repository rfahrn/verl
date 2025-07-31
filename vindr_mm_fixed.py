# Copyright 2025 – Apache-2.0
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs          # optional HDFS sync

DATA_SOURCE = "vindr_grpo"        # arbitrary tag used by RewardManager
ABILITY     = "medical_grounding"
PROMPT_SUFFIX = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
TARGET_TOKENS = 1024
MAX_PIXELS    = TARGET_TOKENS * 28 * 28    # ≈0.8 Mpx  (896×896)

def extract_coordinates(solution_str):
    """Extract coordinates from VinDR ground truth for reward function."""
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, solution_str)

    if not coordinates:
        if "no abnormalities" in solution_str.lower() or "no finding" in solution_str.lower():
            return "NO_FINDING"
        return "PARSING_ERROR"

    # Convert to coordinate list for mAP calculation
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]

def make_map_fn(split):  # ← FIXED: Added "make_" prefix
    def proc(example, idx):
        # Skip if image doesn't exist
        if not example.get("image") or not os.path.exists(example["image"]):
            return None

        img_entry = {
            "image": f"file://{example['image']}",
            "resized_height": 512,           # 24×28px  →  576 tokens
            "resized_width":  512,
            "image_processor_type": "qwen2_vl"
        }

        # user prompt (keep the <image> token so Qwen knows where the pixel goes)
        prompt_msg = {
            "role": "user",
            "content": example["conversations"][0]["value"] + PROMPT_SUFFIX
        }

        # Extract coordinates for reward function
        ground_truth_coords = extract_coordinates(example["conversations"][1]["value"])

        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : [prompt_msg],              # same schema as Geo3K example
            "images"      : [img_entry],
            "ability"     : ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_coords      # coordinates for mAP calculation
            },
            "extra_info"  : {                          # anything you want to keep
                "id":    example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "labels": example.get("labels", []),
                "raw_answer": example["conversations"][1]["value"]
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.8):
    with open(json_path) as f:          # ← use the file path
        rows = json.load(f)             # ← module 'json' is still accessible

    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n  = int(len(ds) * train_ratio)

    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    val_ds   = ds.select(range(n, len(ds))).map(make_map_fn("val"),   with_indices=True, num_proc=8)

    # Filter out None entries (missing images)
    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds   = val_ds.filter(lambda x: x is not None)

    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet  (os.path.join(local_dir, "val.parquet"))

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

    print(f"✅ VinDR preprocessing completed! Train: {len(train_ds)}, Val: {len(val_ds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")              # ← new name
    parser.add_argument("--local_dir", default="~/data/vindr_grpo")
    parser.add_argument("--hdfs_dir",  default=None)
    main(**vars(parser.parse_args()))