# Copyright 2025 – Apache-2.0
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
try:
    from verl.utils.hdfs_io import copy, makedirs # optional HDFS sync
except ImportError:
    # Fallback implementations if verl is not available
    def copy(src, dst):
        print(f"HDFS copy not available: {src} -> {dst}")
    def makedirs(path):
        print(f"HDFS makedirs not available: {path}")

DATA_SOURCE = "grounding_iou_grpo"        # arbitrary tag used by RewardManager
ABILITY     = "radiology_grounding"
PROMPT_SUFFIXE = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
TARGET_TOKENS = 1024  
MAX_PIXELS    = TARGET_TOKENS * 28 * 28    # ≈0.8 Mpx  (896×896)


def extract_bounding_boxes_from_text(text):
    """
    Extract bounding boxes from text that contains coordinates in format [x1, y1, x2, y2].
    Returns a list of bounding box lists.
    """
    if not text:
        return []
    
    # Look for bounding box patterns like [0.1, 0.2, 0.3, 0.4]
    box_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    matches = re.findall(box_pattern, text)
    
    boxes = []
    for match in matches:
        try:
            x1, y1, x2, y2 = [float(coord) for coord in match]
            boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
    
    return boxes


def is_grounding_task(conversations):
    """
    Determine if this is a grounding task based on the conversation content.
    Looks for location-related keywords and bounding box patterns.
    """
    if not conversations:
        return False
    
    # Combine all conversation text
    full_text = ""
    for conv in conversations:
        if conv.get('value'):
            full_text += conv['value'] + " "
    
    full_text = full_text.lower()
    
    # Check for grounding-related keywords
    grounding_keywords = [
        'location', 'locate', 'position', 'coordinates', 'bounding box',
        'where', 'point out', 'identify', 'highlight', 'mark', 'area',
        'region', 'situated', 'found at', 'located at'
    ]
    
    has_grounding_keywords = any(keyword in full_text for keyword in grounding_keywords)
    
    # Check for bounding box patterns in the text
    has_bounding_boxes = len(extract_bounding_boxes_from_text(full_text)) > 0
    
    return has_grounding_keywords or has_bounding_boxes


def make_map_fn(split):
    def proc(example, idx):
        # Skip if not a grounding task
        conversations = example.get("conversations", [])
        if not is_grounding_task(conversations):
            return None
        
        img_entry = {
            "image": f"file://{example['image']}",
            "resized_height": 512,           # 24×28px  →  576 tokens
            "resized_width":  512
        }
        
        # user prompt (keep the <image> token so Qwen knows where the pixel goes)
        prompt_msg = {
            "role": "user",
            "content": example["conversations"][0]["value"] + PROMPT_SUFFIXE
        }
        
        # Extract ground truth bounding boxes from the assistant's response
        assistant_response = example["conversations"][1]["value"] if len(example["conversations"]) > 1 else ""
        ground_truth_boxes = extract_bounding_boxes_from_text(assistant_response)
        
        # If no bounding boxes found, this might not be a proper grounding task
        if not ground_truth_boxes:
            return None
        
        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : [prompt_msg],              # same schema as other examples
            "images": [img_entry],     
            "ability"     : ABILITY,
            "reward_model": {
                "style": "iou",
                "ground_truth": ground_truth_boxes      # list of bounding boxes for IOU computation
            },
            "extra_info"  : {                          # anything you want to keep
                "id":  example["id"],
                "split": split,
                "index": idx,
                "original_assistant_response": assistant_response
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999, dataset_filter=None):
    with open(json_path) as f:
        rows = json.load(f)

    print(f"Initial number of samples: {len(rows)}")
    
    # Filter by dataset prefix if specified
    if dataset_filter:
        print(f"Filtering for datasets with prefix: {dataset_filter}")
        filtered_rows = [row for row in rows if row.get('id', '').startswith(dataset_filter)]
        print(f"Samples after dataset filtering: {len(filtered_rows)}")
        rows = filtered_rows

    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n  = int(len(ds) * train_ratio)

    print(f"Processing {len(ds)} samples for grounding tasks...")
    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=min(os.cpu_count(), 8))
    val_ds   = ds.select(range(n, len(ds))).map(make_map_fn("val"),   with_indices=True, num_proc=min(os.cpu_count(), 8))

    # Filter out None values (non-grounding tasks)
    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds = val_ds.filter(lambda x: x is not None)
    
    print(f"Grounding samples found - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    if len(train_ds) == 0 and len(val_ds) == 0:
        print("WARNING: No grounding samples found!")
        print("This could mean:")
        print("1. The dataset doesn't contain grounding tasks")
        print("2. The bounding box format is different than expected")
        print("3. The conversation structure is different")
        return

    os.makedirs(local_dir, exist_ok=True)
    
    if len(train_ds) > 0:
        train_path = os.path.join(local_dir, "train.parquet")
        print(f"Saving {len(train_ds)} train samples to {train_path}")
        train_ds.to_parquet(train_path)
    
    if len(val_ds) > 0:
        val_path = os.path.join(local_dir, "val.parquet")
        print(f"Saving {len(val_ds)} validation samples to {val_path}")
        val_ds.to_parquet(val_path)

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the input LLaVA JSON file")
    parser.add_argument("--local_dir", default="~/data/grounding_iou", help="Local directory to save processed datasets")
    parser.add_argument("--hdfs_dir",  default=None, help="HDFS directory for synchronization (optional)")
    parser.add_argument("--dataset_filter", default=None, help="Filter for specific dataset prefixes (e.g., 'vindr-cxr')")
    parser.add_argument("--train_ratio", type=float, default=0.999, help="Ratio of train vs validation split")
    
    args = parser.parse_args()
    args.local_dir = os.path.expanduser(args.local_dir)
    main(**vars(args))