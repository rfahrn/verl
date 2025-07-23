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

# Unique identifier for this specific GRPO data (VinDr-CXR grounding with IOU)
DATA_SOURCE = "vindr_cxr_grpo_iou"
ABILITY     = "radiology_grounding" # More specific ability for grounding tasks
PROMPT_SUFFIXE = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
TARGET_TOKENS    = 1024
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
        # Image path from the LLaVA JSON (all_train.json)
        img_path = example.get('image', '')
        if not img_path:
            return None # Skip if no image path

        img_entry = {
            "image": f"file://{img_path}", # Prepend file:// for Qwen or similar models
            "resized_height": 512,
            "resized_width":  512
        }

        # Conversations are already in the correct list-of-dicts format
        verl_prompt_list = example.get('conversations', [])
        processed_prompt_msgs = []
        user_prompt_found = False

        for turn in verl_prompt_list:
            if turn.get('from') == 'human':
                processed_prompt_msgs.append({
                    "role": "user",
                    "content": turn.get('value', '') + PROMPT_SUFFIXE
                })
                user_prompt_found = True
            elif turn.get('from') == 'gpt':
                processed_prompt_msgs.append({
                    "role": "assistant",
                    "content": turn.get('value', '')
                })
            else:
                # Handle unexpected turns or roles, or simply skip
                continue

        if not user_prompt_found or not processed_prompt_msgs:
            return None # Skip if no valid user prompt was processed

        # Check if this is a grounding task
        if not is_grounding_task(verl_prompt_list):
            return None

        # Try to extract reward_model information from the example or conversation
        reward_model_info = example.get('reward_model')
        
        # If no explicit reward_model info, try to extract from conversation
        if not reward_model_info:
            # Look for bounding boxes in the assistant's response (ground truth)
            ground_truth_boxes = []
            for turn in verl_prompt_list:
                if turn.get('from') == 'gpt':
                    boxes = extract_bounding_boxes_from_text(turn.get('value', ''))
                    ground_truth_boxes.extend(boxes)
            
            if ground_truth_boxes:
                reward_model_info = {
                    "style": "iou",
                    "ground_truth": ground_truth_boxes
                }
            else:
                # If we can't find bounding boxes, this might not be a proper grounding task
                return None

        # Final check: ensure we have IOU reward info with ground truth
        if not reward_model_info or reward_model_info.get("style") != "iou" or not reward_model_info.get("ground_truth"):
            return None

        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : processed_prompt_msgs,
            "images": [img_entry],
            "ability"     : ABILITY,
            "reward_model": reward_model_info,
            "extra_info"  : {
                "id":  example.get('id', f"generated_id_{idx}"),
                "split": split,
                "index": idx,
                "original_image_path": img_path,
                "dataset_id_prefix": example.get('id', '').split('_')[0] if example.get('id') else 'unknown'
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999):
    print(f"Loading data from {json_path}...")
    with open(json_path) as f:
        rows = json.load(f)

    # Convert to Hugging Face Dataset for easy filtering and mapping
    ds = datasets.Dataset.from_list(rows)

    initial_count = len(ds)
    print(f"Initial number of samples: {initial_count}")

    # --- Filter for VinDr-CXR datasets ONLY ---
    # The 'id' field from create_json_cell_llava starts with the id_prefix.
    # Filter for 'vindr-cxr' and 'vindr-cxr-mono' prefixes.
    print("Filtering for VinDr-CXR grounding tasks...")
    vindr_ds = ds.filter(lambda x: x.get('id', '').startswith(('vindr-cxr', 'vindr-cxr-mono')))

    if not vindr_ds:
        print("No VinDr-CXR samples found. Trying to process all samples and filter for grounding tasks...")
        # If no VinDr samples found, process all samples and filter for grounding tasks
        vindr_ds = ds
        print(f"Processing all {len(vindr_ds)} samples...")

    print(f"Number of samples after initial filtering: {len(vindr_ds)}")

    # Shuffle and split
    vindr_ds = vindr_ds.shuffle(seed=42)
    n = int(len(vindr_ds) * train_ratio)

    print(f"Mapping and splitting {len(vindr_ds)} samples (Train: {n}, Val: {len(vindr_ds) - n})...")

    train_ds_mapped = vindr_ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=min(os.cpu_count(), 8), remove_columns=vindr_ds.column_names)
    val_ds_mapped   = vindr_ds.select(range(n, len(vindr_ds))).map(make_map_fn("val"),   with_indices=True, num_proc=min(os.cpu_count(), 8), remove_columns=vindr_ds.column_names)

    # Filter out any samples that returned None from make_map_fn (e.g., non-grounding tasks)
    train_ds = train_ds_mapped.filter(lambda x: x is not None)
    val_ds = val_ds_mapped.filter(lambda x: x is not None)

    processed_train_count = len(train_ds)
    processed_val_count = len(val_ds)
    print(f"Processed grounding samples after final filtering: Train: {processed_train_count}, Val: {processed_val_count}")

    if processed_train_count == 0 and processed_val_count == 0:
        print("WARNING: No grounding samples found! This could mean:")
        print("1. The dataset doesn't contain grounding tasks")
        print("2. The bounding box format is different than expected")
        print("3. The conversation structure is different")
        
        # Let's examine a few samples to help debug
        print("\nExamining first few samples for debugging:")
        for i, sample in enumerate(rows[:3]):
            print(f"\nSample {i}:")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"Image: {sample.get('image', 'N/A')}")
            if 'conversations' in sample:
                for j, conv in enumerate(sample['conversations']):
                    print(f"  Conversation {j}: {conv.get('from', 'N/A')} -> {conv.get('value', 'N/A')[:100]}...")
        return

    os.makedirs(local_dir, exist_ok=True)
    train_output_path = os.path.join(local_dir, "train_vindr_grounding_iou.parquet")
    val_output_path = os.path.join(local_dir, "val_vindr_grounding_iou.parquet")

    if processed_train_count > 0:
        print(f"Saving {processed_train_count} train samples to {train_output_path}")
        train_ds.to_parquet(train_output_path)
    else:
        print("No valid train samples to save for grounding.")

    if processed_val_count > 0:
        print(f"Saving {processed_val_count} validation samples to {val_output_path}")
        val_ds.to_parquet(val_output_path)
    else:
        print("No valid validation samples to save for grounding.")

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)
        print(f"HDFS sync initiated for {local_dir} to {hdfs_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", help="Path to the input LLaVA JSON file (e.g., llava_datasets/all_train.json)")
    parser.add_argument("--local_dir", default="~/data/vindr_grpo_iou_dataset", help="Local directory to save the processed datasets for IOU reward training")
    parser.add_argument("--hdfs_dir",  default=None, help="HDFS directory for synchronization (optional)")
    args = parser.parse_args()

    args.local_dir = os.path.expanduser(args.local_dir)

    main(**vars(args))