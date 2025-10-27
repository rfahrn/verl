#!/usr/bin/env python3
"""
Convert LLaVA JSON format to VERL format with IOU reward for grounding tasks.
This version pre-filters based on text length to avoid slow multimodal filtering.
"""

import json
import argparse
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import re
import random

# Optional VERL imports
try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    # Fallback functions if VERL is not available
    import shutil
    import os
    
    def copy(src, dst):
        shutil.copy2(src, dst)
    
    def makedirs(path, exist_ok=True):
        os.makedirs(path, exist_ok=exist_ok)

# Constants matching VERL requirements
DATA_SOURCE = "vindr_cxr_grpo_iou"
ABILITY = "radiology_grounding"
PROMPT_SUFFIXE = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

# Rough estimate: 1 token ≈ 4 characters for text filtering
CHARS_PER_TOKEN = 4
MAX_TEXT_LENGTH = 2048 * CHARS_PER_TOKEN  # Pre-filter at ~8192 chars

def extract_bounding_boxes_from_text(text):
    """Extract bounding boxes from text in format [x1, y1, x2, y2]"""
    pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
    matches = re.findall(pattern, text)
    boxes = []
    for match in matches:
        try:
            box = [float(coord) for coord in match]
            boxes.append(box)
        except ValueError:
            continue
    return boxes

def is_grounding_task(conversations):
    """Check if this is a grounding task based on conversation content"""
    for turn in conversations:
        if turn.get('from') == 'gpt':
            text = turn.get('value', '').lower()
            # Check for bounding box patterns or grounding-related keywords
            if (re.search(r'\[\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\]', text) or
                'location' in text or 'abnormality' in text or 'finding' in text):
                return True
    return False

def estimate_text_length(conversations):
    """Estimate total text length for pre-filtering"""
    total_length = 0
    for turn in conversations:
        content = turn.get('value', '')
        total_length += len(content)
    return total_length + len(PROMPT_SUFFIXE)

def convert_sample_to_verl(example, idx, split):
    """Convert a single LLaVA sample to VERL format with pre-filtering"""
    
    # Extract basic info
    img_path = example.get('image', '')
    conversations = example.get('conversations', [])
    sample_id = example.get('id', f'{split}_{idx}')
    
    if not img_path or not conversations:
        return None  # Skip if no image path or conversations
    
    # Check if this is a grounding task
    if not is_grounding_task(conversations):
        return None
    
    # PRE-FILTER: Estimate text length before expensive processing
    estimated_length = estimate_text_length(conversations)
    if estimated_length > MAX_TEXT_LENGTH:
        return None  # Skip overly long samples early
    
    # Convert conversations to VERL prompt format
    processed_prompt_msgs = []
    ground_truth_boxes = []
    user_prompt_found = False
    
    for turn in conversations:
        if turn.get('from') == 'human':
            processed_prompt_msgs.append({
                "role": "user",
                "content": turn.get('value', '') + PROMPT_SUFFIXE
            })
            user_prompt_found = True
        elif turn.get('from') == 'gpt':
            assistant_response = turn.get('value', '')
            processed_prompt_msgs.append({
                "role": "assistant",
                "content": assistant_response
            })
            # Extract ground truth bounding boxes from assistant response
            boxes = extract_bounding_boxes_from_text(assistant_response)
            ground_truth_boxes.extend(boxes)
        else:
            # Handle unexpected turns or roles, or simply skip
            continue
    
    if not user_prompt_found or not processed_prompt_msgs:
        return None  # Skip if no valid user prompt was processed
    
    # For grounding tasks, we need bounding boxes in the ground truth
    # If no bounding boxes found, this might be a "No finding" case
    if not ground_truth_boxes:
        # Check if this is explicitly a "no finding" case
        assistant_text = ""
        for turn in conversations:
            if turn.get('from') == 'gpt':
                assistant_text += turn.get('value', '').lower()
        
        # Look for explicit "no finding" or similar phrases
        no_finding_phrases = ['no abnormalit', 'no finding', 'no patholog', 'normal', 'unremarkable']
        if any(phrase in assistant_text for phrase in no_finding_phrases):
            ground_truth_boxes = []  # Empty list for "no finding" cases
        else:
            return None  # Skip if it's not a clear "no finding" case
    
    # Create VERL format sample with correct image format
    img_entry = {
        "image": f"file://{img_path}",
        "resized_height": 512,
        "resized_width": 512
    }
    
    verl_sample = {
        "data_source": DATA_SOURCE,
        "prompt": processed_prompt_msgs,
        "images": [img_entry],
        "ability": ABILITY,
        "reward_model": {
            "style": "iou",
            "ground_truth": ground_truth_boxes
        },
        "extra_info": {
            "id": sample_id,
            "split": split,
            "index": idx,
            "original_image_path": img_path,
            "dataset_id_prefix": sample_id.split('_')[0] if sample_id else 'unknown',
            "original_labels": example.get('labels', []),
            "estimated_text_length": estimated_length
        }
    }
    
    return verl_sample


def process_samples_batch(samples, split, start_idx=0):
    """Process a batch of samples and return only the valid VERL samples."""
    verl_samples = []
    skipped_long = 0
    
    for i, sample in enumerate(tqdm(samples, desc=f"Processing {split} batch")):
        verl_sample = convert_sample_to_verl(sample, start_idx + i, split)
        if verl_sample is not None:
            verl_samples.append(verl_sample)
        else:
            # Check if it was skipped due to length
            if sample.get('conversations'):
                est_len = estimate_text_length(sample.get('conversations', []))
                if est_len > MAX_TEXT_LENGTH:
                    skipped_long += 1
    
    if skipped_long > 0:
        print(f"Pre-filtered {skipped_long} samples due to estimated length > {MAX_TEXT_LENGTH} chars")
    
    return verl_samples


def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999, dataset_filter=None, batch_size=10000):
    """Main function to convert LLaVA JSON to VERL format with pre-filtering"""
    
    print(f"Loading JSON from {json_path}")
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    print(f"Loaded {len(data)} samples from JSON")
    
    # Filter by dataset prefix if specified
    if dataset_filter:
        filtered_data = []
        for sample in data:
            sample_id = sample.get('id', '')
            if sample_id.startswith(dataset_filter):
                filtered_data.append(sample)
        data = filtered_data
        print(f"After filtering by prefix '{dataset_filter}': {len(data)} samples")
    
    # Shuffle the data
    random.shuffle(data)
    
    # Split into train and validation
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Process training data in batches
    print("Processing training data...")
    all_train_samples = []
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        batch_samples = process_samples_batch(batch, "train", i)
        all_train_samples.extend(batch_samples)
        print(f"Processed batch {i//batch_size + 1}, got {len(batch_samples)} valid samples")
    
    # Process validation data in batches
    print("Processing validation data...")
    all_val_samples = []
    for i in range(0, len(val_data), batch_size):
        batch = val_data[i:i + batch_size]
        batch_samples = process_samples_batch(batch, "val", i)
        all_val_samples.extend(batch_samples)
        print(f"Processed batch {i//batch_size + 1}, got {len(batch_samples)} valid samples")
    
    print(f"Final train samples: {len(all_train_samples)}")
    print(f"Final validation samples: {len(all_val_samples)}")
    
    # Create datasets and save as parquet
    if all_train_samples:
        train_dataset = Dataset.from_list(all_train_samples)
        train_path = f"{local_dir}/train_verl_iou_fast.parquet"
        makedirs(local_dir, exist_ok=True)
        train_dataset.to_parquet(train_path)
        print(f"Saved training data to {train_path}")
        
        if hdfs_dir:
            hdfs_train_path = f"{hdfs_dir}/train_verl_iou_fast.parquet"
            copy(train_path, hdfs_train_path)
            print(f"Copied to HDFS: {hdfs_train_path}")
    
    if all_val_samples:
        val_dataset = Dataset.from_list(all_val_samples)
        val_path = f"{local_dir}/val_verl_iou_fast.parquet"
        val_dataset.to_parquet(val_path)
        print(f"Saved validation data to {val_path}")
        
        if hdfs_dir:
            hdfs_val_path = f"{hdfs_dir}/val_verl_iou_fast.parquet"
            copy(val_path, hdfs_val_path)
            print(f"Copied to HDFS: {hdfs_val_path}")
    
    print("Conversion completed!")
    print(f"Pre-filtering at ~{MAX_TEXT_LENGTH} characters should eliminate most overly long samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA JSON to VERL format with fast pre-filtering")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the LLaVA JSON file")
    parser.add_argument("--local_dir", type=str, default="./data", help="Local directory to save parquet files")
    parser.add_argument("--hdfs_dir", type=str, help="HDFS directory to copy files (optional)")
    parser.add_argument("--train_ratio", type=float, default=0.999, help="Ratio of data to use for training")
    parser.add_argument("--dataset_filter", type=str, help="Filter samples by ID prefix (e.g., 'vindr-cxr')")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for processing")
    
    args = parser.parse_args()
    main(args.json_path, args.local_dir, args.hdfs_dir, args.train_ratio, args.dataset_filter, args.batch_size)