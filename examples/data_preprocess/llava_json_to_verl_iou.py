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

# VERL format constants
DATA_SOURCE = "vindr_cxr_grpo_iou"
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
        'region', 'situated', 'found at', 'located at', 'their locations',
        'specific locations', 'where are they located', 'where exactly'
    ]
    
    has_grounding_keywords = any(keyword in full_text for keyword in grounding_keywords)
    
    # Check for bounding box patterns in the text
    has_bounding_boxes = len(extract_bounding_boxes_from_text(full_text)) > 0
    
    return has_grounding_keywords or has_bounding_boxes


def make_map_fn(split):
    def proc(example, idx):
        # Extract data from LLaVA JSON format
        img_path = example.get('image', '')
        conversations = example.get('conversations', [])
        sample_id = example.get('id', f'generated_id_{idx}')
        
        if not img_path or not conversations:
            return None  # Skip if no image path or conversations
        
        # Check if this is a grounding task
        if not is_grounding_task(conversations):
            return None
        
        # Create VERL image entry
        img_entry = {
            "image": f"file://{img_path}",
            "resized_height": 512,
            "resized_width": 512
        }
        
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
        # We'll still include it but with empty ground truth for IOU reward
        if not ground_truth_boxes:
            # Check if this is a "No finding" case
            assistant_text = ""
            for turn in conversations:
                if turn.get('from') == 'gpt':
                    assistant_text += turn.get('value', '').lower()
            
            # If it's a grounding question but no findings, we still include it
            # The IOU reward will give 0 score when model predicts boxes but GT is empty
            # and could give positive score if model correctly predicts no boxes
            if any(phrase in assistant_text for phrase in ['no finding', 'no abnormalities', 'no lesions', 'clear', 'clean bill']):
                ground_truth_boxes = []  # Empty list for "no finding" cases
            else:
                return None  # Skip if it's not a clear "no finding" case
        
        # Create VERL format sample
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
                "original_labels": example.get('labels', [])
            }
        }
        
        return verl_sample
    
    return proc


def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999, dataset_filter=None):
    print(f"Loading LLaVA JSON from {json_path}...")
    with open(json_path) as f:
        rows = json.load(f)
    
    print(f"Initial number of samples: {len(rows)}")
    
    # Filter by dataset prefix if specified
    if dataset_filter:
        print(f"Filtering for datasets with prefix: {dataset_filter}")
        filtered_rows = []
        for row in rows:
            sample_id = row.get('id', '')
            if any(sample_id.startswith(prefix) for prefix in dataset_filter if prefix):
                filtered_rows.append(row)
        print(f"Samples after dataset filtering: {len(filtered_rows)}")
        rows = filtered_rows
    
    # Convert to Hugging Face Dataset for easy processing
    ds = datasets.Dataset.from_list(rows)
    
    # Shuffle and split
    ds = ds.shuffle(seed=42)
    n = int(len(ds) * train_ratio)
    
    print(f"Processing {len(ds)} samples for grounding tasks...")
    print(f"Train/Val split: {n}/{len(ds) - n}")
    
    # Process train and validation sets with reduced parallelism to avoid multiprocessing issues
    print("Processing training set...")
    train_ds_mapped = ds.select(range(n)).map(
        make_map_fn("train"), 
        with_indices=True, 
        num_proc=2,  # Reduced from 8 to avoid multiprocessing issues with None filtering
        remove_columns=ds.column_names
    )
    
    print("Processing validation set...")
    val_ds_mapped = ds.select(range(n, len(ds))).map(
        make_map_fn("val"), 
        with_indices=True, 
        num_proc=2,  # Reduced from 8 to avoid multiprocessing issues with None filtering
        remove_columns=ds.column_names
    )
    
    print("Filtering out non-grounding tasks...")
    # Filter out None values (non-grounding tasks)
    train_ds = train_ds_mapped.filter(lambda x: x is not None)
    val_ds = val_ds_mapped.filter(lambda x: x is not None)
    
    processed_train_count = len(train_ds)
    processed_val_count = len(val_ds)
    print(f"Processed grounding samples: Train: {processed_train_count}, Val: {processed_val_count}")
    
    if processed_train_count == 0 and processed_val_count == 0:
        print("WARNING: No grounding samples found!")
        print("This could mean:")
        print("1. The dataset doesn't contain grounding tasks")
        print("2. The grounding keywords don't match")
        print("3. The conversation structure is different")
        
        # Show sample data for debugging
        print("\nExamining first few samples for debugging:")
        for i, sample in enumerate(rows[:3]):
            print(f"\nSample {i}:")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"Image: {sample.get('image', 'N/A')}")
            if 'conversations' in sample:
                for j, conv in enumerate(sample['conversations']):
                    print(f"  Conversation {j}: {conv.get('from', 'N/A')} -> {conv.get('value', 'N/A')[:100]}...")
        return
    
    # Save the processed datasets
    os.makedirs(local_dir, exist_ok=True)
    
    if processed_train_count > 0:
        train_output_path = os.path.join(local_dir, "train.parquet")
        print(f"Saving {processed_train_count} train samples to {train_output_path}")
        train_ds.to_parquet(train_output_path)
        
        # Show sample structure
        print(f"\nTrain dataset columns: {train_ds.column_names}")
        if processed_train_count > 0:
            sample = train_ds[0]
            print(f"Sample structure:")
            print(f"  - data_source: {sample['data_source']}")
            print(f"  - ability: {sample['ability']}")
            print(f"  - reward_model: {sample['reward_model']}")
            print(f"  - prompt length: {len(sample['prompt'])}")
            print(f"  - images: {len(sample['images'])}")
            print(f"  - ground_truth boxes: {len(sample['reward_model']['ground_truth'])}")
    else:
        print("No valid train samples to save for grounding.")
    
    if processed_val_count > 0:
        val_output_path = os.path.join(local_dir, "val.parquet")
        print(f"Saving {processed_val_count} validation samples to {val_output_path}")
        val_ds.to_parquet(val_output_path)
    else:
        print("No valid validation samples to save for grounding.")
    
    if hdfs_dir:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"HDFS sync initiated for {local_dir} to {hdfs_dir}.")
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA JSON directly to VERL format with IOU reward")
    parser.add_argument("json_path", help="Path to the input LLaVA JSON file (e.g., all_train_llava.json)")
    parser.add_argument("--local_dir", default="~/data/verl_grounding_iou", help="Local directory to save the processed VERL datasets")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for synchronization (optional)")
    parser.add_argument("--train_ratio", type=float, default=0.999, help="Ratio of train vs validation split")
    parser.add_argument("--dataset_filter", nargs='+', default=None, 
                        help="Filter for specific dataset prefixes (e.g., --dataset_filter vindr-cxr vindr-cxr-mono)")
    
    args = parser.parse_args()
    args.local_dir = os.path.expanduser(args.local_dir)
    
    main(**vars(args))