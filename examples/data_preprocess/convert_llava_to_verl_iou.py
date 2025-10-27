# Copyright 2025 – Apache-2.0
import json, argparse, os, re, datasets
import pandas as pd
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
        'region', 'situated', 'found at', 'located at'
    ]
    
    has_grounding_keywords = any(keyword in full_text for keyword in grounding_keywords)
    
    # Check for bounding box patterns in the text
    has_bounding_boxes = len(extract_bounding_boxes_from_text(full_text)) > 0
    
    return has_grounding_keywords or has_bounding_boxes


def convert_llava_sample_to_verl(sample, idx, split):
    """
    Convert a single LLaVA sample to VERL format for IOU reward.
    """
    # Extract data from LLaVA format
    img_path = sample.get('image', '')
    conversations = sample.get('conversations', [])
    sample_id = sample.get('id', f'converted_{idx}')
    
    if not img_path or not conversations:
        return None
    
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
    
    for turn in conversations:
        if turn.get('from') == 'human':
            processed_prompt_msgs.append({
                "role": "user",
                "content": turn.get('value', '') + PROMPT_SUFFIXE
            })
        elif turn.get('from') == 'gpt':
            assistant_response = turn.get('value', '')
            processed_prompt_msgs.append({
                "role": "assistant",
                "content": assistant_response
            })
            # Extract ground truth bounding boxes from assistant response
            boxes = extract_bounding_boxes_from_text(assistant_response)
            ground_truth_boxes.extend(boxes)
    
    # Skip if no valid conversations or no bounding boxes found
    if not processed_prompt_msgs or not ground_truth_boxes:
        return None
    
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
            "original_labels": sample.get('labels')
        }
    }
    
    return verl_sample


def convert_parquet_to_verl(input_parquet_path, output_dir, split_name="train"):
    """
    Convert a LLaVA format parquet file to VERL format.
    """
    print(f"Loading parquet file: {input_parquet_path}")
    df = pd.read_parquet(input_parquet_path)
    
    print(f"Original columns: {df.columns.tolist()}")
    print(f"Original shape: {df.shape}")
    
    # Convert to list of dictionaries
    samples = df.to_dict('records')
    
    print(f"Converting {len(samples)} samples to VERL format...")
    
    converted_samples = []
    grounding_count = 0
    
    for idx, sample in enumerate(tqdm(samples)):
        verl_sample = convert_llava_sample_to_verl(sample, idx, split_name)
        if verl_sample is not None:
            converted_samples.append(verl_sample)
            grounding_count += 1
    
    print(f"Found {grounding_count} grounding tasks out of {len(samples)} total samples")
    
    if grounding_count == 0:
        print("WARNING: No grounding samples found!")
        # Show a few samples for debugging
        print("\nSample conversations for debugging:")
        for i in range(min(3, len(samples))):
            sample = samples[i]
            print(f"\nSample {i}:")
            print(f"ID: {sample.get('id', 'N/A')}")
            print(f"Conversations: {sample.get('conversations', [])}")
        return
    
    # Convert to Hugging Face dataset and save
    verl_dataset = datasets.Dataset.from_list(converted_samples)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}_verl_iou.parquet")
    
    print(f"Saving {len(converted_samples)} VERL samples to {output_path}")
    verl_dataset.to_parquet(output_path)
    
    # Show the new structure
    print(f"\nNew VERL columns: {verl_dataset.column_names}")
    print(f"New shape: {verl_dataset.shape}")
    
    # Show a sample
    if len(converted_samples) > 0:
        print(f"\nSample VERL format:")
        sample = converted_samples[0]
        print(f"- data_source: {sample['data_source']}")
        print(f"- ability: {sample['ability']}")
        print(f"- reward_model: {sample['reward_model']}")
        print(f"- prompt length: {len(sample['prompt'])}")
        print(f"- images: {len(sample['images'])}")


def main():
    parser = argparse.ArgumentParser(description="Convert LLaVA format parquet to VERL format with IOU reward")
    parser.add_argument("input_parquet", help="Path to input LLaVA format parquet file")
    parser.add_argument("--output_dir", default="./verl_output", help="Output directory for VERL format files")
    parser.add_argument("--split_name", default="train", help="Split name for output file")
    
    args = parser.parse_args()
    
    convert_parquet_to_verl(args.input_parquet, args.output_dir, args.split_name)
    print("Conversion completed!")


if __name__ == "__main__":
    main()