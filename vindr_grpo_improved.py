# Copyright 2025 – Apache-2.0
import json, argparse, os, datasets, re
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "vindr_grpo"
ABILITY = "medical_grounding"

# Enhanced prompt for better reasoning
REASONING_PROMPT = (
    " Please analyze this chest X-ray systematically. "
    "First, carefully examine the image for any abnormalities or lesions. "
    "Then provide your analysis in the following format:\n"
    "1. First output your thinking process in <think></think> tags, including:\n"
    "   - Initial observations\n"
    "   - Systematic analysis of different regions\n"
    "   - Identification of potential abnormalities\n"
    "   - Localization reasoning\n"
    "2. Then output your final answer in <answer></answer> tags with:\n"
    "   - Clear identification of abnormalities\n"
    "   - Precise coordinate locations in [x1, y1, x2, y2] format\n"
    "   - Confidence in your findings"
)

def extract_coordinates_and_labels(gpt_response):
    """
    Extract coordinate information and labels from the ground truth response.
    This will be used by the reward function for mAP calculation.
    """
    # Pattern to match coordinates like [0.48, 0.34, 0.59, 0.43]
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, gpt_response)
    
    # Convert to float tuples
    coord_boxes = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in coordinates]
    
    # Extract lesion types mentioned before coordinates
    # This is a simple approach - you might want to make this more sophisticated
    lesion_mentions = []
    if "aortic enlargement" in gpt_response.lower():
        lesion_mentions.append("Aortic enlargement")
    if "cardiomegaly" in gpt_response.lower():
        lesion_mentions.append("Cardiomegaly")
    if "pleural effusion" in gpt_response.lower():
        lesion_mentions.append("Pleural effusion")
    if "pleural thickening" in gpt_response.lower():
        lesion_mentions.append("Pleural thickening")
    if "other lesion" in gpt_response.lower():
        lesion_mentions.append("Other lesion")
    if "no finding" in gpt_response.lower() or "no abnormalities" in gpt_response.lower():
        lesion_mentions.append("No finding")
    
    return {
        "coordinates": coord_boxes,
        "lesion_types": lesion_mentions,
        "raw_response": gpt_response
    }

def make_map_fn(split):
    def proc(example, idx):
        # Check if image exists
        img_path = example.get("image", "")
        if not img_path or not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            return None

        # Image configuration optimized for medical images
        img_entry = {
            "image": f"file://{img_path}",
            "resized_height": 512,  # Good resolution for medical images
            "resized_width": 512,
            "image_processor_type": "qwen2_vl"  # Specify the processor
        }

        # Enhanced prompt for reasoning
        human_message = example["conversations"][0]["value"]
        enhanced_prompt = human_message + REASONING_PROMPT

        prompt_msg = {
            "role": "user",
            "content": enhanced_prompt
        }

        # Process ground truth for reward calculation
        gpt_response = example["conversations"][1]["value"]
        parsed_gt = extract_coordinates_and_labels(gpt_response)

        # Enhanced ground truth structure for mAP calculation
        ground_truth_data = {
            "raw_response": gpt_response,
            "parsed_coordinates": parsed_gt["coordinates"],
            "lesion_types": parsed_gt["lesion_types"],
            "original_labels": example.get("labels", []),
            "has_findings": "No finding" not in example.get("labels", [])
        }

        return {
            "data_source": DATA_SOURCE,
            "prompt": [prompt_msg],
            "images": [img_entry],
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_data  # Enhanced structured ground truth
            },
            "extra_info": {
                "id": example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "original_labels": example.get("labels", []),
                "image_path": img_path,
                "raw_gpt_response": gpt_response,  # Keep original for debugging
                "coordinate_count": len(parsed_gt["coordinates"]),
                "lesion_count": len([l for l in example.get("labels", []) if l != "No finding"])
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999):
    print(f"Loading VinDR-CXR dataset from {json_path}")

    with open(json_path) as f:
        rows = json.load(f)

    print(f"Loaded {len(rows)} examples")
    
    # Filter out examples without valid images
    valid_rows = []
    for row in rows:
        if row.get("image") and os.path.exists(row["image"]):
            valid_rows.append(row)
        else:
            print(f"Skipping example with missing image: {row.get('image', 'N/A')}")
    
    print(f"Valid examples after filtering: {len(valid_rows)}")

    # Create dataset and shuffle
    ds = datasets.Dataset.from_list(valid_rows).shuffle(seed=42)

    # Split into train/val with more reasonable validation set for evaluation
    n = int(len(ds) * train_ratio)

    print(f"Processing train split (n={n})...")
    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    train_ds = train_ds.filter(lambda x: x is not None)

    print(f"Processing val split (n={len(ds)-n})...")
    val_ds = ds.select(range(n, len(ds))).map(make_map_fn("val"), with_indices=True, num_proc=8)
    val_ds = val_ds.filter(lambda x: x is not None)

    print(f"Final counts - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Print some statistics
    print("\n📊 Dataset Statistics:")
    sample_example = train_ds[0]
    print(f"Sample prompt length: {len(sample_example['prompt'][0]['content'])}")
    print(f"Sample ground truth keys: {list(sample_example['reward_model']['ground_truth'].keys())}")
    if sample_example['reward_model']['ground_truth']['parsed_coordinates']:
        print(f"Sample coordinates: {sample_example['reward_model']['ground_truth']['parsed_coordinates'][0]}")

    # Save to parquet
    os.makedirs(local_dir, exist_ok=True)

    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet(os.path.join(local_dir, "val.parquet"))

    print(f"Saved parquet files to {local_dir}")

    # Optional HDFS sync
    if hdfs_dir:
        print(f"Syncing to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("HDFS sync completed")

    print("\n✅ Enhanced VinDR preprocessing completed!")
    print("📝 Next steps:")
    print("1. Create a custom reward function for mAP calculation")
    print("2. Configure GRPO training with group sampling (n > 1)")
    print("3. Set up proper evaluation metrics for medical grounding")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced VinDR-CXR preprocessing for GRPO reasoning training")
    parser.add_argument("json_path", help="Path to VinDR-CXR JSON file")
    parser.add_argument("--local_dir", default="~/data/vindr_grpo", help="Local directory to save parquet files")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for sync (optional)")
    parser.add_argument("--train_ratio", type=float, default=0.999, help="Ratio of data for training")

    args = parser.parse_args()
    main(**vars(args))