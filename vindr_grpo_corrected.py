# Copyright 2025 – Apache-2.0
import json, argparse, os, datasets, re
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs

DATA_SOURCE = "vindr_grpo"
ABILITY = "medical_grounding"

# All 28 VinDR-CXR findings as per the dataset documentation
VINDR_FINDINGS = [
    # Local labels (1-22) - require bounding boxes
    "Aortic enlargement", "Atelectasis", "Cardiomegaly", "Calcification", 
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA",
    "Interstitial lung disease", "Infiltration", "Lung cavity", "Lung cyst",
    "Lung opacity", "Mediastinal shift", "Nodule/Mass", "Pulmonary fibrosis",
    "Pneumothorax", "Pleural thickening", "Pleural effusion", "Rib fracture",
    "Other lesion",
    # Global labels (23-28) - diagnostic impressions, no bounding boxes
    "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", 
    "Chronic obstructive pulmonary disease", "No finding"
]

LOCAL_LABELS = VINDR_FINDINGS[:22]  # Require bounding boxes
GLOBAL_LABELS = VINDR_FINDINGS[22:]  # Diagnostic impressions only

# Enhanced reasoning prompt for both localization and classification
REASONING_PROMPT = (
    " Please analyze this chest X-ray systematically for both localized findings and overall diagnostic impressions. "
    "Provide your analysis in the following format:\n"
    "1. First output your detailed thinking process in <think></think> tags, including:\n"
    "   - Systematic examination of different anatomical regions\n"
    "   - Identification of any localized abnormalities with precise locations\n"
    "   - Overall diagnostic impression and global findings\n"
    "   - Confidence assessment for each finding\n"
    "2. Then output your final answer in <answer></answer> tags with:\n"
    "   - LOCAL FINDINGS: Any localized abnormalities with bounding box coordinates [x1, y1, x2, y2]\n"
    "   - GLOBAL FINDINGS: Overall diagnostic impressions without coordinates\n"
    "   - If no abnormalities found, clearly state 'No finding'"
)

def extract_coordinates_and_labels(gpt_response, original_labels):
    """
    Extract coordinate information and labels from the ground truth response.
    Enhanced to handle all 28 VinDR findings properly.
    """
    # Pattern to match coordinates like [0.48, 0.34, 0.59, 0.43]
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, gpt_response)
    
    # Convert to float tuples
    coord_boxes = [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in coordinates]
    
    # Separate local and global labels based on original_labels
    local_findings = []
    global_findings = []
    
    for label in original_labels:
        if label in LOCAL_LABELS:
            local_findings.append(label)
        elif label in GLOBAL_LABELS:
            global_findings.append(label)
    
    # Extract mentioned findings from response text (for model evaluation)
    response_lower = gpt_response.lower()
    mentioned_local = []
    mentioned_global = []
    
    for finding in LOCAL_LABELS:
        if finding.lower() in response_lower:
            mentioned_local.append(finding)
    
    for finding in GLOBAL_LABELS:
        if finding.lower() in response_lower:
            mentioned_global.append(finding)
    
    return {
        "coordinates": coord_boxes,
        "local_findings_gt": local_findings,
        "global_findings_gt": global_findings,
        "local_findings_mentioned": mentioned_local,
        "global_findings_mentioned": mentioned_global,
        "total_findings": len(local_findings) + len(global_findings),
        "has_no_finding": "No finding" in original_labels
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
            "resized_height": 512,
            "resized_width": 512,
            "image_processor_type": "qwen2_vl"
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
        original_labels = example.get("labels", [])
        parsed_gt = extract_coordinates_and_labels(gpt_response, original_labels)

        # IMPORTANT: Ground truth structure for VERL RewardManager
        # This will be passed as the 'ground_truth' parameter to compute_score()
        ground_truth_data = {
            "raw_response": gpt_response,
            "coordinates": parsed_gt["coordinates"],
            "local_findings": parsed_gt["local_findings_gt"],
            "global_findings": parsed_gt["global_findings_gt"],
            "all_labels": original_labels,
            "has_no_finding": parsed_gt["has_no_finding"],
            "image_path": img_path
        }

        return {
            "data_source": DATA_SOURCE,
            "prompt": [prompt_msg],
            "images": [img_entry],
            "ability": ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_data  # This becomes the 'ground_truth' parameter
            },
            "extra_info": {
                "id": example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "original_labels": original_labels,
                "image_path": img_path,
                "local_count": len(parsed_gt["local_findings_gt"]),
                "global_count": len(parsed_gt["global_findings_gt"]),
                "coordinate_count": len(parsed_gt["coordinates"])
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

    # Analyze dataset statistics
    local_count = global_count = no_finding_count = 0
    for row in valid_rows:
        labels = row.get("labels", [])
        has_local = any(label in LOCAL_LABELS for label in labels)
        has_global = any(label in GLOBAL_LABELS for label in labels)
        
        if has_local:
            local_count += 1
        if has_global:
            global_count += 1
        if "No finding" in labels:
            no_finding_count += 1

    print(f"\n📊 Dataset Analysis:")
    print(f"Examples with local findings: {local_count}")
    print(f"Examples with global findings: {global_count}")
    print(f"Examples with no findings: {no_finding_count}")

    # Create dataset and shuffle
    ds = datasets.Dataset.from_list(valid_rows).shuffle(seed=42)

    # Split into train/val
    n = int(len(ds) * train_ratio)

    print(f"Processing train split (n={n})...")
    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    train_ds = train_ds.filter(lambda x: x is not None)

    print(f"Processing val split (n={len(ds)-n})...")
    val_ds = ds.select(range(n, len(ds))).map(make_map_fn("val"), with_indices=True, num_proc=8)
    val_ds = val_ds.filter(lambda x: x is not None)

    print(f"Final counts - Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Print sample structure
    if len(train_ds) > 0:
        sample = train_ds[0]
        print(f"\n📝 Sample structure:")
        print(f"Ground truth keys: {list(sample['reward_model']['ground_truth'].keys())}")
        print(f"Sample local findings: {sample['reward_model']['ground_truth']['local_findings']}")
        print(f"Sample global findings: {sample['reward_model']['ground_truth']['global_findings']}")

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
    print("📋 Next steps:")
    print("1. Create custom reward function with proper VERL signature")
    print("2. Configure GRPO training with group sampling (n > 1)")
    print("3. Set up evaluation for both localization (mAP) and classification (F1)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced VinDR-CXR preprocessing for GRPO with 28 findings")
    parser.add_argument("json_path", help="Path to VinDR-CXR JSON file")
    parser.add_argument("--local_dir", default="~/data/vindr_grpo", help="Local directory to save parquet files")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory for sync (optional)")
    parser.add_argument("--train_ratio", type=float, default=0.999, help="Ratio of data for training")

    args = parser.parse_args()
    main(**vars(args))