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

# Medical finding labels for VinDR-CXR
MEDICAL_LABELS = [
    "cardiomegaly", "aortic enlargement", "pleural effusion", "pulmonary edema",
    "pneumonia", "atelectasis", "pneumothorax", "consolidation", "nodule", "mass",
    "infiltration", "fibrosis", "emphysema", "calcification", "opacity",
    "lesion", "abnormality", "finding"
]

def extract_coordinates(solution_str):
    """Extract coordinates from VinDR ground truth for reward function."""
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, solution_str)

    if not coordinates:
        if "no abnormalities" in solution_str.lower() or "no finding" in solution_str.lower():
            return []  # Empty list for "No finding" cases
        return []  # Empty list for parsing errors too

    # Convert to coordinate list for mAP calculation
    return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]

def extract_medical_labels(text, labels_list=None):
    """
    Extract medical finding labels from text and provided labels list.
    Returns list of normalized medical terms found.
    """
    found_labels = set()
    text_lower = text.lower()
    
    # Extract from provided labels list (from dataset metadata)
    if labels_list:
        for label in labels_list:
            if isinstance(label, str):
                # Normalize label (remove spaces, convert to lowercase)
                normalized_label = label.lower().strip()
                if normalized_label and normalized_label != "no finding":
                    found_labels.add(normalized_label)
    
    # Extract from text using medical label keywords
    for label in MEDICAL_LABELS:
        if label.lower() in text_lower:
            found_labels.add(label.lower())
    
    # Handle compound terms and variations
    label_variations = {
        "cardiac enlargement": "cardiomegaly",
        "heart enlargement": "cardiomegaly", 
        "enlarged heart": "cardiomegaly",
        "lung nodule": "nodule",
        "pulmonary nodule": "nodule",
        "lung mass": "mass",
        "pulmonary mass": "mass",
        "fluid in lungs": "pleural effusion",
        "collapsed lung": "pneumothorax"
    }
    
    for variation, canonical in label_variations.items():
        if variation in text_lower:
            found_labels.add(canonical)
    
    return sorted(list(found_labels))

def make_map_fn(split):
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

        # Extract coordinates and labels for reward function
        raw_answer = example["conversations"][1]["value"]
        ground_truth_coords = extract_coordinates(raw_answer)
        labels = example.get("labels") or []
        
        # Extract medical labels from both raw answer and metadata labels
        medical_labels = extract_medical_labels(raw_answer, labels)
        
        # Create self-contained ground truth structure
        has_no_finding = ("no abnormalities" in raw_answer.lower() or
                         "no finding" in raw_answer.lower() or
                         "No finding" in labels or
                         len(ground_truth_coords) == 0)

        # Enhanced ground truth with labels
        ground_truth_data = {
            "coordinates": ground_truth_coords,
            "has_no_finding": has_no_finding,
            "medical_labels": medical_labels,  # ← NEW: Medical finding labels
            "raw_answer": raw_answer  # Keep for debugging
        }

        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : [prompt_msg],              # same schema as Geo3K example
            "images"      : [img_entry],
            "ability"     : ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_data        # Self-contained dict with labels
            },
            "extra_info"  : {                          # anything you want to keep
                "id":    example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "labels": labels,
                "coord_count": len(ground_truth_coords),
                "label_count": len(medical_labels)  # ← NEW: Track label count
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999):
    with open(json_path) as f:          # ← use the file path
        rows = json.load(f)             # ← module 'json' is still accessible

    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n  = int(len(ds) * train_ratio)

    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    val_ds   = ds.select(range(n, len(ds))).map(make_map_fn("val"),   with_indices=True, num_proc=8)

    # Filter out None entries (missing images)
    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds   = val_ds.filter(lambda x: x is not None)

    # Print statistics
    print("📊 Dataset Statistics:")
    print(f"   Total samples: {len(ds)}")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples: {len(val_ds)}")
    
    # Sample statistics for labels
    sample_labels = [len(ex['reward_model']['ground_truth']['medical_labels']) for ex in train_ds.select(range(min(100, len(train_ds))))]
    print(f"   Avg labels per sample (first 100): {sum(sample_labels)/len(sample_labels):.2f}")

    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet  (os.path.join(local_dir, "val.parquet"))

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

    print(f"✅ VinDR preprocessing with labels completed! Train: {len(train_ds)}, Val: {len(val_ds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")              # ← new name
    parser.add_argument("--local_dir", default="~/data/vindr_grpo")
    parser.add_argument("--hdfs_dir",  default=None)
    main(**vars(parser.parse_args()))