# Copyright 2025 – Apache-2.0
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs          # optional HDFS sync

DATA_SOURCE = "vindr_grpo"        # arbitrary tag used by RewardManager
ABILITY     = "medical_grounding"
PROMPT_SUFFIX = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
TARGET_TOKENS    = 1024
MAX_PIXELS    = TARGET_TOKENS * 28 * 28    # ≈0.8 Mpx  (896×896)

# Official VinDR-CXR labels (28 findings and diagnoses)
# Local labels (1-22): should be marked with bounding boxes
VINDR_LOCAL_LABELS = [
    "aortic enlargement", "atelectasis", "cardiomegaly", "calcification", 
    "clavicle fracture", "consolidation", "edema", "emphysema", 
    "enlarged pa", "interstitial lung disease", "infiltration", "lung cavity", 
    "lung cyst", "lung opacity", "mediastinal shift", "nodule/mass", 
    "pulmonary fibrosis", "pneumothorax", "pleural thickening", 
    "pleural effusion", "rib fracture", "other lesion"
]

# Global labels (23-28): diagnostic impression, no bounding boxes
VINDR_GLOBAL_LABELS = [
    "lung tumor", "pneumonia", "tuberculosis", "other diseases", 
    "chronic obstructive pulmonary disease", "copd", "no finding"
]

# All VinDR labels combined
ALL_VINDR_LABELS = VINDR_LOCAL_LABELS + VINDR_GLOBAL_LABELS

# Label variations and synonyms for robust matching
VINDR_LABEL_VARIATIONS = {
    # Aortic variations
    "aorta enlargement": "aortic enlargement",
    "enlarged aorta": "aortic enlargement",
    "aortic dilation": "aortic enlargement",
    
    # Cardiac variations
    "cardiac enlargement": "cardiomegaly",
    "heart enlargement": "cardiomegaly", 
    "enlarged heart": "cardiomegaly",
    "cardiac hypertrophy": "cardiomegaly",
    
    # Pulmonary artery variations
    "enlarged pulmonary artery": "enlarged pa",
    "pa enlargement": "enlarged pa",
    "pulmonary artery enlargement": "enlarged pa",
    
    # ILD variations
    "ild": "interstitial lung disease",
    "interstitial disease": "interstitial lung disease",
    "pulmonary interstitial disease": "interstitial lung disease",
    
    # Nodule/Mass variations
    "nodule": "nodule/mass",
    "mass": "nodule/mass",
    "lung nodule": "nodule/mass",
    "pulmonary nodule": "nodule/mass",
    "lung mass": "nodule/mass",
    "pulmonary mass": "nodule/mass",
    
    # Effusion variations
    "fluid in lungs": "pleural effusion",
    "pleural fluid": "pleural effusion",
    "effusion": "pleural effusion",
    
    # Pneumothorax variations
    "collapsed lung": "pneumothorax",
    "air in chest": "pneumothorax",
    "pneumo": "pneumothorax",
    
    # COPD variations
    "chronic obstructive pulmonary disease": "copd",
    "chronic obstructive lung disease": "copd",
    
    # Opacity variations
    "opacity": "lung opacity",
    "opacities": "lung opacity",
    "pulmonary opacity": "lung opacity",
    
    # Fracture variations
    "clavicle break": "clavicle fracture",
    "broken clavicle": "clavicle fracture",
    "rib break": "rib fracture",
    "broken rib": "rib fracture",
    
    # Other variations
    "tb": "tuberculosis",
    "pulmonary tuberculosis": "tuberculosis",
    "lung infection": "pneumonia",
    "pneumonic infiltrate": "pneumonia"
}

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

def extract_vindr_labels(text, labels_list=None):
    """
    Extract VinDR-CXR medical finding labels from text and provided labels list.
    Returns dict with local_labels and global_labels lists.
    """
    found_local = set()
    found_global = set()
    text_lower = text.lower()
    
    # Extract from provided labels list (from dataset metadata)
    if labels_list:
        for label in labels_list:
            if isinstance(label, str):
                normalized_label = label.lower().strip()
                if normalized_label and normalized_label != "no finding":
                    # Categorize into local vs global
                    if normalized_label in VINDR_LOCAL_LABELS:
                        found_local.add(normalized_label)
                    elif normalized_label in VINDR_GLOBAL_LABELS:
                        found_global.add(normalized_label)
    
    # Extract from text using VinDR label keywords
    for label in ALL_VINDR_LABELS:
        if label.lower() in text_lower:
            if label in VINDR_LOCAL_LABELS:
                found_local.add(label.lower())
            else:
                found_global.add(label.lower())
    
    # Handle label variations and synonyms
    for variation, canonical in VINDR_LABEL_VARIATIONS.items():
        if variation in text_lower:
            if canonical in VINDR_LOCAL_LABELS:
                found_local.add(canonical)
            else:
                found_global.add(canonical)
    
    return {
        "local_labels": sorted(list(found_local)),    # Labels that should have bounding boxes
        "global_labels": sorted(list(found_global))   # Diagnostic impressions
    }

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
        
        # Extract VinDR labels from both raw answer and metadata labels
        vindr_labels = extract_vindr_labels(raw_answer, labels)
        
        # Create self-contained ground truth structure
        has_no_finding = ("no abnormalities" in raw_answer.lower() or
                         "no finding" in raw_answer.lower() or
                         "No finding" in labels or
                         len(ground_truth_coords) == 0)

        # Enhanced ground truth with VinDR-specific labels
        ground_truth_data = {
            "coordinates": ground_truth_coords,
            "has_no_finding": has_no_finding,
            "local_labels": vindr_labels["local_labels"],      # ← NEW: Local findings (with boxes)
            "global_labels": vindr_labels["global_labels"],    # ← NEW: Global diagnoses
            "raw_answer": raw_answer  # Keep for debugging
        }

        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : [prompt_msg],              # same schema as Geo3K example
            "images"      : [img_entry],
            "ability"     : ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth_data        # Self-contained dict with VinDR labels
            },
            "extra_info"  : {                          # anything you want to keep
                "id":    example.get("id", f"{split}_{idx}"),
                "split": split,
                "index": idx,
                "labels": labels,
                "coord_count": len(ground_truth_coords),
                "local_label_count": len(vindr_labels["local_labels"]),    # ← NEW
                "global_label_count": len(vindr_labels["global_labels"])   # ← NEW
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
    print("📊 VinDR-CXR Dataset Statistics:")
    print(f"   Total samples: {len(ds)}")
    print(f"   Train samples: {len(train_ds)}")
    print(f"   Val samples: {len(val_ds)}")
    
    # Sample statistics for VinDR labels
    sample_size = min(100, len(train_ds))
    sample_local = [len(ex['reward_model']['ground_truth']['local_labels']) for ex in train_ds.select(range(sample_size))]
    sample_global = [len(ex['reward_model']['ground_truth']['global_labels']) for ex in train_ds.select(range(sample_size))]
    
    print(f"   Avg local labels per sample (first {sample_size}): {sum(sample_local)/len(sample_local):.2f}")
    print(f"   Avg global labels per sample (first {sample_size}): {sum(sample_global)/len(sample_global):.2f}")
    print(f"   Total VinDR labels: {len(ALL_VINDR_LABELS)} (22 local + 6 global)")

    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet  (os.path.join(local_dir, "val.parquet"))

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

    print(f"✅ VinDR-CXR preprocessing with official labels completed!")
    print(f"   Train: {len(train_ds)}, Val: {len(val_ds)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")              # ← new name
    parser.add_argument("--local_dir", default="~/data/vindr_grpo")
    parser.add_argument("--hdfs_dir",  default=None)
    main(**vars(parser.parse_args()))