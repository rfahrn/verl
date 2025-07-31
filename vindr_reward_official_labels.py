# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy mAP reward function for medical image grounding.
Enhanced with official VinDR-CXR 28 labels (22 local + 6 global).
Clean implementation following GSM8K/RadGraph pattern.
"""
import re

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

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def extract_coordinates(text):
    """Extract coordinate boxes from text."""
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

def extract_predicted_vindr_labels(text):
    """
    Extract VinDR-CXR medical finding labels from model prediction text.
    Returns dict with local_labels and global_labels lists.
    """
    found_local = set()
    found_global = set()
    text_lower = text.lower()
    
    # Extract VinDR labels using keyword matching
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

def calculate_label_score(pred_labels, gt_labels):
    """
    Calculate label matching score using F1-like metric.
    Returns score between 0.0 and 1.0.
    """
    if not gt_labels and not pred_labels:
        return 1.0  # Perfect no-finding case
    
    if not gt_labels or not pred_labels:
        return 0.0  # Missing predictions or ground truth
    
    # Convert to sets for intersection/union operations
    pred_set = set(pred_labels)
    gt_set = set(gt_labels)
    
    # Calculate precision, recall, F1
    intersection = pred_set & gt_set
    
    if len(intersection) == 0:
        return 0.0
    
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)
    
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def fuzzy_map_score(pred_coords, gt_coords):
    """
    Calculate fuzzy mAP score between predicted and ground truth boxes.
    Returns score between 0.0 and 1.0.
    """
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect no-finding case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth

    # Create IoU matrix between all pred/gt pairs
    iou_matrix = [
        [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        for pred_box in pred_coords
    ]

    # Greedy assignment: match each prediction to best available GT
    assigned_gt = set()
    assigned_pred = set()
    total_score = 0.0

    # Collect all matches above threshold (IoU >= 0.1)
    matches = [
        (iou, pred_idx, gt_idx)
        for pred_idx, row in enumerate(iou_matrix)
        for gt_idx, iou in enumerate(row)
        if iou >= 0.1
    ]

    # Sort by IoU (best matches first)
    matches.sort(key=lambda x: x[0], reverse=True)

    # Assign matches greedily
    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            # Convert IoU to fuzzy score: 0.1 → 0.1, 1.0 → 1.0
            fuzzy_score = (iou - 0.1) / 0.9 * 0.9 + 0.1
            total_score += fuzzy_score
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)

    n_pred = len(pred_coords)
    n_gt = len(gt_coords)

    precision = total_score / n_pred if n_pred > 0 else 0.0
    recall = total_score / n_gt if n_gt > 0 else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def format_reward(solution_str):
    """Reward for proper formatting with <think> and <answer> tags."""
    has_think = bool(re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))

    if has_think and has_answer:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0

def grounding_accuracy(pred_text, gt_data):
    """
    Core grounding accuracy function with VinDR label bonus.
    Similar to radgraph_partial_score but for coordinate grounding + VinDR labels.
    """
    pred_coords = extract_coordinates(pred_text)
    pred_vindr_labels = extract_predicted_vindr_labels(pred_text)

    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        gt_local_labels = gt_data.get("local_labels", [])
        gt_global_labels = gt_data.get("global_labels", [])
        is_no_finding = gt_data.get("has_no_finding", False)
    else:
        gt_coords = extract_coordinates(str(gt_data))
        gt_local_labels = []
        gt_global_labels = []
        is_no_finding = len(gt_coords) == 0

    if is_no_finding:
        phrases = ["no finding", "no abnormalities", "clear", "normal"]
        has_no_finding_text = any(phrase in pred_text.lower() for phrase in phrases)

        if not pred_coords and has_no_finding_text:
            return 1.0
        if not pred_coords:
            return 0.7
        return 0.1

    # Calculate main grounding score
    grounding_score = fuzzy_map_score(pred_coords, gt_coords)
    
    # Calculate VinDR label bonuses
    local_bonus = 0.0
    global_bonus = 0.0
    
    # Local labels bonus (up to 10% for labels that should have bounding boxes)
    if gt_local_labels:
        local_score = calculate_label_score(pred_vindr_labels["local_labels"], gt_local_labels)
        local_bonus = local_score * 0.10  # Max 10% bonus for local labels
    
    # Global labels bonus (up to 5% for diagnostic impressions)
    if gt_global_labels:
        global_score = calculate_label_score(pred_vindr_labels["global_labels"], gt_global_labels)
        global_bonus = global_score * 0.05  # Max 5% bonus for global labels
    
    return grounding_score + local_bonus + global_bonus

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function following GSM8K/RadGraph pattern.
    Enhanced with official VinDR-CXR 28 labels (22 local + 6 global).

    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict with coordinates and VinDR labels)
        extra_info: Additional metadata (unused)

    Returns:
        float: Score between 0.0 and 1.0
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        if match:
            core_answer = match.group(1).strip()
            accuracy_score = grounding_accuracy(core_answer, ground_truth)
        else:
            accuracy_score = grounding_accuracy(solution_str, ground_truth) * 0.5

        # Format bonus (up to 10% bonus)
        format_bonus = format_reward(solution_str) * 0.1
        
        # Total score includes:
        # - Grounding accuracy (max 1.0)
        # - Local label bonus (max 0.10) 
        # - Global label bonus (max 0.05)
        # - Format bonus (max 0.10)
        # = Max possible 1.25 → clamped to 1.0
        final_score = accuracy_score + format_bonus

        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0