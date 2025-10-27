# Copyright 2025 – Apache-2.0
"""
VinDR-CXR reward function - clean, simple, straightforward approach.
Focus: Extract medical labels and match them for rewards.
"""
import re
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
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

def extract_medical_labels(text):
    """
    Extract medical labels - the cleanest, most straightforward approach.
    """
    if nlp:
        return extract_labels_with_spacy(text)
    else:
        return extract_labels_with_regex(text)

def extract_labels_with_spacy(text):
    """Use spaCy for intelligent label extraction."""
    doc = nlp(text)
    labels = []
    
    # Get noun chunks and clean them
    for chunk in doc.noun_chunks:
        label = chunk.text.lower().strip()
        # Remove common articles
        for article in ["the ", "a ", "an "]:
            if label.startswith(article):
                label = label[len(article):]
        
        # Keep meaningful medical terms (length > 3, not common words)
        if (len(label) > 3 and 
            label not in ["this", "that", "these", "those", "image", "x-ray", "chest", "scan"]):
            labels.append(label)
    
    return labels

def extract_labels_with_regex(text):
    """Simple regex extraction as fallback."""
    labels = []
    text_lower = text.lower()
    
    # Pattern: capture words before location indicators
    patterns = [
        r'the\s+([a-z\s]+?)\s+(?:is|at|positioned|located|detected|found)',
        r'([a-z\s]+?)\s+(?:detected|found|present)\s+at',
        r'([a-z\s]+?)\s+at\s+\[',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            label = match.strip()
            if len(label) > 3 and not any(word in label for word in ["image", "x-ray", "chest", "scan"]):
                labels.append(label)
    
    return labels

def normalize_label(label):
    """Simple label normalization for matching."""
    label = label.lower().strip()
    
    # Handle common variations
    if "heart" in label and ("enlarg" in label or "big" in label):
        return "cardiomegaly"
    if "lung" in label and "infection" in label:
        return "pneumonia"
    if "collapsed" in label and "lung" in label:
        return "pneumothorax"
    if "fluid" in label and "lung" in label:
        return "pleural effusion"
    if label in ["tb", "tuberculosis"]:
        return "tuberculosis"
    
    return label

def calculate_label_score(pred_labels, gt_labels):
    """Calculate label matching score."""
    if not gt_labels and not pred_labels:
        return 1.0
    if not gt_labels or not pred_labels:
        return 0.0
    
    # Normalize labels
    pred_norm = [normalize_label(label) for label in pred_labels]
    gt_norm = [normalize_label(label) for label in gt_labels]
    
    # Count matches
    matches = 0
    for pred in pred_norm:
        for gt in gt_norm:
            if pred == gt or pred in gt or gt in pred:
                matches += 1
                break
    
    # Simple F1 calculation
    precision = matches / len(pred_labels)
    recall = matches / len(gt_labels)
    
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def fuzzy_map_score(pred_coords, gt_coords):
    """Calculate fuzzy mAP score for coordinates."""
    if not gt_coords and not pred_coords:
        return 1.0
    if not gt_coords or not pred_coords:
        return 0.0

    # IoU matrix and greedy assignment
    iou_matrix = [
        [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        for pred_box in pred_coords
    ]

    assigned_gt = set()
    assigned_pred = set()
    total_score = 0.0

    # Find best matches
    matches = [
        (iou, pred_idx, gt_idx)
        for pred_idx, row in enumerate(iou_matrix)
        for gt_idx, iou in enumerate(row)
        if iou >= 0.1
    ]
    matches.sort(key=lambda x: x[0], reverse=True)

    for iou, pred_idx, gt_idx in matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            fuzzy_score = (iou - 0.1) / 0.9 * 0.9 + 0.1
            total_score += fuzzy_score
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)

    # Calculate F1
    precision = total_score / len(pred_coords) if pred_coords else 0.0
    recall = total_score / len(gt_coords) if gt_coords else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Main reward function - clean and straightforward.
    
    Returns score between 0.0 and 1.0 based on:
    - Coordinate accuracy (main score)
    - Label matching (bonus)
    - Format compliance (small bonus)
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        # Extract answer
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        answer_text = match.group(1).strip() if match else solution_str

        # Extract predictions
        pred_coords = extract_coordinates(answer_text)
        pred_labels = extract_medical_labels(answer_text)

        # Get ground truth
        if isinstance(ground_truth, dict):
            gt_coords = ground_truth.get("coordinates", [])
            gt_labels = ground_truth.get("labels", [])
            is_no_finding = ground_truth.get("has_no_finding", False)
        else:
            gt_coords = extract_coordinates(str(ground_truth))
            gt_labels = []
            is_no_finding = len(gt_coords) == 0

        # Handle no-finding cases
        if is_no_finding:
            has_no_finding_text = any(phrase in answer_text.lower() 
                                    for phrase in ["no finding", "no abnormalities", "clear", "normal"])
            if not pred_coords and has_no_finding_text:
                return 1.0
            elif not pred_coords:
                return 0.7
            else:
                return 0.1

        # Main score: coordinate accuracy
        coord_score = fuzzy_map_score(pred_coords, gt_coords)
        
        # Bonus: label matching
        label_bonus = 0.0
        if gt_labels:
            label_score = calculate_label_score(pred_labels, gt_labels)
            label_bonus = label_score * 0.15  # Max 15% bonus

        # Small format bonus
        format_bonus = 0.0
        if "<answer>" in solution_str.lower():
            format_bonus = 0.05
            if "<think>" in solution_str.lower():
                format_bonus = 0.1

        return min(1.0, coord_score + label_bonus + format_bonus)

    except Exception:
        return 0.0