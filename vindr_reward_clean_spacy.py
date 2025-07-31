# Copyright 2025 – Apache-2.0
"""
VinDR-CXR reward function with clean spaCy-based medical label extraction.
Simple, straightforward approach focused on label matching.
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

def extract_medical_labels_spacy(text):
    """
    Extract medical labels using spaCy - clean and simple approach.
    """
    if nlp is None:
        return extract_medical_labels_regex(text)
    
    doc = nlp(text)
    labels = set()
    
    # Method 1: Get all noun chunks (medical terms are usually noun phrases)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        # Remove articles
        for article in ["the ", "a ", "an "]:
            if chunk_text.startswith(article):
                chunk_text = chunk_text[len(article):]
        
        # Keep if it looks medical (length > 3 and contains medical indicators)
        if len(chunk_text) > 3:
            labels.add(chunk_text)
    
    # Method 2: Get entities that spaCy recognizes as medical/health related
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "SYMPTOM", "BODY_PART", "CONDITION"]:
            labels.add(ent.text.lower().strip())
    
    # Method 3: Simple pattern - words before "at [coordinates]"
    for token in doc:
        if token.text.lower() in ["at", "positioned", "located"] and token.i > 0:
            # Look backwards for medical terms
            for i in range(max(0, token.i - 5), token.i):
                if doc[i].pos_ in ["NOUN", "ADJ"] and len(doc[i].text) > 3:
                    labels.add(doc[i].text.lower())
    
    return sorted(list(labels))

def extract_medical_labels_regex(text):
    """
    Simple regex fallback when spaCy not available.
    """
    text_lower = text.lower()
    labels = set()
    
    # Pattern: word(s) before "at [" or "positioned at [" etc.
    patterns = [
        r'(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:at|positioned|located)\s*\[',
        r'the\s+(\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:at|positioned|located)\s*\[',
        r'(\w+(?:\s+\w+)*)\s+(?:detected|found|present)\s*(?:at\s*)?\[',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            clean_match = match.strip()
            if len(clean_match) > 3:
                labels.add(clean_match)
    
    return sorted(list(labels))

def normalize_label(label):
    """
    Simple label normalization.
    """
    label = label.lower().strip()
    
    # Basic normalizations
    normalizations = {
        "heart enlargement": "cardiomegaly",
        "cardiac enlargement": "cardiomegaly",
        "enlarged heart": "cardiomegaly",
        "lung infection": "pneumonia",
        "collapsed lung": "pneumothorax",
        "fluid in lungs": "pleural effusion",
        "tb": "tuberculosis",
    }
    
    return normalizations.get(label, label)

def calculate_label_match_score(pred_labels, gt_labels):
    """
    Calculate how well predicted labels match ground truth labels.
    Simple F1-based scoring.
    """
    if not gt_labels and not pred_labels:
        return 1.0
    
    if not gt_labels or not pred_labels:
        return 0.0
    
    # Normalize all labels
    pred_norm = [normalize_label(label) for label in pred_labels]
    gt_norm = [normalize_label(label) for label in gt_labels]
    
    # Convert to sets for matching
    pred_set = set(pred_norm)
    gt_set = set(gt_norm)
    
    # Calculate matches
    matches = len(pred_set & gt_set)
    
    if matches == 0:
        return 0.0
    
    precision = matches / len(pred_set)
    recall = matches / len(gt_set)
    
    return 2 * precision * recall / (precision + recall)

def fuzzy_map_score(pred_coords, gt_coords):
    """
    Calculate fuzzy mAP score for coordinates.
    """
    if not gt_coords and not pred_coords:
        return 1.0
    if not gt_coords or not pred_coords:
        return 0.0

    # Create IoU matrix
    iou_matrix = [
        [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        for pred_box in pred_coords
    ]

    # Greedy assignment
    assigned_gt = set()
    assigned_pred = set()
    total_score = 0.0

    # Find matches above threshold
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

    n_pred = len(pred_coords)
    n_gt = len(gt_coords)

    precision = total_score / n_pred if n_pred > 0 else 0.0
    recall = total_score / n_gt if n_gt > 0 else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Clean, simple reward function.
    
    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data with coordinates and labels
        extra_info: Additional metadata (unused)
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        # Extract answer content
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
        else:
            answer_text = solution_str

        # Extract coordinates and labels from prediction
        pred_coords = extract_coordinates(answer_text)
        
        if nlp:
            pred_labels = extract_medical_labels_spacy(answer_text)
        else:
            pred_labels = extract_medical_labels_regex(answer_text)

        # Get ground truth data
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
            no_finding_phrases = ["no finding", "no abnormalities", "clear", "normal"]
            has_no_finding_text = any(phrase in answer_text.lower() for phrase in no_finding_phrases)
            
            if not pred_coords and has_no_finding_text:
                return 1.0
            elif not pred_coords:
                return 0.7
            else:
                return 0.1

        # Calculate coordinate score (main component)
        coord_score = fuzzy_map_score(pred_coords, gt_coords)
        
        # Calculate label bonus
        label_bonus = 0.0
        if gt_labels:
            label_score = calculate_label_match_score(pred_labels, gt_labels)
            label_bonus = label_score * 0.15  # Max 15% bonus
        
        # Format bonus
        format_bonus = 0.0
        if "<think>" in solution_str.lower() and "<answer>" in solution_str.lower():
            format_bonus = 0.1
        elif "<answer>" in solution_str.lower():
            format_bonus = 0.05

        # Final score
        final_score = coord_score + label_bonus + format_bonus
        return min(1.0, max(0.0, final_score))

    except Exception:
        return 0.0