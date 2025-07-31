# Copyright 2025 – Apache-2.0
"""
VinDR-CXR reward function - FINAL FIXED version.
Completely resolves: repetition, false positives/negatives, clean extraction.
Preserves: Your original multi-box fuzzy mAP logic.
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

def extract_medical_labels_clean(text):
    """
    FINAL: Clean medical label extraction - no repetition, clean terms only.
    """
    labels = set()  # Use set for automatic deduplication
    text_lower = text.lower()
    
    # Method 1: Direct medical term patterns (most reliable)
    medical_patterns = [
        r'\b(pleural effusion)\b',
        r'\b(cardiomegaly)\b', 
        r'\b(pneumonia)\b',
        r'\b(pneumothorax)\b',
        r'\b(atelectasis)\b',
        r'\b(consolidation)\b',
        r'\b(aortic enlargement)\b',
        r'\b(pulmonary edema)\b',
        r'\b(emphysema)\b',
        r'\b(nodule|mass|lesion)\b',
        r'\b(opacity|infiltration|fibrosis)\b',
        r'\b(tuberculosis|tb)\b',
        r'\b(copd)\b',
        r'\b(calcification)\b',
        r'\b(fracture)\b'
    ]
    
    for pattern in medical_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            labels.add(match.strip())
    
    # Method 2: Structure-based extraction (clean the results)
    structure_patterns = [
        r'\b([a-z\s]+?)\s+(?:is\s+)?(?:positioned|located|detected|found)\s+at\s+\[',
        r'(?:the\s+)?([a-z\s]+?)\s+at\s+\[',
    ]
    
    for pattern in structure_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Clean the match thoroughly
            clean_match = match.strip()
            
            # Remove articles
            for article in ["the ", "a ", "an "]:
                if clean_match.startswith(article):
                    clean_match = clean_match[len(article):].strip()
            
            # Remove trailing words that aren't medical
            stop_words = ["is", "are", "was", "were", "and", "or", "but", "with", "at", "on", "in"]
            words = clean_match.split()
            clean_words = []
            for word in words:
                if word not in stop_words:
                    clean_words.append(word)
                else:
                    break  # Stop at first stop word
            
            clean_match = " ".join(clean_words).strip()
            
            # Only add if it's a valid medical term
            if (len(clean_match) > 3 and 
                is_medical_term(clean_match) and
                clean_match not in ["image", "x-ray", "chest", "scan", "side", "right", "left"]):
                labels.add(clean_match)
    
    return sorted(list(labels))

def is_medical_term(term):
    """Check if term is a medical condition (strict filtering)."""
    term = term.lower().strip()
    
    # Known medical terms
    medical_terms = [
        'pleural effusion', 'cardiomegaly', 'pneumonia', 'pneumothorax', 
        'atelectasis', 'consolidation', 'aortic enlargement', 'pulmonary edema',
        'emphysema', 'nodule', 'mass', 'lesion', 'opacity', 'infiltration', 
        'fibrosis', 'tuberculosis', 'tb', 'copd', 'calcification', 'fracture',
        'heart enlargement', 'cardiac enlargement', 'enlarged heart',
        'lung infection', 'collapsed lung', 'fluid in lungs'
    ]
    
    # Exact match or contains medical term
    if term in medical_terms:
        return True
    
    # Check if any medical term is contained
    for med_term in medical_terms:
        if med_term in term or term in med_term:
            return True
    
    # Medical suffixes/prefixes
    medical_suffixes = ['megaly', 'osis', 'itis', 'oma', 'pathy', 'trophy', 'emia', 'uria']
    medical_prefixes = ['pneumo', 'cardio', 'pulmo', 'pleural', 'aortic', 'pulmonary']
    
    if (any(term.endswith(suffix) for suffix in medical_suffixes) or
        any(term.startswith(prefix) for prefix in medical_prefixes)):
        return True
    
    return False

def normalize_label(label):
    """Normalize labels for matching."""
    label = label.lower().strip()
    
    # Comprehensive normalizations
    normalizations = {
        "heart enlargement": "cardiomegaly",
        "cardiac enlargement": "cardiomegaly", 
        "enlarged heart": "cardiomegaly",
        "lung infection": "pneumonia",
        "collapsed lung": "pneumothorax",
        "fluid in lungs": "pleural effusion",
        "pleural fluid": "pleural effusion",
        "tb": "tuberculosis",
        "copd": "chronic obstructive pulmonary disease",
        "lung nodule": "nodule",
        "pulmonary nodule": "nodule",
        "lung mass": "mass",
        "pulmonary mass": "mass",
        "nodule": "nodule/mass",  # Normalize to VinDR format
        "mass": "nodule/mass",   # Normalize to VinDR format
    }
    
    return normalizations.get(label, label)

def calculate_label_score_strict(pred_labels, gt_labels):
    """
    FINAL: Strict label scoring - heavy penalties for errors.
    """
    if not gt_labels and not pred_labels:
        return 1.0  # Perfect no-finding case
    
    if not gt_labels or not pred_labels:
        return 0.0  # No reward for presence mismatch
    
    # Normalize and deduplicate
    pred_norm = list(set([normalize_label(label) for label in pred_labels]))
    gt_norm = list(set([normalize_label(label) for label in gt_labels]))
    
    pred_set = set(pred_norm)
    gt_set = set(gt_norm)
    
    # Count exact matches
    exact_matches = len(pred_set & gt_set)
    
    if exact_matches == 0:
        return 0.0  # No reward for wrong labels
    
    # Calculate precision and recall
    precision = exact_matches / len(pred_set)
    recall = exact_matches / len(gt_set)
    
    # Perfect match gets full score
    if precision == 1.0 and recall == 1.0:
        return 1.0
    
    # Partial matches get heavily penalized
    base_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Apply heavy penalty for any errors
    penalty = 0.7 * (1.0 - precision) + 0.7 * (1.0 - recall)  # Heavy penalty
    final_score = max(0.0, base_f1 - penalty)
    
    return final_score

def fuzzy_map_score(pred_coords, gt_coords):
    """
    YOUR ORIGINAL fuzzy mAP score - COMPLETELY PRESERVED.
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

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    FINAL: Complete reward function with fixed label handling.
    Preserves your original mAP logic, adds clean label bonus.
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        # Extract answer content
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        answer_text = match.group(1).strip() if match else solution_str

        # Extract predictions
        pred_coords = extract_coordinates(answer_text)
        pred_labels = extract_medical_labels_clean(answer_text)  # FINAL clean extraction

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

        # Main score: YOUR ORIGINAL coordinate accuracy (preserved)
        coord_score = fuzzy_map_score(pred_coords, gt_coords)
        
        # Label bonus: FIXED strict scoring
        label_bonus = 0.0
        if gt_labels:
            label_score = calculate_label_score_strict(pred_labels, gt_labels)
            label_bonus = label_score * 0.15  # Max 15% bonus

        # Format bonus
        format_bonus = 0.0
        if "<answer>" in solution_str.lower():
            format_bonus = 0.05
            if "<think>" in solution_str.lower():
                format_bonus = 0.1

        return min(1.0, coord_score + label_bonus + format_bonus)

    except Exception:
        return 0.0