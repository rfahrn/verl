# Copyright 2025 – Apache-2.0
"""
VinDR-CXR reward function - FIXED version with proper label handling.
Fixes: repetition, false positives/negatives, better extraction.
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

def extract_medical_labels_fixed(text):
    """
    FIXED: Better medical label extraction with deduplication and cleaning.
    """
    if nlp:
        return extract_labels_with_spacy_fixed(text)
    else:
        return extract_labels_with_regex_fixed(text)

def extract_labels_with_spacy_fixed(text):
    """FIXED: Use spaCy with better filtering and deduplication."""
    doc = nlp(text)
    labels = set()  # Use set to avoid duplicates
    
    # Method 1: Get medical noun phrases
    for chunk in doc.noun_chunks:
        label = chunk.text.lower().strip()
        
        # Remove articles and clean
        for article in ["the ", "a ", "an "]:
            if label.startswith(article):
                label = label[len(article):]
        
        # Remove common suffixes that aren't medical terms
        for suffix in [" is positioned", " is located", " is detected", " is found", " is present"]:
            if label.endswith(suffix):
                label = label[:-len(suffix)]
        
        # Only keep actual medical terms (not generic phrases)
        if (len(label) > 3 and 
            is_likely_medical_term(label) and
            label not in ["this", "that", "these", "those", "image", "x-ray", "chest", "scan", "side", "right side", "left side"]):
            labels.add(label.strip())
    
    # Method 2: Pattern-based extraction for common medical structures
    medical_patterns = [
        r'\b(cardiomegaly|pneumonia|pneumothorax|atelectasis|consolidation)\b',
        r'\b(pleural effusion|aortic enlargement|pulmonary edema|emphysema)\b',
        r'\b(nodule|mass|lesion|opacity|infiltration|fibrosis)\b',
        r'\b(tuberculosis|copd|calcification|fracture)\b'
    ]
    
    for pattern in medical_patterns:
        matches = re.findall(pattern, text.lower())
        for match in matches:
            labels.add(match.strip())
    
    return sorted(list(labels))

def extract_labels_with_regex_fixed(text):
    """FIXED: Better regex extraction with deduplication."""
    labels = set()  # Use set to avoid duplicates
    text_lower = text.lower()
    
    # Pattern 1: Medical term before location indicators (cleaned)
    patterns = [
        r'\b([a-z\s]+?)\s+(?:is\s+)?(?:positioned|located|detected|found)\s+at\s+\[',
        r'\b([a-z\s]+?)\s+(?:at\s+)?\[',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            label = match.strip()
            
            # Clean the label
            for prefix in ["the ", "a ", "an "]:
                if label.startswith(prefix):
                    label = label[len(prefix):]
            
            # Only keep if it looks medical and isn't too generic
            if (len(label) > 3 and 
                is_likely_medical_term(label) and
                label not in ["image", "x-ray", "chest", "scan", "side", "right side", "left side"]):
                labels.add(label.strip())
    
    return sorted(list(labels))

def is_likely_medical_term(term):
    """
    Check if a term is likely a medical condition.
    FIXED: Better filtering to avoid generic phrases.
    """
    term = term.lower().strip()
    
    # Medical suffixes
    medical_suffixes = ['megaly', 'osis', 'itis', 'oma', 'pathy', 'trophy', 'emia', 'uria']
    if any(term.endswith(suffix) for suffix in medical_suffixes):
        return True
    
    # Medical prefixes
    medical_prefixes = ['pneumo', 'cardio', 'pulmo', 'pleural', 'aortic', 'pulmonary']
    if any(term.startswith(prefix) for prefix in medical_prefixes):
        return True
    
    # Common medical terms
    medical_terms = [
        'effusion', 'enlargement', 'nodule', 'mass', 'lesion', 'opacity', 
        'infiltration', 'consolidation', 'atelectasis', 'emphysema', 'fibrosis',
        'fracture', 'calcification', 'tuberculosis', 'pneumonia'
    ]
    if any(med_term in term for med_term in medical_terms):
        return True
    
    # Multi-word medical phrases (often compound conditions)
    if len(term.split()) > 1 and not any(generic in term for generic in ['is', 'at', 'the', 'and', 'or']):
        return True
    
    return False

def normalize_label(label):
    """FIXED: Better label normalization."""
    label = label.lower().strip()
    
    # Handle common variations
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
    }
    
    return normalizations.get(label, label)

def calculate_label_score_fixed(pred_labels, gt_labels):
    """
    FIXED: More strict label scoring that heavily penalizes false positives/negatives.
    """
    if not gt_labels and not pred_labels:
        return 1.0  # Perfect no-finding case
    
    if not gt_labels or not pred_labels:
        return 0.0  # No reward if mismatch in presence
    
    # Normalize and deduplicate
    pred_norm = list(set([normalize_label(label) for label in pred_labels]))
    gt_norm = list(set([normalize_label(label) for label in gt_labels]))
    
    # Calculate exact matches only (stricter)
    exact_matches = 0
    pred_set = set(pred_norm)
    gt_set = set(gt_norm)
    
    for pred in pred_set:
        if pred in gt_set:
            exact_matches += 1
    
    # FIXED: Stricter scoring
    if exact_matches == 0:
        return 0.0  # No partial credit for wrong labels
    
    # Calculate precision and recall
    precision = exact_matches / len(pred_set)
    recall = exact_matches / len(gt_set)
    
    # FIXED: Penalize false positives and negatives more heavily
    if precision < 1.0 or recall < 1.0:
        # Heavy penalty for hallucinations or missed labels
        penalty = 0.5 * (1.0 - precision) + 0.5 * (1.0 - recall)
        base_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return max(0.0, base_score - penalty)
    
    # Perfect match
    return 1.0

def fuzzy_map_score(pred_coords, gt_coords):
    """Calculate fuzzy mAP score for coordinates (unchanged)."""
    if not gt_coords and not pred_coords:
        return 1.0
    if not gt_coords or not pred_coords:
        return 0.0

    iou_matrix = [
        [calculate_iou(pred_box, gt_box) for gt_box in gt_coords]
        for pred_box in pred_coords
    ]

    assigned_gt = set()
    assigned_pred = set()
    total_score = 0.0

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

    precision = total_score / len(pred_coords) if pred_coords else 0.0
    recall = total_score / len(gt_coords) if gt_coords else 0.0

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    FIXED: Main reward function with better label handling.
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        answer_text = match.group(1).strip() if match else solution_str

        pred_coords = extract_coordinates(answer_text)
        pred_labels = extract_medical_labels_fixed(answer_text)  # FIXED extraction

        if isinstance(ground_truth, dict):
            gt_coords = ground_truth.get("coordinates", [])
            gt_labels = ground_truth.get("labels", [])
            is_no_finding = ground_truth.get("has_no_finding", False)
        else:
            gt_coords = extract_coordinates(str(ground_truth))
            gt_labels = []
            is_no_finding = len(gt_coords) == 0

        if is_no_finding:
            has_no_finding_text = any(phrase in answer_text.lower() 
                                    for phrase in ["no finding", "no abnormalities", "clear", "normal"])
            if not pred_coords and has_no_finding_text:
                return 1.0
            elif not pred_coords:
                return 0.7
            else:
                return 0.1

        # Main score: coordinate accuracy (unchanged)
        coord_score = fuzzy_map_score(pred_coords, gt_coords)
        
        # FIXED: Label bonus with stricter scoring
        label_bonus = 0.0
        if gt_labels:
            label_score = calculate_label_score_fixed(pred_labels, gt_labels)  # FIXED function
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