# Copyright 2025 – Apache-2.0
"""
VinDR-CXR reward function - BULLETPROOF version.
Completely prevents label repetition gaming and ensures clean extraction.
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

def extract_medical_labels_bulletproof(text):
    """
    BULLETPROOF: Extract medical labels with complete anti-gaming protection.
    """
    # Step 1: Clean the text and normalize
    text_clean = text.lower().strip()
    
    # Step 2: Define exact medical terms we care about (VinDR-CXR labels)
    vindr_labels = {
        'aortic enlargement', 'atelectasis', 'cardiomegaly', 'calcification',
        'clavicle fracture', 'consolidation', 'edema', 'emphysema', 'enlarged pa',
        'interstitial lung disease', 'ild', 'infiltration', 'lung cavity',
        'lung cyst', 'lung opacity', 'mediastinal shift', 'nodule/mass', 'nodule',
        'mass', 'pulmonary fibrosis', 'pneumothorax', 'pleural thickening',
        'pleural effusion', 'rib fracture', 'other lesion', 'lung tumor',
        'pneumonia', 'tuberculosis', 'other diseases', 'copd',
        'chronic obstructive pulmonary disease'
    }
    
    # Step 3: Find medical terms using multiple methods, then deduplicate
    found_labels = set()
    
    # Method 1: Direct exact matching (most reliable)
    for label in vindr_labels:
        if label in text_clean:
            found_labels.add(label)
    
    # Method 2: Structure-based extraction (clean and validate)
    structure_patterns = [
        r'\b([a-z\s]{3,25}?)\s+(?:is\s+)?(?:positioned|located|detected|found|present)\s+(?:at\s+)?\[',
        r'(?:the\s+)?([a-z\s]{3,25}?)\s+(?:at\s+)?\[',
        r'\b([a-z\s]{3,25}?)\s+(?:is\s+)?(?:visible|apparent|shown|seen)\b'
    ]
    
    for pattern in structure_patterns:
        matches = re.findall(pattern, text_clean)
        for match in matches:
            # Aggressively clean the match
            clean_match = clean_medical_term(match.strip())
            
            # Only add if it's a known medical term
            if clean_match in vindr_labels:
                found_labels.add(clean_match)
    
    # Step 4: BULLETPROOF deduplication and normalization
    final_labels = set()
    for label in found_labels:
        normalized = normalize_medical_label(label)
        if normalized:  # Only add valid normalized labels
            final_labels.add(normalized)
    
    return sorted(list(final_labels))

def clean_medical_term(term):
    """
    BULLETPROOF: Clean extracted terms to remove junk.
    """
    term = term.lower().strip()
    
    # Remove common prefixes
    prefixes_to_remove = ['the ', 'a ', 'an ', 'this ', 'that ', 'these ', 'those ']
    for prefix in prefixes_to_remove:
        if term.startswith(prefix):
            term = term[len(prefix):].strip()
    
    # Remove common suffixes that aren't medical
    suffixes_to_remove = [
        ' is positioned', ' is located', ' is detected', ' is found', ' is present',
        ' is visible', ' is apparent', ' is shown', ' is seen', ' are positioned',
        ' are located', ' are detected', ' are found', ' are present', ' are visible'
    ]
    for suffix in suffixes_to_remove:
        if term.endswith(suffix):
            term = term[:-len(suffix)].strip()
    
    # Remove stop words from the end
    words = term.split()
    stop_words = {'is', 'are', 'was', 'were', 'and', 'or', 'but', 'with', 'at', 'on', 'in', 'the', 'a', 'an'}
    
    # Keep words until we hit a stop word
    clean_words = []
    for word in words:
        if word not in stop_words:
            clean_words.append(word)
        else:
            break
    
    return ' '.join(clean_words).strip()

def normalize_medical_label(label):
    """
    BULLETPROOF: Normalize labels to standard VinDR format.
    """
    label = label.lower().strip()
    
    # Comprehensive normalization map
    normalization_map = {
        # Heart conditions
        'heart enlargement': 'cardiomegaly',
        'cardiac enlargement': 'cardiomegaly',
        'enlarged heart': 'cardiomegaly',
        'big heart': 'cardiomegaly',
        
        # Lung conditions
        'lung infection': 'pneumonia',
        'pulmonary infection': 'pneumonia',
        'collapsed lung': 'pneumothorax',
        'lung collapse': 'pneumothorax',
        
        # Fluid conditions
        'fluid in lungs': 'pleural effusion',
        'pleural fluid': 'pleural effusion',
        'lung fluid': 'pleural effusion',
        
        # Nodules/masses
        'lung nodule': 'nodule/mass',
        'pulmonary nodule': 'nodule/mass',
        'lung mass': 'nodule/mass',
        'pulmonary mass': 'nodule/mass',
        'nodule': 'nodule/mass',
        'mass': 'nodule/mass',
        'lesion': 'nodule/mass',
        
        # Abbreviations
        'tb': 'tuberculosis',
        'copd': 'chronic obstructive pulmonary disease',
        'ild': 'interstitial lung disease',
        
        # Other conditions
        'lung opacity': 'opacity',
        'pulmonary opacity': 'opacity',
        'lung infiltration': 'infiltration',
        'pulmonary infiltration': 'infiltration',
    }
    
    # Apply normalization
    normalized = normalization_map.get(label, label)
    
    # Only return if it's a valid VinDR label
    vindr_labels = {
        'aortic enlargement', 'atelectasis', 'cardiomegaly', 'calcification',
        'clavicle fracture', 'consolidation', 'edema', 'emphysema', 'enlarged pa',
        'interstitial lung disease', 'infiltration', 'lung cavity', 'lung cyst',
        'opacity', 'mediastinal shift', 'nodule/mass', 'pulmonary fibrosis',
        'pneumothorax', 'pleural thickening', 'pleural effusion', 'rib fracture',
        'other lesion', 'lung tumor', 'pneumonia', 'tuberculosis', 'other diseases',
        'chronic obstructive pulmonary disease'
    }
    
    return normalized if normalized in vindr_labels else None

def calculate_label_score_bulletproof(pred_labels, gt_labels):
    """
    BULLETPROOF: Ultra-strict label scoring with gaming prevention.
    """
    if not gt_labels and not pred_labels:
        return 1.0  # Perfect no-finding case
    
    if not gt_labels or not pred_labels:
        return 0.0  # No reward for presence mismatch
    
    # Normalize both sets (already deduplicated by sets)
    pred_norm = {normalize_medical_label(label.lower()) for label in pred_labels}
    gt_norm = {normalize_medical_label(label.lower()) for label in gt_labels}
    
    # Remove None values from normalization
    pred_norm = {label for label in pred_norm if label is not None}
    gt_norm = {label for label in gt_norm if label is not None}
    
    if not pred_norm or not gt_norm:
        return 0.0
    
    # Calculate exact matches
    exact_matches = len(pred_norm & gt_norm)
    
    if exact_matches == 0:
        return 0.0  # No reward for wrong labels
    
    # Calculate precision and recall
    precision = exact_matches / len(pred_norm)
    recall = exact_matches / len(gt_norm)
    
    # BULLETPROOF: Only perfect matches get good scores
    if precision == 1.0 and recall == 1.0:
        return 1.0  # Perfect match
    elif precision >= 0.8 and recall >= 0.8:
        return 0.8  # Very good match
    elif precision >= 0.6 and recall >= 0.6:
        return 0.5  # Decent match
    else:
        return 0.1  # Poor match (heavy penalty)

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
    BULLETPROOF: Complete reward function with anti-gaming protection.
    """
    if data_source != "vindr_grpo":
        return 0.0

    try:
        # Extract answer content
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        answer_text = match.group(1).strip() if match else solution_str

        # Extract predictions
        pred_coords = extract_coordinates(answer_text)
        pred_labels = extract_medical_labels_bulletproof(answer_text)  # BULLETPROOF extraction

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
        
        # Label bonus: BULLETPROOF strict scoring
        label_bonus = 0.0
        if gt_labels:
            label_score = calculate_label_score_bulletproof(pred_labels, gt_labels)
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