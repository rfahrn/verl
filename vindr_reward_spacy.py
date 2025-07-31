# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy mAP reward function for medical image grounding.
Enhanced with spaCy-based intelligent medical entity extraction.
Clean implementation following GSM8K/RadGraph pattern.
"""
import re
try:
    import spacy
    from spacy.matcher import Matcher
    # Try to load the model, fallback to basic if not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # Fallback to basic English model
        nlp = spacy.load("en_core_web_md")
except ImportError:
    # Fallback to basic regex if spaCy not available
    nlp = None
    print("Warning: spaCy not available, using basic regex matching")

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

def extract_medical_entities_spacy(text):
    """
    Extract medical entities using spaCy NLP with medical pattern matching.
    Returns list of normalized medical terms found.
    """
    if nlp is None:
        # Fallback to basic regex if spaCy not available
        return extract_medical_entities_regex(text)
    
    # Process text with spaCy
    doc = nlp(text.lower())
    
    # Extract medical entities
    medical_entities = set()
    
    # 1. Use named entity recognition for medical terms
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "SYMPTOM", "BODY_PART", "MEDICAL_CONDITION"]:
            medical_entities.add(ent.text.lower().strip())
    
    # 2. Pattern matching for medical terms using spaCy's Matcher
    matcher = Matcher(nlp.vocab)
    
    # Define medical patterns
    medical_patterns = [
        # Cardiac conditions
        [{"LOWER": {"IN": ["cardiomegaly", "cardiac", "heart"]}},
         {"LOWER": {"IN": ["enlargement", "enlarged", "hypertrophy"]}, "OP": "?"}],
        
        # Lung conditions
        [{"LOWER": {"IN": ["pneumonia", "pneumothorax", "atelectasis", "consolidation"]}},
         {"LOWER": {"IN": ["bilateral", "left", "right"]}, "OP": "?"}],
        
        # Pleural conditions
        [{"LOWER": {"IN": ["pleural"]}},
         {"LOWER": {"IN": ["effusion", "thickening", "fluid"]}}],
        
        # Aortic conditions
        [{"LOWER": {"IN": ["aortic", "aorta"]}},
         {"LOWER": {"IN": ["enlargement", "enlarged", "dilation"]}}],
        
        # Nodules and masses
        [{"LOWER": {"IN": ["nodule", "mass", "lesion", "opacity"]}},
         {"LOWER": {"IN": ["lung", "pulmonary"]}, "OP": "?"}],
        
        # Other conditions
        [{"LOWER": {"IN": ["emphysema", "fibrosis", "edema", "infiltration"]}},
         {"LOWER": {"IN": ["pulmonary", "lung"]}, "OP": "?"}],
        
        # Fractures
        [{"LOWER": {"IN": ["fracture", "break", "broken"]}},
         {"LOWER": {"IN": ["rib", "clavicle", "bone"]}}],
        
        # General findings
        [{"LOWER": {"IN": ["calcification", "cavity", "cyst", "tumor"]}},
         {"LOWER": {"IN": ["lung", "pulmonary"]}, "OP": "?"}],
    ]
    
    # Add patterns to matcher
    for i, pattern in enumerate(medical_patterns):
        matcher.add(f"MEDICAL_PATTERN_{i}", [pattern])
    
    # Find matches
    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        medical_entities.add(span.text.lower().strip())
    
    # 3. Look for specific medical noun phrases
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        # Check if chunk contains medical keywords
        medical_keywords = [
            "cardiomegaly", "pneumonia", "pneumothorax", "atelectasis", "consolidation",
            "effusion", "enlargement", "nodule", "mass", "opacity", "emphysema",
            "fibrosis", "edema", "infiltration", "fracture", "calcification",
            "cavity", "cyst", "tumor", "tuberculosis", "copd"
        ]
        
        for keyword in medical_keywords:
            if keyword in chunk_text:
                medical_entities.add(chunk_text)
                break
    
    # 4. Clean and normalize entities
    normalized_entities = []
    for entity in medical_entities:
        entity = entity.strip()
        if len(entity) > 2 and entity not in ["the", "and", "or", "with", "of", "in", "on", "at"]:
            # Normalize common variations
            entity = normalize_medical_term(entity)
            if entity:
                normalized_entities.append(entity)
    
    return sorted(list(set(normalized_entities)))

def extract_medical_entities_regex(text):
    """
    Fallback medical entity extraction using regex patterns.
    Used when spaCy is not available.
    """
    text_lower = text.lower()
    medical_entities = set()
    
    # Basic medical term patterns
    medical_terms = [
        r'\b(cardiomegaly|cardiac enlargement|heart enlargement|enlarged heart)\b',
        r'\b(pneumonia|lung infection)\b',
        r'\b(pneumothorax|collapsed lung)\b',
        r'\b(atelectasis)\b',
        r'\b(consolidation)\b',
        r'\b(pleural effusion|fluid in lungs)\b',
        r'\b(pleural thickening)\b',
        r'\b(aortic enlargement|enlarged aorta)\b',
        r'\b(nodule|mass|lesion|opacity)\b',
        r'\b(emphysema)\b',
        r'\b(fibrosis|pulmonary fibrosis)\b',
        r'\b(edema|pulmonary edema)\b',
        r'\b(infiltration)\b',
        r'\b(fracture|break|broken)\b',
        r'\b(calcification)\b',
        r'\b(cavity|lung cavity)\b',
        r'\b(cyst|lung cyst)\b',
        r'\b(tumor|lung tumor)\b',
        r'\b(tuberculosis|tb)\b',
        r'\b(copd|chronic obstructive pulmonary disease)\b',
    ]
    
    for pattern in medical_terms:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            normalized = normalize_medical_term(match)
            if normalized:
                medical_entities.add(normalized)
    
    return sorted(list(medical_entities))

def normalize_medical_term(term):
    """
    Normalize medical terms to standard forms.
    """
    term = term.lower().strip()
    
    # Normalization mappings
    normalizations = {
        "cardiac enlargement": "cardiomegaly",
        "heart enlargement": "cardiomegaly",
        "enlarged heart": "cardiomegaly",
        "lung infection": "pneumonia",
        "collapsed lung": "pneumothorax",
        "fluid in lungs": "pleural effusion",
        "enlarged aorta": "aortic enlargement",
        "pulmonary fibrosis": "fibrosis",
        "pulmonary edema": "edema",
        "lung cavity": "cavity",
        "lung cyst": "cyst",
        "lung tumor": "tumor",
        "tb": "tuberculosis",
        "chronic obstructive pulmonary disease": "copd",
    }
    
    return normalizations.get(term, term)

def calculate_label_score(pred_labels, gt_labels):
    """
    Calculate label matching score using F1-like metric.
    Returns score between 0.0 and 1.0.
    """
    if not gt_labels and not pred_labels:
        return 1.0  # Perfect no-finding case
    
    if not gt_labels or not pred_labels:
        return 0.0  # Missing predictions or ground truth
    
    # Normalize both sets for comparison
    pred_set = set([normalize_medical_term(label) for label in pred_labels])
    gt_set = set([normalize_medical_term(label) for label in gt_labels])
    
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
    Core grounding accuracy function with spaCy-based medical entity bonus.
    Similar to radgraph_partial_score but for coordinate grounding + intelligent labels.
    """
    pred_coords = extract_coordinates(pred_text)
    
    # Extract medical entities using spaCy
    if nlp:
        pred_labels = extract_medical_entities_spacy(pred_text)
    else:
        pred_labels = extract_medical_entities_regex(pred_text)

    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        gt_labels = gt_data.get("labels", [])  # Use existing labels from JSON
        is_no_finding = gt_data.get("has_no_finding", False)
    else:
        gt_coords = extract_coordinates(str(gt_data))
        gt_labels = []
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
    
    # Calculate medical entity bonus
    label_bonus = 0.0
    if gt_labels:
        label_score = calculate_label_score(pred_labels, gt_labels)
        label_bonus = label_score * 0.15  # Max 15% bonus for medical entities
    
    return grounding_score + label_bonus

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function following GSM8K/RadGraph pattern.
    Enhanced with spaCy-based intelligent medical entity extraction.

    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict with coordinates and labels)
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
        # - Medical entity bonus (max 0.15) 
        # - Format bonus (max 0.10)
        # = Max possible 1.25 → clamped to 1.0
        final_score = accuracy_score + format_bonus

        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0