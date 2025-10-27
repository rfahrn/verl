# Copyright 2025 – Apache-2.0
"""
VinDR-CXR fuzzy mAP reward function for medical image grounding.
Enhanced with intelligent spaCy-based entity extraction from sentence structure.
Clean implementation following GSM8K/RadGraph pattern.
"""
import re
try:
    import spacy
    # Try to load the model, fallback to basic if not available
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            nlp = spacy.load("en_core_web_lg")
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

def extract_medical_entities_from_structure(text):
    """
    Extract medical entities using spaCy's understanding of sentence structure.
    Looks for patterns like "The [condition] is at/positioned at [coordinates]"
    """
    if nlp is None:
        return extract_medical_entities_basic(text)
    
    # Process text with spaCy
    doc = nlp(text)
    medical_entities = set()
    
    # Method 1: Extract noun phrases that come before location indicators
    location_indicators = ["is at", "is positioned at", "is located at", "at", "detected at", "found at", "present at"]
    
    for sent in doc.sents:
        sent_text = sent.text.lower()
        
        # Check if sentence contains location indicators
        for indicator in location_indicators:
            if indicator in sent_text:
                # Find noun phrases before the location indicator
                for chunk in sent.noun_chunks:
                    chunk_text = chunk.text.lower().strip()
                    
                    # Skip common articles and pronouns
                    if chunk_text in ["the", "a", "an", "this", "that", "these", "those"]:
                        continue
                    
                    # Check if this chunk comes before coordinates
                    chunk_end = chunk.end_char
                    indicator_start = sent_text.find(indicator)
                    
                    if chunk_end <= indicator_start + len(indicator):
                        # Clean the chunk (remove articles)
                        clean_chunk = chunk_text
                        for article in ["the ", "a ", "an "]:
                            if clean_chunk.startswith(article):
                                clean_chunk = clean_chunk[len(article):]
                        
                        if len(clean_chunk) > 2:  # Avoid very short chunks
                            medical_entities.add(clean_chunk)
                break
    
    # Method 2: Look for medical-sounding noun phrases (using POS tags)
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower().strip()
        
        # Skip articles and pronouns
        if chunk_text in ["the", "a", "an", "this", "that", "these", "those"]:
            continue
        
        # Clean the chunk
        clean_chunk = chunk_text
        for article in ["the ", "a ", "an "]:
            if clean_chunk.startswith(article):
                clean_chunk = clean_chunk[len(article):]
        
        # Check if it looks medical (contains medical-sounding suffixes or is multi-word)
        medical_suffixes = ["osis", "itis", "oma", "pathy", "trophy", "megaly", "emia", "uria"]
        medical_prefixes = ["pneumo", "cardio", "pulmo", "pleural", "aortic"]
        
        is_medical = (
            any(clean_chunk.endswith(suffix) for suffix in medical_suffixes) or
            any(clean_chunk.startswith(prefix) for prefix in medical_prefixes) or
            len(clean_chunk.split()) > 1  # Multi-word phrases often medical
        )
        
        if is_medical and len(clean_chunk) > 3:
            medical_entities.add(clean_chunk)
    
    # Method 3: Use dependency parsing to find subjects of medical sentences
    for token in doc:
        if token.dep_ == "nsubj" and token.head.lemma_ in ["be", "show", "indicate", "suggest", "detect", "find"]:
            # This is likely a subject describing a medical condition
            # Get the full noun phrase
            for chunk in doc.noun_chunks:
                if token in chunk:
                    chunk_text = chunk.text.lower().strip()
                    clean_chunk = chunk_text
                    for article in ["the ", "a ", "an "]:
                        if clean_chunk.startswith(article):
                            clean_chunk = clean_chunk[len(article):]
                    
                    if len(clean_chunk) > 2:
                        medical_entities.add(clean_chunk)
                    break
    
    return sorted(list(medical_entities))

def extract_medical_entities_basic(text):
    """
    Basic fallback extraction when spaCy is not available.
    Uses simple patterns to extract entities from structured sentences.
    """
    text_lower = text.lower()
    medical_entities = set()
    
    # Pattern: "The [entity] is at/positioned at/located at [coordinates]"
    patterns = [
        r'the\s+([^.]+?)\s+is\s+(?:at|positioned at|located at)\s+\[',
        r'the\s+([^.]+?)\s+(?:detected|found|present)\s+at\s+\[',
        r'([^.]+?)\s+(?:detected|found|present)\s+at\s+\[',
        r'signs?\s+of\s+([^.]+?)\s+(?:detected|at)\s+\[',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            entity = match.strip()
            # Clean up the entity
            if entity and len(entity) > 2:
                medical_entities.add(entity)
    
    return sorted(list(medical_entities))

def normalize_entities_for_comparison(entities):
    """
    Normalize entities for comparison by removing common variations.
    """
    normalized = []
    for entity in entities:
        # Convert to lowercase and strip
        norm = entity.lower().strip()
        
        # Remove common prefixes/suffixes that might vary
        norm = norm.replace("pulmonary ", "").replace("lung ", "")
        norm = norm.replace("bilateral ", "").replace("left ", "").replace("right ", "")
        
        # Handle common medical term variations
        variations = {
            "heart enlargement": "cardiomegaly",
            "cardiac enlargement": "cardiomegaly", 
            "enlarged heart": "cardiomegaly",
            "lung infection": "pneumonia",
            "collapsed lung": "pneumothorax",
            "fluid in lungs": "pleural effusion",
            "tb": "tuberculosis",
        }
        
        norm = variations.get(norm, norm)
        normalized.append(norm)
    
    return normalized

def calculate_entity_similarity(pred_entities, gt_entities):
    """
    Calculate similarity between predicted and ground truth entities.
    Uses fuzzy matching to handle variations.
    """
    if not gt_entities and not pred_entities:
        return 1.0  # Perfect no-finding case
    
    if not gt_entities or not pred_entities:
        return 0.0  # Missing predictions or ground truth
    
    # Normalize both sets
    pred_norm = normalize_entities_for_comparison(pred_entities)
    gt_norm = normalize_entities_for_comparison(gt_entities)
    
    # Calculate matches
    matches = 0
    for pred in pred_norm:
        for gt in gt_norm:
            # Exact match
            if pred == gt:
                matches += 1
                break
            # Partial match (one contains the other)
            elif pred in gt or gt in pred:
                matches += 0.8  # Partial credit
                break
            # Similar words (simple heuristic)
            elif len(set(pred.split()) & set(gt.split())) > 0:
                matches += 0.5  # Some credit for shared words
                break
    
    # Calculate F1-like score
    precision = matches / len(pred_entities) if pred_entities else 0
    recall = matches / len(gt_entities) if gt_entities else 0
    
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

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
    Core grounding accuracy function with intelligent entity extraction.
    Uses spaCy to understand sentence structure for medical entity extraction.
    """
    pred_coords = extract_coordinates(pred_text)
    
    # Extract medical entities using intelligent NLP
    pred_entities = extract_medical_entities_from_structure(pred_text)
    
    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        gt_labels = gt_data.get("labels", [])
        gt_raw_answer = gt_data.get("raw_answer", "")
        is_no_finding = gt_data.get("has_no_finding", False)
        
        # Extract entities from ground truth raw answer using same method
        gt_entities_from_text = extract_medical_entities_from_structure(gt_raw_answer)
        
        # Combine with provided labels
        all_gt_entities = list(set(gt_labels + gt_entities_from_text))
        
    else:
        gt_coords = extract_coordinates(str(gt_data))
        all_gt_entities = extract_medical_entities_from_structure(str(gt_data))
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
    
    # Calculate medical entity bonus using intelligent similarity
    entity_bonus = 0.0
    if all_gt_entities:
        entity_score = calculate_entity_similarity(pred_entities, all_gt_entities)
        entity_bonus = entity_score * 0.15  # Max 15% bonus for medical entities
    
    return grounding_score + entity_bonus

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function following GSM8K/RadGraph pattern.
    Enhanced with intelligent spaCy-based entity extraction from sentence structure.

    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict with coordinates, labels, raw_answer)
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
        
        final_score = accuracy_score + format_bonus

        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0