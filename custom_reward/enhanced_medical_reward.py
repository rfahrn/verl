import re
import json
import numpy as np
from typing import List, Tuple, Union, Any, Dict
import logging

# Medical knowledge base for enhanced evaluation
MEDICAL_KNOWLEDGE = {
    # Anatomical regions and their clinical importance
    "anatomical_regions": {
        "lung": {"importance": 0.9, "keywords": ["lung", "pulmonary", "pleural", "bronch", "alveolar"]},
        "heart": {"importance": 1.0, "keywords": ["cardiac", "heart", "pericardial", "coronary", "aortic"]},
        "mediastinum": {"importance": 0.8, "keywords": ["mediastinal", "hilar", "lymph", "esophag"]},
        "chest_wall": {"importance": 0.6, "keywords": ["rib", "sternum", "chest wall", "intercostal"]},
        "diaphragm": {"importance": 0.7, "keywords": ["diaphragm", "phrenic"]}
    },
    
    # Abnormality types and their severity weights
    "abnormality_types": {
        "mass": {"severity": 1.0, "keywords": ["mass", "tumor", "nodule", "lesion", "growth"]},
        "pneumonia": {"severity": 0.8, "keywords": ["pneumonia", "consolidation", "infiltrate", "opacity"]},
        "pneumothorax": {"severity": 0.9, "keywords": ["pneumothorax", "collapsed lung", "air"]},
        "effusion": {"severity": 0.7, "keywords": ["effusion", "fluid", "pleural fluid"]},
        "fracture": {"severity": 0.8, "keywords": ["fracture", "break", "broken", "crack"]},
        "emphysema": {"severity": 0.6, "keywords": ["emphysema", "hyperinflation", "bullae"]},
        "atelectasis": {"severity": 0.5, "keywords": ["atelectasis", "collapse", "volume loss"]}
    },
    
    # Visual descriptors for better alignment
    "visual_descriptors": {
        "size": ["small", "large", "tiny", "massive", "moderate"],
        "shape": ["round", "oval", "irregular", "linear", "branching"],
        "density": ["dense", "faint", "subtle", "prominent", "calcified"],
        "location": ["upper", "lower", "middle", "central", "peripheral", "bilateral", "unilateral"]
    },
    
    # Clinical reasoning indicators
    "reasoning_quality": {
        "systematic": ["systematic", "methodical", "thorough", "comprehensive"],
        "differential": ["differential", "consider", "rule out", "versus"],
        "evidence": ["evidence", "findings", "suggests", "indicates", "consistent"],
        "uncertainty": ["uncertain", "unclear", "possible", "probable", "likely"]
    }
}

def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info=None) -> float:
    """
    Enhanced medical reward function with knowledge decomposition.
    
    Combines multiple evaluation criteria:
    1. Spatial accuracy (IOU)
    2. Semantic understanding (medical terminology)
    3. Anatomical context awareness
    4. Clinical reasoning quality
    5. Severity-weighted scoring
    
    Args:
        data_source: Source identifier
        solution_str: Model's complete response
        ground_truth: List of ground truth bounding boxes
        extra_info: Additional context information
    
    Returns:
        float: Comprehensive score between 0 and 1
    """
    try:
        # Extract components from the solution
        thinking_content = extract_thinking_content(solution_str)
        answer_content = extract_answer_content(solution_str)
        predicted_boxes = extract_bounding_boxes_from_answer(solution_str)
        
        # Validate ground truth
        if not isinstance(ground_truth, list):
            return 0.0
        
        # Component scores
        spatial_score = compute_spatial_score(predicted_boxes, ground_truth, answer_content)
        semantic_score = compute_semantic_score(thinking_content, answer_content, ground_truth)
        reasoning_score = compute_reasoning_score(thinking_content)
        clinical_score = compute_clinical_relevance_score(answer_content, ground_truth)
        
        # Weighted combination based on clinical importance
        if len(ground_truth) == 0:  # No finding cases
            # Emphasize semantic understanding and reasoning for negative cases
            final_score = (
                0.3 * spatial_score +      # Spatial accuracy (lower weight for negative cases)
                0.4 * semantic_score +     # Semantic understanding (higher weight)
                0.2 * reasoning_score +    # Reasoning quality
                0.1 * clinical_score       # Clinical relevance
            )
        else:  # Positive finding cases
            # Emphasize spatial accuracy and clinical relevance for positive cases
            final_score = (
                0.4 * spatial_score +      # Spatial accuracy (higher weight)
                0.3 * semantic_score +     # Semantic understanding
                0.2 * reasoning_score +    # Reasoning quality
                0.1 * clinical_score       # Clinical relevance
            )
        
        return min(1.0, max(0.0, final_score))
        
    except Exception as e:
        logging.warning(f"Error in enhanced medical reward: {e}")
        # Fallback to basic IOU scoring
        return compute_basic_iou_score(predicted_boxes, ground_truth, answer_content)


def compute_spatial_score(predicted_boxes: List[List[float]], ground_truth: List[List[float]], 
                         answer_content: str) -> float:
    """Enhanced spatial scoring with position awareness."""
    
    # Case 1: No ground truth boxes (negative cases)
    if len(ground_truth) == 0:
        if len(predicted_boxes) == 0:
            # Check for explicit negative finding statements
            negative_indicators = [
                'no finding', 'no abnormalities', 'no lesions', 'clear', 'normal',
                'no detectable', 'no visible', 'unremarkable', 'within normal limits'
            ]
            if any(phrase in answer_content.lower() for phrase in negative_indicators):
                return 1.0  # Perfect score for correctly identifying negative cases
            else:
                return 0.7  # Good score for correct prediction without explicit statement
        else:
            return 0.0  # False positive penalty
    
    # Case 2: Ground truth has boxes but no predictions
    if len(predicted_boxes) == 0:
        return 0.0  # False negative penalty
    
    # Case 3: Both have boxes - compute enhanced IOU with position weighting
    max_iou = 0.0
    total_weighted_iou = 0.0
    
    for pred_box in predicted_boxes:
        best_iou_for_pred = 0.0
        for gt_box in ground_truth:
            iou = compute_iou(pred_box, gt_box)
            best_iou_for_pred = max(best_iou_for_pred, iou)
        
        # Weight IOU by anatomical importance
        anatomical_weight = get_anatomical_importance(pred_box, answer_content)
        weighted_iou = best_iou_for_pred * anatomical_weight
        total_weighted_iou += weighted_iou
        max_iou = max(max_iou, best_iou_for_pred)
    
    # Return the better of max IOU or average weighted IOU
    avg_weighted_iou = total_weighted_iou / len(predicted_boxes)
    return max(max_iou, avg_weighted_iou)


def compute_semantic_score(thinking_content: str, answer_content: str, ground_truth: List[List[float]]) -> float:
    """Evaluate semantic understanding of medical concepts."""
    
    combined_text = f"{thinking_content} {answer_content}".lower()
    
    # Check for medical terminology usage
    terminology_score = 0.0
    terminology_count = 0
    
    for category, info in MEDICAL_KNOWLEDGE["abnormality_types"].items():
        if any(keyword in combined_text for keyword in info["keywords"]):
            terminology_score += info["severity"]
            terminology_count += 1
    
    # Normalize terminology score
    if terminology_count > 0:
        terminology_score = min(1.0, terminology_score / terminology_count)
    
    # Check for anatomical awareness
    anatomical_score = 0.0
    anatomical_count = 0
    
    for region, info in MEDICAL_KNOWLEDGE["anatomical_regions"].items():
        if any(keyword in combined_text for keyword in info["keywords"]):
            anatomical_score += info["importance"]
            anatomical_count += 1
    
    # Normalize anatomical score
    if anatomical_count > 0:
        anatomical_score = min(1.0, anatomical_score / anatomical_count)
    
    # Check for visual descriptors
    descriptor_score = 0.0
    descriptor_count = 0
    
    for desc_type, descriptors in MEDICAL_KNOWLEDGE["visual_descriptors"].items():
        if any(desc in combined_text for desc in descriptors):
            descriptor_score += 0.2  # Each descriptor type contributes 0.2
            descriptor_count += 1
    
    descriptor_score = min(1.0, descriptor_score)
    
    # Combine semantic components
    semantic_score = (
        0.4 * terminology_score +
        0.4 * anatomical_score +
        0.2 * descriptor_score
    )
    
    return semantic_score


def compute_reasoning_score(thinking_content: str) -> float:
    """Evaluate the quality of clinical reasoning process."""
    
    if not thinking_content:
        return 0.5  # Neutral score if no thinking provided
    
    thinking_lower = thinking_content.lower()
    
    reasoning_scores = []
    
    # Check for systematic approach
    systematic_indicators = MEDICAL_KNOWLEDGE["reasoning_quality"]["systematic"]
    systematic_score = min(1.0, sum(1 for indicator in systematic_indicators 
                                  if indicator in thinking_lower) * 0.3)
    reasoning_scores.append(systematic_score)
    
    # Check for differential diagnosis consideration
    differential_indicators = MEDICAL_KNOWLEDGE["reasoning_quality"]["differential"]
    differential_score = min(1.0, sum(1 for indicator in differential_indicators 
                                    if indicator in thinking_lower) * 0.4)
    reasoning_scores.append(differential_score)
    
    # Check for evidence-based reasoning
    evidence_indicators = MEDICAL_KNOWLEDGE["reasoning_quality"]["evidence"]
    evidence_score = min(1.0, sum(1 for indicator in evidence_indicators 
                                if indicator in thinking_lower) * 0.3)
    reasoning_scores.append(evidence_score)
    
    # Check for appropriate uncertainty handling
    uncertainty_indicators = MEDICAL_KNOWLEDGE["reasoning_quality"]["uncertainty"]
    uncertainty_count = sum(1 for indicator in uncertainty_indicators 
                          if indicator in thinking_lower)
    # Moderate uncertainty is good, too much or too little is problematic
    if uncertainty_count == 0:
        uncertainty_score = 0.7  # Slightly lower for overconfidence
    elif uncertainty_count <= 2:
        uncertainty_score = 1.0  # Appropriate uncertainty
    else:
        uncertainty_score = 0.6  # Too much uncertainty
    
    reasoning_scores.append(uncertainty_score)
    
    return np.mean(reasoning_scores)


def compute_clinical_relevance_score(answer_content: str, ground_truth: List[List[float]]) -> float:
    """Evaluate clinical relevance and actionability of findings."""
    
    answer_lower = answer_content.lower()
    
    # Base score
    relevance_score = 0.5
    
    # Check for actionable language
    actionable_terms = [
        "recommend", "suggest", "follow", "monitor", "urgent", "immediate",
        "further evaluation", "additional imaging", "clinical correlation"
    ]
    
    if any(term in answer_lower for term in actionable_terms):
        relevance_score += 0.2
    
    # Check for severity assessment
    severity_terms = ["severe", "mild", "moderate", "critical", "stable", "acute", "chronic"]
    if any(term in answer_lower for term in severity_terms):
        relevance_score += 0.1
    
    # Check for location specificity
    location_terms = ["location", "position", "situated", "extends", "involves"]
    if any(term in answer_lower for term in location_terms):
        relevance_score += 0.1
    
    # Penalize vague or non-specific language
    vague_terms = ["something", "unclear", "difficult to determine", "hard to say"]
    if any(term in answer_lower for term in vague_terms):
        relevance_score -= 0.2
    
    return min(1.0, max(0.0, relevance_score))


def get_anatomical_importance(bbox: List[float], context: str) -> float:
    """Determine anatomical importance based on bounding box location and context."""
    
    # Default importance
    importance = 0.7
    
    context_lower = context.lower()
    
    # Check context for anatomical region mentions
    for region, info in MEDICAL_KNOWLEDGE["anatomical_regions"].items():
        if any(keyword in context_lower for keyword in info["keywords"]):
            importance = max(importance, info["importance"])
    
    # Adjust based on bounding box characteristics
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Central locations typically more important
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Boost importance for central findings
        if 0.3 <= center_x <= 0.7 and 0.3 <= center_y <= 0.7:
            importance += 0.1
        
        # Large findings may be more significant
        if area > 0.1:  # Assuming normalized coordinates
            importance += 0.05
    
    return min(1.0, importance)


def extract_thinking_content(solution_str: str) -> str:
    """Extract content from <think> tags."""
    think_match = re.search(r"<think>(.*?)</think>", solution_str, flags=re.I|re.S)
    if think_match:
        return think_match.group(1).strip()
    return ""


def extract_answer_content(solution_str: str) -> str:
    """Extract content from <answer> tags."""
    answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, flags=re.I|re.S)
    if answer_match:
        return answer_match.group(1).strip()
    return ""


def extract_bounding_boxes_from_answer(solution_str: str) -> List[List[float]]:
    """Extract bounding boxes from the model's answer section."""
    answer_content = extract_answer_content(solution_str)
    if not answer_content:
        return []
    
    # Look for bounding box patterns like [x1, y1, x2, y2]
    box_pattern = r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]'
    matches = re.findall(box_pattern, answer_content)
    
    boxes = []
    for match in matches:
        try:
            x1, y1, x2, y2 = [float(coord) for coord in match]
            boxes.append([x1, y1, x2, y2])
        except ValueError:
            continue
    
    return boxes


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Ensure boxes are valid
    if x1_1 >= x2_1 or y1_1 >= y2_1 or x1_2 >= x2_2 or y1_2 >= y2_2:
        return 0.0
    
    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    
    # Calculate areas
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def compute_basic_iou_score(predicted_boxes: List[List[float]], ground_truth: List[List[float]], 
                           answer_content: str) -> float:
    """Fallback to basic IOU scoring if enhanced scoring fails."""
    
    if len(ground_truth) == 0:
        if len(predicted_boxes) == 0:
            negative_indicators = ['no finding', 'no abnormalities', 'normal', 'clear']
            if any(phrase in answer_content.lower() for phrase in negative_indicators):
                return 0.8
            else:
                return 0.3
        else:
            return 0.0
    
    if len(predicted_boxes) == 0:
        return 0.0
    
    max_iou = 0.0
    for pred_box in predicted_boxes:
        for gt_box in ground_truth:
            iou = compute_iou(pred_box, gt_box)
            max_iou = max(max_iou, iou)
    
    return max_iou