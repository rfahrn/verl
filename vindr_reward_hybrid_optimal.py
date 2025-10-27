# Copyright 2025 – Apache-2.0
"""
VinDR-CXR Hybrid Optimal reward function.
Ultimate approach combining the best of all methods:
1. Smooth differentiable scoring (no hard thresholds)
2. Multi-scale evaluation (coarse + fine + ultra-fine)
3. Medical semantic understanding
4. Adaptive difficulty scaling
5. Optimized for GRPO learning dynamics
"""
import re
import math

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

def smooth_iou_score(iou, temperature=3.0):
    """
    Convert IoU to smooth score with temperature scaling.
    Better learning dynamics than hard thresholds.
    """
    # Sigmoid-like transformation for smooth gradients
    return 1.0 / (1.0 + math.exp(-temperature * (iou - 0.5)))

def calculate_spatial_similarity(box1, box2):
    """
    Multi-dimensional spatial similarity beyond just IoU.
    Considers center distance, size ratio, and aspect ratio.
    """
    # IoU component
    iou = calculate_iou(box1, box2)
    
    # Center distance component
    cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    center_dist = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    proximity = math.exp(-center_dist * 4.0)  # Exponential decay
    
    # Size similarity component
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if area1 > 0 and area2 > 0:
        size_ratio = min(area1, area2) / max(area1, area2)
    else:
        size_ratio = 0.0
    
    # Aspect ratio similarity
    aspect1 = (box1[2] - box1[0]) / max(box1[3] - box1[1], 1e-6)
    aspect2 = (box2[2] - box2[0]) / max(box2[3] - box2[1], 1e-6)
    aspect_sim = min(aspect1, aspect2) / max(aspect1, aspect2)
    
    # Weighted combination
    spatial_score = (
        0.5 * iou +           # Primary: overlap
        0.25 * proximity +    # Secondary: center proximity
        0.15 * size_ratio +   # Tertiary: size consistency
        0.1 * aspect_sim      # Quaternary: shape consistency
    )
    
    return spatial_score

def get_medical_context_weight(box, condition_type="general"):
    """
    Advanced medical weighting based on condition type and location.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    
    # Condition-specific weighting
    condition_weights = {
        "cardiomegaly": 1.3,      # Heart conditions are critical
        "pneumothorax": 1.25,     # Lung collapse is urgent
        "pleural effusion": 1.2,  # Fluid accumulation important
        "pneumonia": 1.15,        # Infection needs attention
        "atelectasis": 1.1,       # Lung collapse partial
        "general": 1.0            # Default weight
    }
    
    base_weight = condition_weights.get(condition_type.lower(), 1.0)
    
    # Anatomical region modifiers
    # Critical regions get additional weighting
    if 0.3 <= cx <= 0.7 and 0.4 <= cy <= 0.8:  # Heart region
        region_modifier = 1.1
    elif 0.4 <= cx <= 0.6 and 0.1 <= cy <= 0.4:  # Upper mediastinum
        region_modifier = 1.05
    else:
        region_modifier = 1.0
    
    return base_weight * region_modifier

def adaptive_assignment_score(pred_box, gt_box, difficulty_level=1.0):
    """
    Adaptive scoring that adjusts difficulty based on training progress.
    Early training: more forgiving, later training: more strict.
    """
    spatial_sim = calculate_spatial_similarity(pred_box, gt_box)
    medical_weight = get_medical_context_weight(gt_box)
    
    # Adaptive difficulty scaling
    # difficulty_level: 0.5 (easy) -> 1.0 (normal) -> 1.5 (hard)
    adaptive_score = spatial_sim ** difficulty_level
    
    # Apply medical weighting
    final_score = adaptive_score * medical_weight
    
    return min(1.0, final_score)

def hungarian_soft_assignment(pred_coords, gt_coords, difficulty=1.0):
    """
    Soft version of Hungarian algorithm for optimal assignment.
    Uses smooth assignment weights instead of binary matching.
    """
    if not pred_coords or not gt_coords:
        return 0.0
    
    # Create assignment matrix
    assignment_matrix = []
    for pred_box in pred_coords:
        row = []
        for gt_box in gt_coords:
            score = adaptive_assignment_score(pred_box, gt_box, difficulty)
            row.append(score)
        assignment_matrix.append(row)
    
    # Soft Hungarian: find optimal soft assignment
    n_pred, n_gt = len(pred_coords), len(gt_coords)
    
    if n_pred == n_gt:
        # Perfect case: try all permutations (for small n)
        if n_pred <= 3:  # Computationally feasible
            best_score = 0.0
            import itertools
            for perm in itertools.permutations(range(n_gt)):
                score = sum(assignment_matrix[i][perm[i]] for i in range(n_pred))
                best_score = max(best_score, score)
            return best_score / n_pred
        else:
            # Greedy approximation for larger cases
            return greedy_soft_assignment(assignment_matrix)
    else:
        # Unequal sizes: use greedy with penalty
        base_score = greedy_soft_assignment(assignment_matrix)
        size_penalty = abs(n_pred - n_gt) * 0.1
        return max(0.0, base_score - size_penalty)

def greedy_soft_assignment(assignment_matrix):
    """Greedy soft assignment with smooth scores."""
    n_pred = len(assignment_matrix)
    n_gt = len(assignment_matrix[0]) if assignment_matrix else 0
    
    if n_pred == 0 or n_gt == 0:
        return 0.0
    
    used_gt = set()
    total_score = 0.0
    
    # Sort all assignments by score
    all_assignments = []
    for i in range(n_pred):
        for j in range(n_gt):
            all_assignments.append((assignment_matrix[i][j], i, j))
    all_assignments.sort(reverse=True)
    
    used_pred = set()
    for score, pred_idx, gt_idx in all_assignments:
        if pred_idx not in used_pred and gt_idx not in used_gt:
            total_score += score
            used_pred.add(pred_idx)
            used_gt.add(gt_idx)
    
    return total_score / max(n_pred, n_gt)

def multi_scale_hybrid_score(pred_coords, gt_coords):
    """
    Multi-scale hybrid scoring combining all best practices.
    """
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect no-finding case
    if not gt_coords or not pred_coords:
        return 0.0  # Missing predictions or ground truth
    
    # Scale 1: Coarse-level (are we in the right ballpark?)
    coarse_score = hungarian_soft_assignment(pred_coords, gt_coords, difficulty=0.7)
    
    # Scale 2: Fine-level (precise localization)
    fine_score = hungarian_soft_assignment(pred_coords, gt_coords, difficulty=1.0)
    
    # Scale 3: Ultra-fine (perfect alignment)
    ultra_fine_score = hungarian_soft_assignment(pred_coords, gt_coords, difficulty=1.3)
    
    # Hierarchical combination: coarse enables fine enables ultra-fine
    hierarchical_score = (
        0.3 * coarse_score +           # Foundation: general area
        0.5 * fine_score +             # Main: precise localization  
        0.2 * ultra_fine_score         # Bonus: perfect alignment
    )
    
    # Medical consistency bonus for multi-box cases
    consistency_bonus = 0.0
    if len(pred_coords) > 1 and len(gt_coords) > 1:
        # Bilateral symmetry detection
        if len(pred_coords) == 2 and len(gt_coords) == 2:
            pred_centers_x = [(box[0] + box[2])/2 for box in pred_coords]
            gt_centers_x = [(box[0] + box[2])/2 for box in gt_coords]
            
            pred_bilateral = abs(pred_centers_x[0] - pred_centers_x[1])
            gt_bilateral = abs(gt_centers_x[0] - gt_centers_x[1])
            
            if pred_bilateral > 0.2 and gt_bilateral > 0.2:  # Likely bilateral
                bilateral_consistency = 1.0 - abs(pred_bilateral - gt_bilateral)
                consistency_bonus = 0.05 * max(0, bilateral_consistency)
    
    return min(1.0, hierarchical_score + consistency_bonus)

def extract_coordinates(text):
    """Extract coordinate boxes from text."""
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

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
    Core grounding accuracy using hybrid optimal scoring.
    """
    pred_coords = extract_coordinates(pred_text)

    if isinstance(gt_data, dict):
        gt_coords = gt_data.get("coordinates", [])
        is_no_finding = gt_data.get("has_no_finding", False)
    else:
        gt_coords = extract_coordinates(str(gt_data))
        is_no_finding = len(gt_coords) == 0

    if is_no_finding:
        phrases = ["no finding", "no abnormalities", "clear", "normal"]
        has_no_finding_text = any(phrase in pred_text.lower() for phrase in phrases)

        if not pred_coords and has_no_finding_text:
            return 1.0
        if not pred_coords:
            return 0.7
        return 0.1

    return multi_scale_hybrid_score(pred_coords, gt_coords)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant hybrid optimal reward function.

    Args:
        data_source: Dataset identifier
        solution_str: Model's complete response
        ground_truth: Ground truth data (dict or string)
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

        format_bonus = format_reward(solution_str) * 0.1  # Up to 10% bonus
        final_score = accuracy_score + format_bonus

        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0