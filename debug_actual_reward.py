#!/usr/bin/env python3

import re

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
    Core grounding accuracy function.
    Similar to radgraph_partial_score but for coordinate grounding.
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

    return fuzzy_map_score(pred_coords, gt_coords)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant reward function following GSM8K/RadGraph pattern.
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

# Test the exact case from the log
solution_str = """<think>
The image shows a chest X-ray with no obvious signs of a nodule or mass. However, based on the provided coordinates, the nodule/mass appears to be located in the upper right quadrant of the image, near the clavicle. The area around these coordinates shows a subtle increase in density compared to the surrounding lung tissue, which could represent the nodule.
</think>
<answer>
The nodule/mass is situated at [0.22, 0.48, 0.24, 0.5] in the image.
</answer>"""

ground_truth = {
    'coordinates': [[0.22, 0.47, 0.25, 0.5]], 
    'has_no_finding': False, 
    'raw_answer': 'The nodule/mass is positioned at [0.22, 0.47, 0.25, 0.5] on the image.'
}

print("🔍 Debugging ACTUAL Reward Function (map_reward_fuzzy2_fixed.py)")
print("=" * 60)

# Extract core answer
match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
core_answer = match.group(1).strip()
print(f"Core answer: {repr(core_answer)}")

# Test coordinate extraction
pred_coords = extract_coordinates(core_answer)
gt_coords = ground_truth['coordinates']

print(f"Predicted coords: {pred_coords}")
print(f"Ground truth coords: {gt_coords}")

if pred_coords and gt_coords:
    # Calculate IoU manually
    iou = calculate_iou(pred_coords[0], gt_coords[0])
    print(f"Raw IoU: {iou:.6f}")
    
    # Convert IoU to fuzzy score
    fuzzy_score_conversion = (iou - 0.1) / 0.9 * 0.9 + 0.1
    print(f"Fuzzy score conversion: {fuzzy_score_conversion:.6f}")
    
    # Calculate fuzzy mAP score (F1 of precision/recall)
    fuzzy_map = fuzzy_map_score(pred_coords, gt_coords)
    print(f"Fuzzy mAP F1 score: {fuzzy_map:.6f}")

# Test format reward
format_score = format_reward(solution_str)
print(f"Format reward: {format_score}")
print(f"Format bonus (10%): {format_score * 0.1:.6f}")

# Test full reward function
final_score = compute_score("vindr_grpo", solution_str, ground_truth)
print(f"Final reward score: {final_score:.6f}")

# Expected from log
print(f"Expected from log: 0.5444444444444443")

print("\n" + "=" * 60)
print("Analysis:")
print(f"- IoU: {iou:.6f}")
print(f"- Fuzzy conversion: {fuzzy_score_conversion:.6f}") 
print(f"- F1 score: {fuzzy_map:.6f}")
print(f"- Format bonus: {format_score * 0.1:.6f}")
print(f"- Final: {fuzzy_map:.6f} + {format_score * 0.1:.6f} = {final_score:.6f}")

if abs(final_score - 0.5444444444444443) < 0.001:
    print("✅ Score matches log output perfectly!")
else:
    print("❌ Score doesn't match log output")