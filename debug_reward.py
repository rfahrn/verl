#!/usr/bin/env python3

import re

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes."""
    try:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    except ValueError as e:
        print(f"IoU unpacking error - box1: {box1}, box2: {box2}")
        return 0.0

    x1_inter, y1_inter = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_inter, y2_inter = min(x2_1, x2_2), min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def extract_predicted_coords(response_str):
    """Extract coordinates from model response."""
    print(f"🔍 EXTRACTING FROM: {repr(response_str)}")
    
    # Extract answer section
    answer_match = re.search(r'<answer>(.*?)</answer>', response_str, re.DOTALL | re.IGNORECASE)
    text = answer_match.group(1) if answer_match else response_str
    print(f"📝 EXTRACTED TEXT: {repr(text)}")

    # Find all coordinate patterns
    coord_pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    coordinates = re.findall(coord_pattern, text)
    print(f"🎯 RAW MATCHES: {coordinates}")

    try:
        result = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in coordinates]
        print(f"✅ PARSED COORDS: {result}")
        return result
    except ValueError as e:
        print(f"❌ Coordinate parsing error - coordinates: {coordinates}")
        return []

def calculate_fuzzy_multibox_score(pred_coords, gt_coords, min_iou=0.1):
    """Calculate fuzzy multi-box score with optimal assignment."""
    print(f"🎲 SCORING: pred={pred_coords}, gt={gt_coords}")
    
    if not gt_coords and not pred_coords:
        return 1.0  # Perfect "No finding" case
    if not gt_coords or not pred_coords:
        print(f"❌ Missing coords: gt={bool(gt_coords)}, pred={bool(pred_coords)}")
        return 0.0  # Missing predictions or ground truth

    # Create IoU matrix: pred_boxes x gt_boxes
    iou_matrix = []
    for i, pred_box in enumerate(pred_coords):
        iou_row = []
        for j, gt_box in enumerate(gt_coords):
            iou = calculate_iou(pred_box, gt_box)
            print(f"📊 IoU[{i}][{j}]: {pred_box} vs {gt_box} = {iou:.4f}")
            iou_row.append(iou)
        iou_matrix.append(iou_row)

    # Simple greedy assignment
    assigned_gt = set()
    assigned_pred = set()
    total_fuzzy_score = 0.0

    # Gather all matches above threshold
    all_matches = []
    for pred_idx, iou_row in enumerate(iou_matrix):
        for gt_idx, iou_score in enumerate(iou_row):
            if iou_score >= min_iou:
                all_matches.append((iou_score, pred_idx, gt_idx))

    print(f"🔗 VALID MATCHES: {all_matches}")

    # Sort by IoU descending (best matches first)
    all_matches.sort(key=lambda x: x[0], reverse=True)

    # Assign matches greedily
    matched_pairs = []
    for iou_score, pred_idx, gt_idx in all_matches:
        if pred_idx not in assigned_pred and gt_idx not in assigned_gt:
            # Convert IoU to fuzzy score
            fuzzy_score = (iou_score - min_iou) / (1.0 - min_iou) * 0.9 + 0.1
            print(f"✅ MATCH: pred[{pred_idx}] ↔ gt[{gt_idx}], IoU={iou_score:.4f}, fuzzy={fuzzy_score:.4f}")
            matched_pairs.append((pred_idx, gt_idx, fuzzy_score))
            assigned_pred.add(pred_idx)
            assigned_gt.add(gt_idx)
            total_fuzzy_score += fuzzy_score

    # Calculate metrics
    n_pred = len(pred_coords)
    n_gt = len(gt_coords)
    n_matched = len(matched_pairs)

    fuzzy_precision = total_fuzzy_score / n_pred if n_pred > 0 else 0.0
    fuzzy_recall = total_fuzzy_score / n_gt if n_gt > 0 else 0.0

    print(f"📈 METRICS: precision={fuzzy_precision:.4f}, recall={fuzzy_recall:.4f}")

    if fuzzy_precision + fuzzy_recall > 0:
        fuzzy_f1 = (2 * fuzzy_precision * fuzzy_recall) / (fuzzy_precision + fuzzy_recall)
    else:
        fuzzy_f1 = 0.0

    # Penalties for unmatched boxes
    unmatched_pred_penalty = (n_pred - n_matched) * 0.1
    unmatched_gt_penalty = (n_gt - n_matched) * 0.1

    print(f"⚖️  F1={fuzzy_f1:.4f}, penalties: pred={unmatched_pred_penalty:.4f}, gt={unmatched_gt_penalty:.4f}")

    # Final score: F1 with penalties, clamped to [0, 1]
    final_score = fuzzy_f1 - (unmatched_pred_penalty + unmatched_gt_penalty) / max(n_pred, n_gt)
    final_clamped = max(0.0, min(1.0, final_score))
    
    print(f"🏆 FINAL SCORE: {final_score:.4f} → {final_clamped:.4f}")
    return final_clamped

# Test with your exact example
response_str = """<think>
The heart appears enlarged, extending beyond the expected cardiac silhouette boundaries. The cardiomegaly is most prominent in the central lower portion of the image, where the heart's shadow overlaps with the diaphragm. The coordinates provided likely correspond to this area, indicating the heart's enlarged size.
</think>
<answer>
The cardiomegaly is at [0.34, 0.53, 0.81, 0.66] on the X-ray.
</answer>"""

ground_truth = {
    'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
    'has_no_finding': False, 
    'raw_answer': 'The cardiomegaly is at [0.34, 0.52, 0.8, 0.66] on the X-ray.'
}

print("=" * 60)
print("🧪 TESTING REWARD FUNCTION")
print("=" * 60)

# Extract predicted coordinates
pred_coords = extract_predicted_coords(response_str)

# Get ground truth coordinates
gt_coords = ground_truth.get("coordinates", [])
print(f"🎯 GT COORDS: {gt_coords}")

# Calculate score
score = calculate_fuzzy_multibox_score(pred_coords, gt_coords)

print("=" * 60)
print(f"🏆 FINAL RESULT: {score}")
print("=" * 60)

# Manual IoU check
if pred_coords and gt_coords:
    manual_iou = calculate_iou(pred_coords[0], gt_coords[0])
    print(f"🔍 MANUAL IoU CHECK: {manual_iou:.6f}")
    
    # Expected fuzzy score
    min_iou = 0.1
    expected_fuzzy = (manual_iou - min_iou) / (1.0 - min_iou) * 0.9 + 0.1
    print(f"📊 EXPECTED FUZZY SCORE: {expected_fuzzy:.6f}")