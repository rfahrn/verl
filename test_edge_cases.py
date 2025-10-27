#!/usr/bin/env python3

import re

# Copy the actual reward function code
def calculate_iou(box1, box2):
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
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

def fuzzy_map_score(pred_coords, gt_coords):
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
    n_pred = len(pred_coords)
    n_gt = len(gt_coords)
    precision = total_score / n_pred if n_pred > 0 else 0.0
    recall = total_score / n_gt if n_gt > 0 else 0.0
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0

def format_reward(solution_str):
    has_think = bool(re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE))
    has_answer = bool(re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE))
    if has_think and has_answer:
        return 1.0
    if has_answer:
        return 0.5
    return 0.0

def grounding_accuracy(pred_text, gt_data):
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
    if data_source != "vindr_grpo":
        return 0.0
    try:
        match = re.search(r'<answer>(.*?)</answer>', solution_str, flags=re.IGNORECASE | re.DOTALL)
        if match:
            core_answer = match.group(1).strip()
            accuracy_score = grounding_accuracy(core_answer, ground_truth)
        else:
            accuracy_score = grounding_accuracy(solution_str, ground_truth) * 0.5
        format_bonus = format_reward(solution_str) * 0.1
        final_score = accuracy_score + format_bonus
        return min(1.0, max(0.0, final_score))
    except Exception:
        return 0.0

# Test cases from your log
test_cases = [
    {
        "name": "Perfect No Finding (From Your Log)",
        "response": """<think>
The bilateral lung fields appear clear without any visible opacities or masses. The heart size seems normal, and there are no signs of pleural effusion or pneumothorax. The bony structures, including the ribs and clavicles, show no fractures or abnormalities. The mediastinum appears unremarkable. Given these observations, there are no apparent lesions or abnormalities.
</think>
<answer>
The image does not display any abnormalities or lesions.
</answer>""",
        "ground_truth": {'coordinates': [], 'has_no_finding': True, 'raw_answer': 'No abnormalities or lesions are detected on this Chest X-ray.'},
        "expected": 1.0
    },
    
    {
        "name": "Incomplete Response (Truncated)",
        "response": """<think>
The aorta is typically located posteriorly and slightly left of midline in the thoracic cavity. In this X-ray, there appears to be increased opacity and widening in the upper mediastinum, which could indicate aortic enlargement. The coordinates provided likely correspond to this area, suggesting the aortic enlargement is situated near the upper central part of the image.
</think>
<answer>
""",  # Truncated/empty answer
        "ground_truth": {'coordinates': [[0.3, 0.2, 0.7, 0.4]], 'has_no_finding': False},
        "expected": "~0.0-0.1"
    },
    
    {
        "name": "No Finding but with False Positive",
        "response": """<answer>
I see an abnormality at [0.5, 0.5, 0.6, 0.6].
</answer>""",
        "ground_truth": {'coordinates': [], 'has_no_finding': True},
        "expected": "~0.1"
    },
    
    {
        "name": "No Finding without Explicit Statement",
        "response": """<answer>
The X-ray appears normal.
</answer>""",
        "ground_truth": {'coordinates': [], 'has_no_finding': True},
        "expected": "~0.75"  # 0.7 + 0.05 format bonus
    }
]

print("🧪 Testing Edge Cases from Your Training Log")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 40)
    
    score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.6f}")
    
    # Detailed breakdown for the no-finding case
    if "No Finding" in test["name"]:
        match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
        if match:
            core_answer = match.group(1).strip()
            print(f"Core answer: {repr(core_answer)}")
            
            pred_coords = extract_coordinates(core_answer)
            print(f"Extracted coords: {pred_coords}")
            
            phrases = ["no finding", "no abnormalities", "clear", "normal"]
            has_no_finding_text = any(phrase in core_answer.lower() for phrase in phrases)
            print(f"Has no-finding phrase: {has_no_finding_text}")
            
            format_score = format_reward(test["response"])
            print(f"Format score: {format_score}")
    
    status = "✅ PASS" if score >= 0.9 else "⚠️  CHECK" if score >= 0.5 else "❌ FAIL"
    print(f"Status:   {status}")

print("\n" + "=" * 60)
print("✅ Edge case analysis completed!")