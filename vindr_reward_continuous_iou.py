# Copyright 2025 – Apache-2.0
"""
VinDR-CXR Continuous IoU reward function - Following tutor's approach.
Simpler, more direct continuous IoU scoring vs fuzzy mAP.
"""
import re
import json
import os
from datetime import datetime

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

def extract_boxes(content):
    """Extract coordinate boxes from model response."""
    # Extract answer section if present
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
    text = answer_match.group(1) if answer_match else content
    
    # Find coordinate patterns
    pattern = r'\[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]'
    matches = re.findall(pattern, text)
    
    try:
        return [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in matches]
    except ValueError:
        return []

def continuous_iou_reward(pred_boxes, gt_boxes, alpha=0.5):
    """
    Continuous IoU reward - tutor's approach.
    
    Args:
        pred_boxes: List of predicted bounding boxes
        gt_boxes: List of ground truth bounding boxes  
        alpha: Smoothing parameter for continuous reward
        
    Returns:
        float: Continuous reward between 0.0 and 1.0
    """
    if not gt_boxes and not pred_boxes:
        return 1.0  # Perfect no-finding case
    
    if not gt_boxes or not pred_boxes:
        return 0.0  # Missing predictions or ground truth
    
    # Method 1: Average best IoU for each GT box (tutor's likely approach)
    total_iou = 0.0
    
    for gt_box in gt_boxes:
        # Find best matching predicted box for this GT box
        best_iou = 0.0
        for pred_box in pred_boxes:
            iou = calculate_iou(pred_box, gt_box)
            best_iou = max(best_iou, iou)
        total_iou += best_iou
    
    # Average IoU across all GT boxes
    avg_iou = total_iou / len(gt_boxes)
    
    # Apply continuous transformation with alpha smoothing
    # This makes the reward smoother and more continuous
    continuous_reward = (avg_iou ** alpha)
    
    return continuous_reward

def ap_05(pred_boxes, gt_boxes):
    """
    Calculate AP@0.5 for logging (like tutor's approach).
    """
    if not gt_boxes and not pred_boxes:
        return 1.0
    
    if not gt_boxes or not pred_boxes:
        return 0.0
    
    # Simple AP@0.5: count matches above 0.5 IoU threshold
    matches = 0
    for gt_box in gt_boxes:
        for pred_box in pred_boxes:
            if calculate_iou(pred_box, gt_box) >= 0.5:
                matches += 1
                break  # Each GT can only match once
    
    return matches / len(gt_boxes)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    VERL-compliant wrapper for continuous IoU approach.
    """
    if data_source != "vindr_grpo":
        return 0.0
    
    try:
        # Extract predictions
        pred_boxes = extract_boxes(solution_str)
        
        # Get ground truth coordinates
        if isinstance(ground_truth, dict):
            gt_boxes = ground_truth.get("coordinates", [])
            is_no_finding = ground_truth.get("has_no_finding", False)
        else:
            # Handle string/list format
            if isinstance(ground_truth, str):
                try:
                    gt_data = json.loads(ground_truth)
                    gt_boxes = gt_data if isinstance(gt_data, list) else []
                except:
                    gt_boxes = extract_boxes(str(ground_truth))
            else:
                gt_boxes = ground_truth if isinstance(ground_truth, list) else []
            is_no_finding = len(gt_boxes) == 0
        
        # Handle no-finding cases
        if is_no_finding:
            has_no_finding_text = any(phrase in solution_str.lower() 
                                    for phrase in ["no finding", "no abnormalities", "clear", "normal"])
            if not pred_boxes and has_no_finding_text:
                return 1.0
            elif not pred_boxes:
                return 0.7
            else:
                return 0.1
        
        # Main continuous IoU reward
        base_reward = continuous_iou_reward(pred_boxes, gt_boxes, alpha=0.5)
        
        # Format bonus (small)
        format_bonus = 0.0
        if "<answer>" in solution_str.lower():
            format_bonus = 0.05
            if "<think>" in solution_str.lower():
                format_bonus = 0.1
        
        # Debug logging (if enabled)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "/tmp/debug.log")
            now = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path.replace(".txt", "_vindr_continuous.txt"), "a", encoding="utf-8") as f:
                f.write(f"------------- {now} VinDr Continuous IoU -------------\n")
                f.write(f"Content: {solution_str[:100]}...\n")
                f.write(f"Pred boxes: {pred_boxes}\n")
                f.write(f"GT boxes: {gt_boxes}\n")
                f.write(f"Base reward: {base_reward:.4f} (continuous IoU)\n")
                f.write(f"AP@0.5: {ap_05(pred_boxes, gt_boxes):.4f}\n")
                f.write(f"Final score: {min(1.0, base_reward + format_bonus):.4f}\n\n")
        
        return min(1.0, base_reward + format_bonus)
    
    except Exception as e:
        if os.getenv("DEBUG_MODE") == "true":
            print(f"Continuous IoU reward error: {e}")
        return 0.0

# Alternative: Batch processing like tutor's original
def vindr_cxr_rewards_batch(completions, solutions, **kwargs):
    """
    Batch processing version matching tutor's signature exactly.
    For frameworks that prefer batch processing.
    """
    contents = [c[0]["content"] for c in completions] if completions else []
    rewards = []
    ap05s = []
    
    for content, sol in zip(contents, solutions):
        pred_boxes = extract_boxes(content)
        
        # Parse ground truth
        try:
            if isinstance(sol, dict):
                gt_boxes = sol.get("coordinates", [])
            else:
                clean_sol = re.sub(r'<[^>]+>', '', str(sol)).strip()
                gt_boxes = json.loads(clean_sol) if clean_sol else []
        except Exception:
            gt_boxes = []
        
        reward = continuous_iou_reward(pred_boxes, gt_boxes, alpha=0.5)
        rewards.append(reward)
        ap05s.append(ap_05(pred_boxes, gt_boxes))
    
    # Debug logging
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH", "/tmp/debug.log")
        now = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path.replace(".txt", "_vindr_batch.txt"), "a", encoding="utf-8") as f:
            f.write(f"------------- {now} VinDr Batch Rewards -------------\n")
            for content, sol, rw, ap in zip(contents, solutions, rewards, ap05s):
                f.write(f"Content: {content[:50]}...\n")
                f.write(f"Pred boxes: {extract_boxes(content)}\n")
                f.write(f"GT boxes: {sol}\n")
                f.write(f"Reward: {rw:.4f} (continuous IoU)\n")
                f.write(f"AP@0.5: {ap:.4f}\n\n")
    
    return rewards