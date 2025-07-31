#!/usr/bin/env python3

from vindr_fuzzy_map_reward import compute_score, calculate_iou, fuzzy_map_score, extract_coordinates

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

print("🔍 Debugging Specific Case from Log")
print("=" * 50)

# Test coordinate extraction
pred_coords = extract_coordinates(solution_str)
gt_coords = ground_truth['coordinates']

print(f"Predicted coords: {pred_coords}")
print(f"Ground truth coords: {gt_coords}")

if pred_coords and gt_coords:
    # Calculate IoU manually
    iou = calculate_iou(pred_coords[0], gt_coords[0])
    print(f"IoU: {iou:.6f}")
    
    # Calculate fuzzy score
    fuzzy_score = fuzzy_map_score(pred_coords, gt_coords)
    print(f"Fuzzy mAP score: {fuzzy_score:.6f}")

# Test full reward function
final_score = compute_score("vindr_grpo", solution_str, ground_truth)
print(f"Final reward score: {final_score:.6f}")

# Expected from log
print(f"Expected from log: 0.5444444444444443")

print("\n" + "=" * 50)
print("Analysis:")
print(f"- Coordinates are very close (max diff: 0.01)")
print(f"- Should have high IoU (~0.95+)")
print(f"- Has proper formatting (<think> + <answer>)")
print(f"- Expected score should be ~1.0")

if abs(final_score - 0.5444444444444443) < 0.001:
    print("✅ Score matches log output")
else:
    print("❌ Score doesn't match log output")
    print("This suggests the reward function might have been different when the log was generated")