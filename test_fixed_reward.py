#!/usr/bin/env python3

from map_reward_fuzzy2_fixed import compute_score

# Test with your exact example
solution_str = """<think>
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
print("🧪 TESTING FIXED REWARD FUNCTION")
print("=" * 60)

score = compute_score("vindr_grpo", solution_str, ground_truth)

print("=" * 60)
print(f"🏆 FINAL RESULT: {score}")
print("=" * 60)

# Test a few more cases
print("\n🔍 Additional Test Cases:")

# Test case 2: Perfect match
perfect_response = """<answer>The cardiomegaly is at [0.34, 0.52, 0.8, 0.66] on the X-ray.</answer>"""
perfect_score = compute_score("vindr_grpo", perfect_response, ground_truth)
print(f"Perfect match score: {perfect_score}")

# Test case 3: No coordinates extracted
no_coords_response = """<answer>I can see some abnormalities but cannot locate them precisely.</answer>"""
no_coords_score = compute_score("vindr_grpo", no_coords_response, ground_truth)
print(f"No coordinates score: {no_coords_score}")

# Test case 4: No finding case
no_finding_gt = {'coordinates': [], 'has_no_finding': True, 'raw_answer': 'No abnormalities detected.'}
no_finding_response = """<answer>No finding detected on this X-ray.</answer>"""
no_finding_score = compute_score("vindr_grpo", no_finding_response, no_finding_gt)
print(f"No finding score: {no_finding_score}")

print("\n✅ All tests completed!")