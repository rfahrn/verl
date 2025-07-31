#!/usr/bin/env python3

from map_reward_fuzzy_with_labels import compute_score

# Test cases with labels
test_cases = [
    {
        "name": "Perfect Match with Correct Label",
        "response": """<think>
I can see an enlarged heart in this X-ray image.
</think>
<answer>
The cardiomegaly is located at [0.34, 0.52, 0.8, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'medical_labels': ['cardiomegaly']
        },
        "expected": "~1.0 (perfect coords + label + format)"
    },
    
    {
        "name": "Good Coords but Wrong Label",
        "response": """<think>
I can see an abnormality in this X-ray.
</think>
<answer>
The pneumonia is located at [0.34, 0.52, 0.8, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'medical_labels': ['cardiomegaly']
        },
        "expected": "~1.0 (perfect coords but no label bonus)"
    },
    
    {
        "name": "Close Coords with Correct Label",
        "response": """<think>
The heart appears enlarged in this image.
</think>
<answer>
The cardiomegaly is at [0.34, 0.53, 0.81, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'medical_labels': ['cardiomegaly']
        },
        "expected": "~0.65 (good coords + label bonus + format)"
    },
    
    {
        "name": "Multiple Labels - Partial Match",
        "response": """<answer>
Found cardiomegaly at [0.3, 0.2, 0.7, 0.4] and pneumonia at [0.1, 0.6, 0.4, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.3, 0.2, 0.7, 0.4], [0.1, 0.6, 0.4, 0.8]], 
            'has_no_finding': False,
            'medical_labels': ['cardiomegaly', 'pleural effusion']  # Only 1/2 labels match
        },
        "expected": "~1.0 (perfect coords + partial label bonus)"
    },
    
    {
        "name": "Label Variations - Heart Enlargement",
        "response": """<answer>
The enlarged heart is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'medical_labels': ['cardiomegaly']
        },
        "expected": "~1.0 (perfect coords + label variation match)"
    },
    
    {
        "name": "No Finding Case (No Labels)",
        "response": """<answer>
No abnormalities detected on this X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [], 
            'has_no_finding': True,
            'medical_labels': []
        },
        "expected": "~0.55 (perfect no-finding + format bonus)"
    }
]

print("🧪 Testing Enhanced Reward Function with Label Bonus")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 50)
    
    score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.6f}")
    
    # Detailed breakdown
    from map_reward_fuzzy_with_labels import extract_coordinates, extract_predicted_labels
    
    # Extract core answer
    import re
    match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
    core_answer = match.group(1).strip() if match else test["response"]
    
    pred_coords = extract_coordinates(core_answer)
    pred_labels = extract_predicted_labels(core_answer)
    gt_labels = test["ground_truth"].get("medical_labels", [])
    
    print(f"Pred coords: {pred_coords}")
    print(f"Pred labels: {pred_labels}")
    print(f"GT labels:   {gt_labels}")
    
    status = "✅ EXCELLENT" if score >= 0.9 else "✅ GOOD" if score >= 0.7 else "⚠️  OKAY" if score >= 0.5 else "❌ POOR"
    print(f"Status:   {status}")

print("\n" + "=" * 70)
print("✅ Label bonus testing completed!")

# Summary of bonuses
print("\n📊 Bonus Structure:")
print("   • Grounding accuracy: 0.0 - 1.0 (main score)")
print("   • Label bonus: 0.0 - 0.15 (15% max for correct labels)")
print("   • Format bonus: 0.0 - 0.1 (10% max for <think>+<answer>)")
print("   • Total possible: 1.25 → clamped to 1.0")