#!/usr/bin/env python3

from vindr_fuzzy_map_reward import compute_score

# Test cases
test_cases = [
    {
        "name": "Perfect Match with Formatting",
        "solution": """<think>
I need to identify the cardiomegaly in this X-ray image.
The heart appears enlarged in the central area.
</think>
<answer>
The cardiomegaly is located at [0.34, 0.52, 0.8, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False
        },
        "expected": "~1.0"
    },
    
    {
        "name": "Close Match (Your Original Example)",
        "solution": """<think>
The heart appears enlarged, extending beyond the expected cardiac silhouette boundaries.
</think>
<answer>
The cardiomegaly is at [0.34, 0.53, 0.81, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False
        },
        "expected": "~0.95"
    },
    
    {
        "name": "No Finding Case",
        "solution": """<think>
I examined the X-ray carefully for any abnormalities.
</think>
<answer>
No finding detected on this X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [], 
            'has_no_finding': True
        },
        "expected": "~1.0"
    },
    
    {
        "name": "Multiple Boxes",
        "solution": """<answer>
Found cardiomegaly at [0.34, 0.52, 0.8, 0.66] and pleural effusion at [0.1, 0.7, 0.3, 0.9].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66], [0.1, 0.7, 0.3, 0.9]], 
            'has_no_finding': False
        },
        "expected": "~1.0"
    },
    
    {
        "name": "No Answer Tags (Penalized)",
        "solution": "The cardiomegaly is at [0.34, 0.52, 0.8, 0.66].",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False
        },
        "expected": "~0.5"
    }
]

print("🧪 Testing Clean VinDR Reward Function")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 40)
    
    score = compute_score("vindr_grpo", test["solution"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.4f}")
    print(f"Status:   {'✅ PASS' if score > 0.8 else '⚠️  CHECK' if score > 0.4 else '❌ FAIL'}")

print("\n" + "=" * 60)
print("✅ Clean reward function testing completed!")