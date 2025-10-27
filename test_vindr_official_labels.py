#!/usr/bin/env python3

from vindr_reward_official_labels import compute_score

# Test cases with official VinDR-CXR labels
test_cases = [
    {
        "name": "Local Label: Cardiomegaly with Perfect Coords",
        "response": """<think>
I can see an enlarged heart in this X-ray image.
</think>
<answer>
The cardiomegaly is located at [0.34, 0.52, 0.8, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'local_labels': ['cardiomegaly'],  # Local label (should have bounding box)
            'global_labels': []
        },
        "expected": "~1.0 (perfect coords + local label + format)"
    },
    
    {
        "name": "Global Label: Pneumonia (No Coords Expected)",
        "response": """<think>
This appears to be pneumonia based on the overall pattern.
</think>
<answer>
The patient has pneumonia based on the X-ray findings.
</answer>""",
        "ground_truth": {
            'coordinates': [], 
            'has_no_finding': False,
            'local_labels': [],
            'global_labels': ['pneumonia']  # Global label (diagnostic impression)
        },
        "expected": "~0.15 (no coords needed + global label + format)"
    },
    
    {
        "name": "Mixed Labels: Aortic Enlargement + COPD",
        "response": """<answer>
Found aortic enlargement at [0.3, 0.1, 0.7, 0.3] and patient has COPD.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.3, 0.1, 0.7, 0.3]], 
            'has_no_finding': False,
            'local_labels': ['aortic enlargement'],  # Local (needs coords)
            'global_labels': ['copd']                # Global (diagnostic)
        },
        "expected": "~1.0 (perfect coords + both label types)"
    },
    
    {
        "name": "VinDR Variation: Enlarged Heart → Cardiomegaly",
        "response": """<answer>
The enlarged heart is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'local_labels': ['cardiomegaly'],
            'global_labels': []
        },
        "expected": "~1.0 (perfect coords + variation match)"
    },
    
    {
        "name": "Multiple Local Labels: Nodule + Pleural Effusion",
        "response": """<answer>
Found nodule at [0.2, 0.3, 0.4, 0.5] and pleural effusion at [0.6, 0.7, 0.8, 0.9].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]], 
            'has_no_finding': False,
            'local_labels': ['nodule/mass', 'pleural effusion'],
            'global_labels': []
        },
        "expected": "~1.0 (perfect coords + multiple local labels)"
    },
    
    {
        "name": "Wrong Local Label (Should Get Coords Credit Only)",
        "response": """<answer>
The pneumothorax is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'local_labels': ['cardiomegaly'],  # Different from predicted
            'global_labels': []
        },
        "expected": "~1.0 (perfect coords but no label bonus)"
    },
    
    {
        "name": "No Finding Case",
        "response": """<answer>
No finding detected on this X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [], 
            'has_no_finding': True,
            'local_labels': [],
            'global_labels': ['no finding']
        },
        "expected": "~0.55 (perfect no-finding + format bonus)"
    },
    
    {
        "name": "Complex VinDR Case: ILD + TB",
        "response": """<think>
I can see interstitial patterns and signs suggesting tuberculosis.
</think>
<answer>
Interstitial lung disease at [0.1, 0.2, 0.9, 0.8] with tuberculosis.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.1, 0.2, 0.9, 0.8]], 
            'has_no_finding': False,
            'local_labels': ['interstitial lung disease'],  # Local
            'global_labels': ['tuberculosis']               # Global
        },
        "expected": "~1.0 (coords + local + global + format)"
    }
]

print("🧪 Testing Official VinDR-CXR Labels (28 Labels: 22 Local + 6 Global)")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{i}. {test['name']}")
    print("-" * 60)
    
    score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.6f}")
    
    # Detailed breakdown
    from vindr_reward_official_labels import extract_coordinates, extract_predicted_vindr_labels
    
    # Extract core answer
    import re
    match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
    core_answer = match.group(1).strip() if match else test["response"]
    
    pred_coords = extract_coordinates(core_answer)
    pred_labels = extract_predicted_vindr_labels(core_answer)
    gt_local = test["ground_truth"].get("local_labels", [])
    gt_global = test["ground_truth"].get("global_labels", [])
    
    print(f"Pred coords: {pred_coords}")
    print(f"Pred local:  {pred_labels['local_labels']}")
    print(f"Pred global: {pred_labels['global_labels']}")
    print(f"GT local:    {gt_local}")
    print(f"GT global:   {gt_global}")
    
    status = "✅ EXCELLENT" if score >= 0.9 else "✅ GOOD" if score >= 0.7 else "⚠️  OKAY" if score >= 0.5 else "❌ POOR"
    print(f"Status:   {status}")

print("\n" + "=" * 80)
print("✅ Official VinDR-CXR label testing completed!")

# Summary of VinDR-CXR structure
print("\n📊 VinDR-CXR Label Structure:")
print("   🎯 LOCAL LABELS (22) - Require bounding boxes:")
print("      • Aortic enlargement, Atelectasis, Cardiomegaly, Calcification")
print("      • Clavicle fracture, Consolidation, Edema, Emphysema")
print("      • Enlarged PA, ILD, Infiltration, Lung cavity, Lung cyst")
print("      • Lung opacity, Mediastinal shift, Nodule/Mass, Pulmonary fibrosis")
print("      • Pneumothorax, Pleural thickening, Pleural effusion, Rib fracture, Other lesion")
print()
print("   🌐 GLOBAL LABELS (6) - Diagnostic impressions:")
print("      • Lung tumor, Pneumonia, Tuberculosis, Other diseases, COPD, No finding")
print()
print("📈 Bonus Structure:")
print("   • Grounding accuracy: 0.0 - 1.0 (main fuzzy mAP score)")
print("   • Local label bonus:  0.0 - 0.10 (10% max for localized findings)")
print("   • Global label bonus: 0.0 - 0.05 (5% max for diagnostic impressions)")
print("   • Format bonus:       0.0 - 0.10 (10% max for <think>+<answer>)")
print("   • Total possible:     1.25 → clamped to 1.0")