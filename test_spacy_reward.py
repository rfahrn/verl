#!/usr/bin/env python3

from vindr_reward_spacy import compute_score, extract_medical_entities_spacy, extract_medical_entities_regex

# Test cases with existing JSON labels (as they would come from your dataset)
test_cases = [
    {
        "name": "Perfect Match: Cardiomegaly",
        "response": """<think>
I can see an enlarged heart in this X-ray image.
</think>
<answer>
The cardiomegaly is located at [0.34, 0.52, 0.8, 0.66] on the X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'labels': ['Cardiomegaly'],  # From JSON dataset
            'raw_answer': 'The cardiomegaly is at [0.34, 0.52, 0.8, 0.66].'
        },
        "expected": "~1.0 (perfect coords + entity match + format)"
    },
    
    {
        "name": "Entity Variation: Heart Enlargement → Cardiomegaly",
        "response": """<answer>
The heart enlargement is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'labels': ['Cardiomegaly'],
            'raw_answer': 'Heart enlargement detected.'
        },
        "expected": "~1.0 (perfect coords + normalized match)"
    },
    
    {
        "name": "Multiple Findings: Pneumonia + Pleural Effusion",
        "response": """<answer>
Found pneumonia at [0.2, 0.3, 0.6, 0.7] and pleural effusion at [0.1, 0.8, 0.9, 0.95].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.2, 0.3, 0.6, 0.7], [0.1, 0.8, 0.9, 0.95]], 
            'has_no_finding': False,
            'labels': ['Pneumonia', 'Pleural effusion'],
            'raw_answer': 'Pneumonia and pleural effusion present.'
        },
        "expected": "~1.0 (perfect coords + multiple entities)"
    },
    
    {
        "name": "Partial Match: Correct Coords, Wrong Entity",
        "response": """<answer>
The pneumothorax is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'labels': ['Cardiomegaly'],
            'raw_answer': 'Cardiomegaly present.'
        },
        "expected": "~0.55 (perfect coords but no entity bonus)"
    },
    
    {
        "name": "No Finding Case",
        "response": """<answer>
No abnormalities detected on this X-ray.
</answer>""",
        "ground_truth": {
            'coordinates': [], 
            'has_no_finding': True,
            'labels': ['No finding'],
            'raw_answer': 'No abnormalities.'
        },
        "expected": "~0.55 (perfect no-finding + format)"
    },
    
    {
        "name": "Complex Medical Language",
        "response": """<think>
The patient shows signs of pulmonary edema with bilateral infiltrates.
</think>
<answer>
Pulmonary edema with infiltration at [0.1, 0.2, 0.9, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.1, 0.2, 0.9, 0.8]], 
            'has_no_finding': False,
            'labels': ['Edema', 'Infiltration'],
            'raw_answer': 'Pulmonary edema and infiltration.'
        },
        "expected": "~1.0 (coords + multiple medical entities)"
    },
    
    {
        "name": "Abbreviation Test: TB → Tuberculosis",
        "response": """<answer>
Signs of TB detected at [0.3, 0.4, 0.7, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.3, 0.4, 0.7, 0.8]], 
            'has_no_finding': False,
            'labels': ['Tuberculosis'],
            'raw_answer': 'Tuberculosis present.'
        },
        "expected": "~1.0 (coords + abbreviation normalization)"
    }
]

print("🧪 Testing spaCy-Based Medical Entity Extraction Reward Function")
print("=" * 80)

# Check spaCy availability
try:
    import spacy
    spacy_available = True
    print("✅ spaCy available - using advanced NLP entity extraction")
except ImportError:
    spacy_available = False
    print("⚠️  spaCy not available - using regex fallback")

print()

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']}")
    print("-" * 60)
    
    score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.6f}")
    
    # Show entity extraction details
    import re
    match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
    core_answer = match.group(1).strip() if match else test["response"]
    
    if spacy_available:
        try:
            pred_entities = extract_medical_entities_spacy(core_answer)
        except:
            pred_entities = extract_medical_entities_regex(core_answer)
    else:
        pred_entities = extract_medical_entities_regex(core_answer)
    
    gt_labels = test["ground_truth"].get("labels", [])
    
    print(f"Predicted entities: {pred_entities}")
    print(f"Ground truth labels: {gt_labels}")
    
    status = "✅ EXCELLENT" if score >= 0.9 else "✅ GOOD" if score >= 0.7 else "⚠️  OKAY" if score >= 0.5 else "❌ POOR"
    print(f"Status: {status}")
    print()

print("=" * 80)
print("✅ spaCy-based reward function testing completed!")

print("\n🎯 Key Features:")
print("   • 🧠 Intelligent entity extraction (spaCy NLP + regex fallback)")
print("   • 🔄 Automatic normalization (heart enlargement → cardiomegaly)")
print("   • 📝 Pattern matching for medical terms")
print("   • 🏷️  Uses existing JSON labels (no hardcoded lists)")
print("   • 🎪 Noun phrase analysis for complex medical language")
print("   • 📊 F1-based entity scoring with coordinate grounding")

print("\n📈 Scoring Breakdown:")
print("   • Grounding accuracy: 0.0 - 1.0 (fuzzy mAP for coordinates)")
print("   • Medical entity bonus: 0.0 - 0.15 (15% max for label matching)")
print("   • Format bonus: 0.0 - 0.10 (10% max for <think>+<answer>)")
print("   • Total: max 1.25 → clamped to 1.0")