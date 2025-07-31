#!/usr/bin/env python3

from vindr_reward_smart_nlp import compute_score, extract_medical_entities_from_structure

# Test cases showing intelligent entity extraction from sentence structure
test_cases = [
    {
        "name": "Classic VinDR Structure: 'The X is positioned at [coords]'",
        "response": """<answer>
The pulmonary fibrosis is positioned at [0.34, 0.14, 0.39, 0.22] on the image.
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.14, 0.39, 0.22]], 
            'has_no_finding': False,
            'labels': ['Pulmonary fibrosis'],
            'raw_answer': 'The pulmonary fibrosis is positioned at [0.34, 0.14, 0.39, 0.22] on the image.'
        },
        "expected": "~1.0 (perfect structure parsing)"
    },
    
    {
        "name": "Alternative Structure: 'X detected at [coords]'",
        "response": """<answer>
Cardiomegaly detected at [0.3, 0.4, 0.7, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.3, 0.4, 0.7, 0.8]], 
            'has_no_finding': False,
            'labels': ['Cardiomegaly'],
            'raw_answer': 'Cardiomegaly detected at [0.3, 0.4, 0.7, 0.8].'
        },
        "expected": "~1.0 (structure parsing without 'The')"
    },
    
    {
        "name": "Complex Medical Phrase: Multi-word condition",
        "response": """<answer>
The interstitial lung disease is located at [0.1, 0.2, 0.9, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.1, 0.2, 0.9, 0.8]], 
            'has_no_finding': False,
            'labels': ['Interstitial lung disease'],
            'raw_answer': 'The interstitial lung disease is located at [0.1, 0.2, 0.9, 0.8].'
        },
        "expected": "~1.0 (multi-word medical phrase)"
    },
    
    {
        "name": "Variation Handling: Heart enlargement → Cardiomegaly",
        "response": """<answer>
The heart enlargement is at [0.34, 0.52, 0.8, 0.66].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
            'has_no_finding': False,
            'labels': ['Cardiomegaly'],
            'raw_answer': 'The cardiomegaly is at [0.34, 0.52, 0.8, 0.66].'
        },
        "expected": "~1.0 (variation normalization)"
    },
    
    {
        "name": "Multiple Entities: Two conditions",
        "response": """<answer>
The pneumonia is found at [0.2, 0.3, 0.6, 0.7] and pleural effusion at [0.1, 0.8, 0.9, 0.95].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.2, 0.3, 0.6, 0.7], [0.1, 0.8, 0.9, 0.95]], 
            'has_no_finding': False,
            'labels': ['Pneumonia', 'Pleural effusion'],
            'raw_answer': 'Pneumonia and pleural effusion present.'
        },
        "expected": "~1.0 (multiple entity extraction)"
    },
    
    {
        "name": "Medical Suffix Recognition: -megaly, -osis",
        "response": """<answer>
Hepatomegaly is present at [0.4, 0.5, 0.8, 0.9].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.4, 0.5, 0.8, 0.9]], 
            'has_no_finding': False,
            'labels': ['Hepatomegaly'],
            'raw_answer': 'Hepatomegaly present.'
        },
        "expected": "~1.0 (medical suffix recognition)"
    },
    
    {
        "name": "No Hardcoded Lists: Novel medical term",
        "response": """<answer>
The bronchiectasis is positioned at [0.2, 0.3, 0.7, 0.6].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.2, 0.3, 0.7, 0.6]], 
            'has_no_finding': False,
            'labels': ['Bronchiectasis'],
            'raw_answer': 'The bronchiectasis is positioned at [0.2, 0.3, 0.7, 0.6].'
        },
        "expected": "~1.0 (novel term via structure)"
    },
    
    {
        "name": "Dependency Parsing: 'Patient shows signs of X'",
        "response": """<answer>
Patient shows signs of tuberculosis at [0.3, 0.4, 0.7, 0.8].
</answer>""",
        "ground_truth": {
            'coordinates': [[0.3, 0.4, 0.7, 0.8]], 
            'has_no_finding': False,
            'labels': ['Tuberculosis'],
            'raw_answer': 'Tuberculosis detected.'
        },
        "expected": "~1.0 (dependency parsing)"
    }
]

print("🧠 Testing Intelligent NLP-Based Medical Entity Extraction")
print("=" * 80)
print("🎯 Key: NO hardcoded medical term lists - uses sentence structure!")
print()

# Check spaCy availability
try:
    import spacy
    spacy_available = True
    print("✅ spaCy available - using advanced NLP structure analysis")
except ImportError:
    spacy_available = False
    print("⚠️  spaCy not available - using regex structure parsing")

print()

for i, test in enumerate(test_cases, 1):
    print(f"{i}. {test['name']}")
    print("-" * 60)
    
    score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
    
    print(f"Expected: {test['expected']}")
    print(f"Actual:   {score:.6f}")
    
    # Show intelligent entity extraction
    import re
    match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
    core_answer = match.group(1).strip() if match else test["response"]
    
    pred_entities = extract_medical_entities_from_structure(core_answer)
    gt_entities_from_text = extract_medical_entities_from_structure(test["ground_truth"]["raw_answer"])
    gt_labels = test["ground_truth"].get("labels", [])
    
    print(f"📝 Response: '{core_answer}'")
    print(f"🧠 Extracted entities: {pred_entities}")
    print(f"🎯 GT from text: {gt_entities_from_text}")
    print(f"🏷️  GT labels: {gt_labels}")
    
    status = "✅ EXCELLENT" if score >= 0.9 else "✅ GOOD" if score >= 0.7 else "⚠️  OKAY" if score >= 0.5 else "❌ POOR"
    print(f"Status: {status}")
    print()

print("=" * 80)
print("✅ Intelligent NLP entity extraction testing completed!")

print("\n🧠 Intelligent Features Demonstrated:")
print("   • 🏗️  Sentence structure analysis (noun phrases before location indicators)")
print("   • 🔍 Dependency parsing (subjects of medical verbs)")
print("   • 📝 Medical suffix/prefix recognition (-megaly, -osis, pneumo-, cardio-)")
print("   • 🎪 Multi-word medical phrase extraction")
print("   • 🔄 Automatic variation normalization")
print("   • 🚫 NO hardcoded medical term lists!")

print("\n📊 How It Works:")
print("   1. 🎯 Finds patterns: 'The [X] is at/positioned at/located at [coords]'")
print("   2. 🧠 Uses spaCy noun chunks and dependency parsing")
print("   3. 🏥 Recognizes medical-sounding terms by structure")
print("   4. 🔄 Normalizes variations intelligently")
print("   5. 📈 Calculates fuzzy similarity for robust matching")

print("\n🎯 This approach works with ANY medical condition that follows")
print("   the VinDR sentence structure - no hardcoded lists needed!")