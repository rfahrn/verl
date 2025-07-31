#!/usr/bin/env python3

from vindr_reward_clean_spacy import compute_score, extract_medical_labels_spacy, extract_medical_labels_regex

def test_label_extraction():
    """Test the clean label extraction methods."""
    
    test_texts = [
        "The pulmonary fibrosis is positioned at [0.34, 0.14, 0.39, 0.22] on the image.",
        "Cardiomegaly detected at [0.3, 0.4, 0.7, 0.8].",
        "The heart enlargement is at [0.34, 0.52, 0.8, 0.66].",
        "Pneumonia found at [0.2, 0.3, 0.6, 0.7] and pleural effusion at [0.1, 0.8, 0.9, 0.95].",
        "No abnormalities detected on this X-ray."
    ]
    
    print("🧪 Testing Clean spaCy Label Extraction")
    print("=" * 60)
    
    try:
        import spacy
        spacy_available = True
        print("✅ spaCy available")
    except ImportError:
        spacy_available = False
        print("⚠️  spaCy not available - using regex")
    
    print()
    
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. Text: '{text}'")
        
        if spacy_available:
            try:
                labels = extract_medical_labels_spacy(text)
            except:
                labels = extract_medical_labels_regex(text)
        else:
            labels = extract_medical_labels_regex(text)
        
        print(f"   Extracted: {labels}")
        print()

def test_reward_function():
    """Test the complete reward function."""
    
    test_cases = [
        {
            "name": "Perfect Match",
            "response": """<answer>The cardiomegaly is at [0.34, 0.52, 0.8, 0.66].</answer>""",
            "ground_truth": {
                'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
                'labels': ['Cardiomegaly'],
                'has_no_finding': False
            }
        },
        {
            "name": "Label Variation",
            "response": """<answer>The heart enlargement is at [0.34, 0.52, 0.8, 0.66].</answer>""",
            "ground_truth": {
                'coordinates': [[0.34, 0.52, 0.8, 0.66]], 
                'labels': ['Cardiomegaly'],
                'has_no_finding': False
            }
        },
        {
            "name": "Multiple Labels",
            "response": """<answer>Pneumonia at [0.2, 0.3, 0.6, 0.7] and pleural effusion at [0.1, 0.8, 0.9, 0.95].</answer>""",
            "ground_truth": {
                'coordinates': [[0.2, 0.3, 0.6, 0.7], [0.1, 0.8, 0.9, 0.95]], 
                'labels': ['Pneumonia', 'Pleural effusion'],
                'has_no_finding': False
            }
        },
        {
            "name": "No Finding",
            "response": """<answer>No abnormalities detected.</answer>""",
            "ground_truth": {
                'coordinates': [], 
                'labels': ['No finding'],
                'has_no_finding': True
            }
        }
    ]
    
    print("🎯 Testing Complete Reward Function")
    print("=" * 60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['name']}")
        score = compute_score("vindr_grpo", test["response"], test["ground_truth"])
        print(f"   Score: {score:.4f}")
        
        # Show what was extracted
        import re
        match = re.search(r'<answer>(.*?)</answer>', test["response"], flags=re.IGNORECASE | re.DOTALL)
        answer = match.group(1).strip() if match else test["response"]
        
        try:
            labels = extract_medical_labels_spacy(answer)
        except:
            labels = extract_medical_labels_regex(answer)
        
        print(f"   Extracted labels: {labels}")
        print(f"   GT labels: {test['ground_truth']['labels']}")
        print()

if __name__ == "__main__":
    test_label_extraction()
    print()
    test_reward_function()
    
    print("✅ Clean spaCy approach testing completed!")
    print("\n🎯 Key Features:")
    print("   • Simple spaCy noun chunk extraction")
    print("   • Clean regex fallback")
    print("   • Straightforward label matching")
    print("   • Minimal, focused code")
    print("   • Easy to understand and maintain")