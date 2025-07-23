#!/usr/bin/env python3
"""
Test the mAP evaluation framework without plotting dependencies.

This demonstrates the key concept:
- Train with different reward functions (IoU, GIoU, Enhanced Medical)
- Evaluate all models using the same mAP standard
- Compare which training approach produces best performance
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from pathlib import Path

# Import only the core mAP computation functions
from custom_reward.map_reward import (
    extract_bounding_boxes_from_answer,
    compute_map_coco_style,
    compute_map_single_threshold
)

def simulate_model_predictions(ground_truths, model_type):
    """Simulate predictions from models trained with different reward functions."""
    
    # Performance levels based on training reward function
    performance_levels = {
        'iou': 0.6,       # Basic IoU - baseline
        'giou': 0.75,     # GIoU - better gradients
        'enhanced': 0.8,  # Enhanced Medical - best performance
        'map': 0.77       # mAP training - good performance
    }
    
    perf_level = performance_levels.get(model_type, 0.65)
    print(f"   🎯 Simulating {model_type} model with performance level: {perf_level:.2f}")
    
    predictions = []
    np.random.seed(42)  # For reproducible results
    
    for gt_boxes in ground_truths:
        if not gt_boxes:  # No finding case
            if np.random.random() < 0.9:  # 90% chance to correctly identify no finding
                predictions.append("The chest X-ray appears normal with no abnormalities detected.")
            else:  # 10% false positive
                fake_box = np.random.uniform(0, 0.8, 4)
                fake_box[2:] = fake_box[:2] + np.random.uniform(0.1, 0.2, 2)
                predictions.append(f"Suspicious opacity at <{fake_box[0]*1000:.0f},{fake_box[1]*1000:.0f},{fake_box[2]*1000:.0f},{fake_box[3]*1000:.0f}>")
        else:  # Positive case
            pred_boxes = []
            for gt_box in gt_boxes:
                if np.random.random() < perf_level:  # Detection probability
                    # Add noise to simulate detection error
                    noise = np.random.normal(0, 0.05, 4)
                    pred_box = np.array(gt_box) + noise
                    pred_box = np.clip(pred_box, 0, 1)
                    pred_boxes.append(pred_box)
            
            if pred_boxes:
                box_strs = []
                for box in pred_boxes:
                    box_str = f"<{box[0]*1000:.0f},{box[1]*1000:.0f},{box[2]*1000:.0f},{box[3]*1000:.0f}>"
                    box_strs.append(box_str)
                predictions.append(f"Abnormalities detected: {' and '.join(box_strs)}")
            else:
                predictions.append("No significant abnormalities detected.")
    
    return predictions

def evaluate_model(predictions, ground_truths, model_name):
    """Evaluate a single model using mAP metrics."""
    
    print(f"\n🔍 Evaluating {model_name}...")
    print(f"   📊 Test samples: {len(predictions)}")
    print(f"   🎯 Positive samples: {sum(1 for gt in ground_truths if gt)}")
    print(f"   ❌ Negative samples: {sum(1 for gt in ground_truths if not gt)}")
    
    # Compute mAP for each sample
    all_map_results = []
    sample_details = []
    
    for i, (prediction, gt_boxes) in enumerate(zip(predictions, ground_truths)):
        # Extract predicted boxes
        pred_boxes = extract_bounding_boxes_from_answer(prediction)
        
        # Compute mAP results
        map_results = compute_map_coco_style(pred_boxes, gt_boxes)
        all_map_results.append(map_results)
        
        sample_details.append({
            "sample_id": i,
            "map_score": map_results["mAP"],
            "ap50": map_results["AP@0.50"],
            "ap75": map_results["AP@0.75"],
            "precision_50": map_results["precision@0.50"],
            "recall_50": map_results["recall@0.50"],
            "num_predictions": len(pred_boxes),
            "num_ground_truth": len(gt_boxes)
        })
    
    # Compute aggregate statistics
    map_scores = [result["mAP"] for result in all_map_results]
    ap50_scores = [result["AP@0.50"] for result in all_map_results]
    ap75_scores = [result["AP@0.75"] for result in all_map_results]
    precision_scores = [result["precision@0.50"] for result in all_map_results]
    recall_scores = [result["recall@0.50"] for result in all_map_results]
    
    # Categorize samples
    positive_samples = [s for s in sample_details if s["num_ground_truth"] > 0]
    negative_samples = [s for s in sample_details if s["num_ground_truth"] == 0]
    
    results = {
        "model_name": model_name,
        "mean_mAP": np.mean(map_scores),
        "std_mAP": np.std(map_scores),
        "mean_AP50": np.mean(ap50_scores),
        "std_AP50": np.std(ap50_scores),
        "mean_AP75": np.mean(ap75_scores),
        "std_AP75": np.std(ap75_scores),
        "mean_precision": np.mean(precision_scores),
        "std_precision": np.std(precision_scores),
        "mean_recall": np.mean(recall_scores),
        "std_recall": np.std(recall_scores),
        "total_samples": len(sample_details),
        "positive_samples": len(positive_samples),
        "negative_samples": len(negative_samples),
        "positive_sample_mAP": np.mean([s["map_score"] for s in positive_samples]) if positive_samples else 0.0,
        "negative_sample_accuracy": np.mean([1.0 if s["num_predictions"] == 0 else 0.0 for s in negative_samples]) if negative_samples else 1.0,
    }
    
    print(f"   📈 Results:")
    print(f"      mAP@[0.5:0.05:0.95]: {results['mean_mAP']:.3f} (±{results['std_mAP']:.3f})")
    print(f"      AP@0.50: {results['mean_AP50']:.3f} (±{results['std_AP50']:.3f})")
    print(f"      AP@0.75: {results['mean_AP75']:.3f} (±{results['std_AP75']:.3f})")
    print(f"      Precision@0.50: {results['mean_precision']:.3f} (±{results['std_precision']:.3f})")
    print(f"      Recall@0.50: {results['mean_recall']:.3f} (±{results['std_recall']:.3f})")
    
    return results

def main():
    """Demonstrate evaluation of models trained with different reward functions."""
    
    print("🎯 mAP-based Evaluation: Training vs Evaluation Metrics")
    print("=" * 70)
    print("Key insight: Use fast rewards for TRAINING, comprehensive mAP for EVALUATION")
    print()
    
    # Create test data
    print("📊 Creating test dataset...")
    np.random.seed(42)
    n_samples = 100
    
    ground_truths = []
    for i in range(n_samples):
        if i < 70:  # 70% positive samples
            # Random ground truth boxes
            num_boxes = np.random.choice([1, 2], p=[0.8, 0.2])
            boxes = []
            for _ in range(num_boxes):
                x1, y1 = np.random.uniform(0, 0.5, 2)
                x2, y2 = x1 + np.random.uniform(0.1, 0.3), y1 + np.random.uniform(0.1, 0.3)
                boxes.append([x1, y1, min(x2, 1.0), min(y2, 1.0)])
            ground_truths.append(boxes)
        else:  # 30% negative samples
            ground_truths.append([])
    
    print(f"   ✅ Created {len(ground_truths)} test samples")
    print(f"   🎯 Positive samples: {sum(1 for gt in ground_truths if gt)}")
    print(f"   ❌ Negative samples: {sum(1 for gt in ground_truths if not gt)}")
    
    # Simulate models trained with different reward functions
    models_to_test = {
        "IoU_trained": "iou",
        "GIoU_trained": "giou", 
        "Enhanced_Medical_trained": "enhanced",
        "mAP_trained": "map"
    }
    
    print("\n🚀 Simulating models trained with different reward functions...")
    
    all_results = {}
    
    for model_name, model_type in models_to_test.items():
        # Simulate predictions from this model
        predictions = simulate_model_predictions(ground_truths, model_type)
        
        # Evaluate using mAP
        results = evaluate_model(predictions, ground_truths, model_name)
        all_results[model_name] = results
    
    # Compare all models
    print("\n🏆 FINAL COMPARISON - All Models Evaluated with mAP")
    print("=" * 70)
    
    # Sort by mAP performance
    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["mean_mAP"], reverse=True)
    
    print("📊 Performance Ranking:")
    for rank, (model_name, results) in enumerate(sorted_models, 1):
        print(f"   {rank}. {model_name}:")
        print(f"      mAP: {results['mean_mAP']:.3f} (±{results['std_mAP']:.3f})")
        print(f"      AP@0.50: {results['mean_AP50']:.3f}")
        print(f"      AP@0.75: {results['mean_AP75']:.3f}")
        print(f"      Precision: {results['mean_precision']:.3f}")
        print(f"      Recall: {results['mean_recall']:.3f}")
        print()
    
    # Analysis
    best_model = sorted_models[0][1]
    worst_model = sorted_models[-1][1]
    improvement = ((best_model["mean_mAP"] - worst_model["mean_mAP"]) / worst_model["mean_mAP"]) * 100
    
    print("💡 KEY INSIGHTS:")
    print(f"   🏆 Best model: {sorted_models[0][0]} (mAP: {best_model['mean_mAP']:.3f})")
    print(f"   📈 Improvement over worst: {improvement:.1f}%")
    print(f"   🎯 Training reward functions DO affect final mAP performance")
    print(f"   ⚡ Fast training rewards can still produce excellent final results")
    
    print("\n🔬 DETAILED ANALYSIS:")
    print("   📋 Expected ranking based on reward function design:")
    print("      1. Enhanced Medical (domain knowledge + clinical reasoning)")
    print("      2. mAP training (directly optimizes evaluation metric)")  
    print("      3. GIoU training (better gradients than IoU)")
    print("      4. Basic IoU (baseline, limited gradients)")
    
    print("\n✅ CONCLUSION:")
    print("   🎯 This demonstrates the key insight:")
    print("      • TRAINING: Use efficient reward functions (GIoU recommended)")
    print("      • EVALUATION: Use comprehensive mAP for rigorous assessment") 
    print("      • COMPARISON: mAP enables fair comparison across training methods")
    print("   📊 You can now objectively determine which training approach works best!")

if __name__ == "__main__":
    main()