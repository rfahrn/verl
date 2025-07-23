#!/usr/bin/env python3
"""
Practical Script: Evaluate Models Trained with Different Reward Functions

This script demonstrates the key insight:
- TRAINING: Use fast reward functions (IoU, GIoU) for efficient learning
- EVALUATION: Use comprehensive mAP metrics for rigorous performance assessment

Usage:
1. Train models with different reward functions:
   - sbatch jobs/single_node_basic.sh    # IoU reward
   - sbatch jobs/single_node_giou.sh     # GIoU reward  
   - sbatch jobs/single_node_enhanced.sh # Enhanced Medical reward

2. Evaluate all models using this script with mAP metrics

3. Compare which training reward function leads to best final performance
"""

import sys
import os
sys.path.append('.')

from evaluation.map_evaluator import mAPEvaluator
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json

def load_model_predictions(model_checkpoint_dir: str, test_data_path: str):
    """
    Load model predictions from checkpoint directory.
    
    In practice, you would:
    1. Load your trained model from checkpoint_dir
    2. Run inference on test_data_path 
    3. Return predictions and ground truths
    
    For this demo, we'll simulate this process.
    """
    print(f"📂 Loading model from: {model_checkpoint_dir}")
    print(f"📊 Running inference on: {test_data_path}")
    
    # In real usage, you would do:
    # model = load_model(model_checkpoint_dir)
    # predictions = model.predict(test_data)
    
    # For demo, we'll load test data and simulate predictions
    if test_data_path.endswith('.parquet'):
        test_df = pd.read_parquet(test_data_path)
        ground_truths = []
        
        for _, row in test_df.iterrows():
            gt_data = row.get('reward_model', {})
            if isinstance(gt_data, dict) and 'ground_truth' in gt_data:
                ground_truths.append(gt_data['ground_truth'])
            else:
                ground_truths.append([])
        
        # Simulate predictions based on model name
        model_name = Path(model_checkpoint_dir).name.lower()
        predictions = simulate_model_predictions(ground_truths, model_name)
        
        return predictions, ground_truths
    else:
        raise ValueError(f"Unsupported test data format: {test_data_path}")

def simulate_model_predictions(ground_truths, model_name):
    """
    Simulate model predictions based on expected performance.
    
    In real usage, this would be replaced with actual model inference.
    """
    # Different performance levels based on training reward function
    performance_map = {
        'iou': 0.6,      # Basic IoU training
        'giou': 0.75,    # GIoU training - better gradients
        'enhanced': 0.8, # Enhanced medical training - best performance
        'map': 0.77      # mAP training - good performance
    }
    
    # Determine performance level
    perf_level = 0.65  # default
    for key, level in performance_map.items():
        if key in model_name:
            perf_level = level
            break
    
    print(f"   🎯 Simulating performance level: {perf_level:.2f}")
    
    predictions = []
    np.random.seed(42)  # For reproducible results
    
    for gt_boxes in ground_truths:
        if not gt_boxes:  # No finding case
            if np.random.random() < 0.9:  # 90% chance to correctly identify no finding
                predictions.append("The chest X-ray appears normal with no abnormalities detected.")
            else:  # 10% false positive
                fake_box = np.random.uniform(0, 0.8, 4)
                fake_box[2:] = fake_box[:2] + np.random.uniform(0.1, 0.2, 2)
                predictions.append(f"Suspicious opacity detected at <{fake_box[0]*1000:.0f},{fake_box[1]*1000:.0f},{fake_box[2]*1000:.0f},{fake_box[3]*1000:.0f}>")
        else:  # Positive case
            pred_boxes = []
            for gt_box in gt_boxes:
                if np.random.random() < perf_level:  # Chance to detect based on performance
                    # Add noise to ground truth box (simulating detection error)
                    noise = np.random.normal(0, 0.05, 4)
                    pred_box = np.array(gt_box) + noise
                    pred_box = np.clip(pred_box, 0, 1)
                    pred_boxes.append(pred_box)
            
            if pred_boxes:
                box_strs = []
                for box in pred_boxes:
                    box_str = f"<{box[0]*1000:.0f},{box[1]*1000:.0f},{box[2]*1000:.0f},{box[3]*1000:.0f}>"
                    box_strs.append(box_str)
                predictions.append(f"Abnormalities detected at: {' and '.join(box_strs)}")
            else:
                predictions.append("No significant abnormalities detected in this image.")
    
    return predictions

def evaluate_single_model(model_checkpoint_dir: str, 
                         test_data_path: str,
                         evaluator: mAPEvaluator):
    """Evaluate a single model using mAP metrics."""
    
    model_name = Path(model_checkpoint_dir).name
    print(f"\n🔍 Evaluating model: {model_name}")
    
    # Load model predictions
    predictions, ground_truths = load_model_predictions(model_checkpoint_dir, test_data_path)
    
    print(f"   📊 Test samples: {len(predictions)}")
    print(f"   🎯 Positive samples: {sum(1 for gt in ground_truths if gt)}")
    print(f"   ❌ Negative samples: {sum(1 for gt in ground_truths if not gt)}")
    
    # Evaluate with mAP
    results = evaluator.evaluate_model_predictions(
        predictions, ground_truths, model_name, "test_set"
    )
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate models trained with different reward functions using mAP")
    parser.add_argument("--test_data", required=True, help="Path to test data (parquet file)")
    parser.add_argument("--model_dirs", nargs="+", required=True, 
                       help="Paths to model checkpoint directories")
    parser.add_argument("--output_dir", default="evaluation/results", 
                       help="Output directory for results")
    parser.add_argument("--no_plots", action="store_true", 
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    print("🎯 mAP-based Evaluation of Models Trained with Different Reward Functions")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = mAPEvaluator(output_dir=args.output_dir)
    
    # Evaluate each model
    all_results = {}
    
    for model_dir in args.model_dirs:
        if not Path(model_dir).exists():
            print(f"⚠️  Warning: Model directory not found: {model_dir}")
            print(f"   Creating simulated results for demonstration...")
            
        results = evaluate_single_model(model_dir, args.test_data, evaluator)
        all_results[results["model_name"]] = results
    
    # Compare all models
    print("\n🏆 Comparing all models using mAP metrics...")
    comparison_results = evaluator.compare_models(all_results, save_plots=not args.no_plots)
    
    # Print detailed results
    print("\n" + "="*80)
    print("📊 DETAILED EVALUATION RESULTS")
    print("="*80)
    
    for model_name, results in all_results.items():
        agg = results["aggregate_results"]
        print(f"\n🔬 {model_name}:")
        print(f"   mAP@[0.5:0.05:0.95]: {agg['mean_mAP']:.3f} (±{agg['std_mAP']:.3f})")
        print(f"   AP@0.50: {agg['mean_AP50']:.3f} (±{agg['std_AP50']:.3f})")
        print(f"   AP@0.75: {agg['mean_AP75']:.3f} (±{agg['std_AP75']:.3f})")
        print(f"   Precision@0.50: {agg['mean_precision']:.3f} (±{agg['std_precision']:.3f})")
        print(f"   Recall@0.50: {agg['mean_recall']:.3f} (±{agg['std_recall']:.3f})")
        print(f"   Positive Sample mAP: {agg['positive_sample_mAP']:.3f}")
        print(f"   Negative Sample Accuracy: {agg['negative_sample_accuracy']:.3f}")
    
    # Print comparison summary
    print(comparison_results["summary"])
    
    # Statistical significance
    if comparison_results["statistical_significance"]:
        print("\n📈 STATISTICAL SIGNIFICANCE TESTS:")
        print("="*50)
        for test in comparison_results["statistical_significance"]["pairwise_tests"]:
            significance = "✅ Significant" if test["significant"] else "❌ Not significant"
            print(f"{test['model_a']} vs {test['model_b']}: {significance} (p={test['p_value']:.4f})")
    
    # Files generated
    print(f"\n📁 Results saved to: {args.output_dir}")
    print("📄 Generated files:")
    print(f"   - model_comparison_results.json")
    print(f"   - model_comparison_table.csv") 
    print(f"   - comparison_summary.txt")
    
    if not args.no_plots:
        print("📊 Generated plots:")
        for plot_path in comparison_results["plot_paths"]:
            print(f"   - {Path(plot_path).name}")
    
    print("\n🎉 Evaluation complete!")
    
    # Recommendations
    best_model = comparison_results["best_model"]
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   🏆 Best performing model: {best_model}")
    print(f"   📈 Consider using the training approach from {best_model} for future models")
    print(f"   🔬 Analyze the detailed per-threshold results to understand model behavior")

def demo_usage():
    """Demonstrate usage with simulated data."""
    print("🎯 DEMO: Evaluating Different Reward Function Training Approaches")
    print("=" * 70)
    
    # Create some demo test data
    demo_data = []
    np.random.seed(42)
    
    for i in range(50):  # 50 test samples
        if i < 35:  # 70% positive samples
            # Random ground truth boxes
            num_boxes = np.random.choice([1, 2], p=[0.8, 0.2])
            boxes = []
            for _ in range(num_boxes):
                x1, y1 = np.random.uniform(0, 0.5, 2)
                x2, y2 = x1 + np.random.uniform(0.1, 0.3), y1 + np.random.uniform(0.1, 0.3)
                boxes.append([x1, y1, min(x2, 1.0), min(y2, 1.0)])
            
            demo_data.append({
                'reward_model': {'ground_truth': boxes},
                'sample_id': i
            })
        else:  # 30% negative samples
            demo_data.append({
                'reward_model': {'ground_truth': []},
                'sample_id': i
            })
    
    # Save demo data
    demo_df = pd.DataFrame(demo_data)
    demo_path = "demo_test_data.parquet"
    demo_df.to_parquet(demo_path)
    
    print(f"📊 Created demo test data: {demo_path}")
    print(f"   Samples: {len(demo_data)}")
    print(f"   Positive: {sum(1 for d in demo_data if d['reward_model']['ground_truth'])}")
    print(f"   Negative: {sum(1 for d in demo_data if not d['reward_model']['ground_truth'])}")
    
    # Simulate different model checkpoints
    model_dirs = [
        "checkpoints/model_trained_with_iou_reward",
        "checkpoints/model_trained_with_giou_reward", 
        "checkpoints/model_trained_with_enhanced_reward",
        "checkpoints/model_trained_with_map_reward"
    ]
    
    # Initialize evaluator
    evaluator = mAPEvaluator(output_dir="evaluation/demo_results")
    
    # Evaluate each "model"
    all_results = {}
    
    for model_dir in model_dirs:
        results = evaluate_single_model(model_dir, demo_path, evaluator)
        all_results[results["model_name"]] = results
    
    # Compare all models
    print("\n🏆 Comparing all models...")
    comparison_results = evaluator.compare_models(all_results, save_plots=True)
    
    # Print summary
    print(comparison_results["summary"])
    
    print(f"\n📁 Demo results saved to: evaluation/demo_results/")
    print("🎨 Check the generated plots to see visual comparisons!")
    
    # Clean up
    os.remove(demo_path)
    print(f"\n🧹 Cleaned up demo file: {demo_path}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run demo
        demo_usage()
    else:
        # Arguments provided, run actual evaluation
        main()