#!/usr/bin/env python3
"""
Test script for mAP-based reward function.

This script demonstrates the comprehensive mAP evaluation system with various
test cases showing different detection scenarios and their corresponding
mAP scores across multiple IoU thresholds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from custom_reward.map_reward import compute_reward, compute_map_coco_style
import numpy as np

def print_separator(title):
    """Print a formatted separator with title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_map_details(map_results):
    """Print detailed mAP results."""
    print(f"📊 Comprehensive mAP Analysis:")
    print(f"   🎯 mAP@[0.5:0.05:0.95]: {map_results['mAP']:.3f}")
    print(f"   🎯 AP@0.50: {map_results['AP@0.50']:.3f}")
    print(f"   🎯 AP@0.75: {map_results['AP@0.75']:.3f}")
    print(f"   📈 Precision@0.50: {map_results['precision@0.50']:.3f}")
    print(f"   📈 Recall@0.50: {map_results['recall@0.50']:.3f}")
    print(f"   📦 Predictions: {map_results['num_predictions']}")
    print(f"   🎯 Ground Truth: {map_results['num_ground_truth']}")
    
    print(f"\n   📋 AP per IoU threshold:")
    for i, (thresh, ap) in enumerate(zip(map_results['iou_thresholds'], map_results['ap_per_threshold'])):
        if i % 5 == 0:  # Print every 5th threshold to avoid clutter
            print(f"      AP@{thresh:.2f}: {ap:.3f}")

def test_map_reward():
    """Test the mAP reward function with various scenarios."""
    
    print_separator("mAP REWARD FUNCTION COMPREHENSIVE TEST")
    
    test_cases = [
        {
            "name": "🎯 Perfect Single Detection",
            "description": "Single perfect detection with exact overlap",
            "answer": "The chest X-ray shows a pneumothorax located at <100,150,200,250>.",
            "ground_truth": [[0.1, 0.15, 0.2, 0.25]],
            "expected": "Perfect mAP = 1.0 across all thresholds"
        },
        {
            "name": "📍 Good Single Detection",
            "description": "Single detection with good but not perfect overlap",
            "answer": "Abnormality detected at <80,130,180,230>.",
            "ground_truth": [[0.1, 0.15, 0.2, 0.25]],
            "expected": "High mAP at lower thresholds, decreasing at higher thresholds"
        },
        {
            "name": "🎯 Multiple Perfect Detections",
            "description": "Multiple detections, all perfectly matching ground truth",
            "answer": "Multiple findings: <50,100,150,200> and <300,400,450,550>",
            "ground_truth": [[0.05, 0.1, 0.15, 0.2], [0.3, 0.4, 0.45, 0.55]],
            "expected": "Perfect mAP = 1.0 for all detections"
        },
        {
            "name": "⚖️ Mixed Quality Detections",
            "description": "Mix of good and poor detections",
            "answer": "Findings at: <50,100,150,200> (good) and <600,700,800,900> (poor match)",
            "ground_truth": [[0.05, 0.1, 0.15, 0.2], [0.3, 0.4, 0.45, 0.55]],
            "expected": "Moderate mAP due to one good and one poor detection"
        },
        {
            "name": "❌ False Positive Only",
            "description": "Detection when no abnormality exists",
            "answer": "Suspicious opacity at <100,100,200,200>",
            "ground_truth": [],
            "expected": "mAP = 0.0 due to false positive"
        },
        {
            "name": "❌ False Negative (Missed Detection)",
            "description": "No detection when abnormality exists",
            "answer": "The chest X-ray appears normal with no abnormality.",
            "ground_truth": [[0.1, 0.1, 0.3, 0.3]],
            "expected": "mAP = 0.0 due to missed detection"
        },
        {
            "name": "✅ Correct No Finding",
            "description": "Correctly identifying no abnormality",
            "answer": "The chest X-ray appears normal with no abnormality detected.",
            "ground_truth": [],
            "expected": "Perfect score = 1.0 for correct negative"
        },
        {
            "name": "🔢 Excess Detections",
            "description": "More detections than ground truth (some false positives)",
            "answer": "Multiple findings: <50,100,150,200> and <300,400,450,550> and <600,700,800,900>",
            "ground_truth": [[0.05, 0.1, 0.15, 0.2], [0.3, 0.4, 0.45, 0.55]],
            "expected": "Reduced mAP due to false positive penalty"
        },
        {
            "name": "📏 Scale Sensitivity Test",
            "description": "Testing with very small detection",
            "answer": "Small nodule at <450,450,470,470>",
            "ground_truth": [[0.45, 0.45, 0.47, 0.47]],
            "expected": "Perfect match despite small size"
        },
        {
            "name": "🎯 Partial Overlap Test",
            "description": "Detection with significant but not complete overlap",
            "answer": "Large opacity at <50,50,250,250>",
            "ground_truth": [[0.1, 0.1, 0.2, 0.2]],
            "expected": "Good score at IoU=0.5, poor at IoU=0.75+"
        }
    ]
    
    print(f"Testing {len(test_cases)} different scenarios...\n")
    
    results_summary = []
    
    for i, test_case in enumerate(test_cases, 1):
        print_separator(f"TEST {i}: {test_case['name']}")
        
        print(f"📝 Description: {test_case['description']}")
        print(f"💬 Answer: {test_case['answer']}")
        print(f"🎯 Ground Truth: {test_case['ground_truth']}")
        print(f"🔮 Expected: {test_case['expected']}")
        
        # Compute reward
        reward_input = {"ground_truth": test_case["ground_truth"]}
        reward_score = compute_reward(test_case["answer"], reward_input)
        
        # Get detailed mAP results
        from custom_reward.map_reward import extract_bounding_boxes_from_answer
        predicted_boxes = extract_bounding_boxes_from_answer(test_case["answer"])
        map_results = compute_map_coco_style(predicted_boxes, test_case["ground_truth"])
        
        print(f"\n🏆 FINAL REWARD SCORE: {reward_score:.3f}")
        print_map_details(map_results)
        
        results_summary.append({
            "name": test_case["name"],
            "reward": reward_score,
            "map": map_results["mAP"],
            "ap50": map_results["AP@0.50"],
            "ap75": map_results["AP@0.75"]
        })
        
        print()
    
    # Summary comparison
    print_separator("RESULTS SUMMARY & COMPARISON")
    
    print("📊 Performance Ranking by mAP Score:")
    sorted_results = sorted(results_summary, key=lambda x: x["map"], reverse=True)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:2d}. {result['name']}")
        print(f"    Final Reward: {result['reward']:.3f} | "
              f"mAP: {result['map']:.3f} | "
              f"AP@0.50: {result['ap50']:.3f} | "
              f"AP@0.75: {result['ap75']:.3f}")
    
    print_separator("mAP vs BASIC IoU COMPARISON")
    
    print("🔍 Key Advantages of mAP Reward:")
    print("   ✅ Multi-threshold evaluation (robust assessment)")
    print("   ✅ Industry-standard object detection metric")
    print("   ✅ Precision-Recall curve analysis")
    print("   ✅ Handles varying detection quality gracefully")
    print("   ✅ COCO-style evaluation (mAP@[0.5:0.05:0.95])")
    print("   ✅ Detailed performance breakdown")
    print("   ✅ Separates precision vs recall performance")
    
    print("\n⚖️ Trade-offs:")
    print("   ⚠️  More computationally intensive than basic IoU")
    print("   ⚠️  Complex implementation with many components")
    print("   ⚠️  May be overkill for simple detection tasks")
    print("   ⚠️  Requires understanding of AP/mAP concepts")
    
    print("\n🎯 Best Use Cases:")
    print("   🏥 Medical imaging with varying abnormality sizes")
    print("   🔬 Research requiring detailed performance analysis")
    print("   📊 Benchmarking against standard object detection metrics")
    print("   🎓 Training models that need robust localization")
    print("   🏆 Competition or publication-quality evaluation")
    
    print_separator("IMPLEMENTATION DETAILS")
    
    print("🔧 Technical Features:")
    print("   📐 IoU Thresholds: [0.5, 0.55, 0.6, ..., 0.95] (10 thresholds)")
    print("   📊 Interpolation: 101-point (COCO standard)")
    print("   🎯 Matching: Hungarian-style optimal assignment")
    print("   📈 Metrics: AP, mAP, Precision, Recall per threshold")
    print("   🔄 Fallback: Graceful handling of edge cases")
    
    print("\n📏 Scoring Breakdown:")
    print("   🎯 Base Score: mAP@[0.5:0.05:0.95] (primary metric)")
    print("   ➕ Precision Bonus: 0.1 × Precision@0.50 (clinical relevance)")
    print("   ➕ Recall Bonus: 0.1 × Recall@0.50 (don't miss findings)")
    print("   ➖ False Positive Penalty: 0.05 × FP_rate (reduce noise)")
    print("   🔒 Range: [0.0, 1.0] (clamped)")

def main():
    """Main function to run all tests."""
    test_map_reward()
    
    print_separator("USAGE INSTRUCTIONS")
    
    print("🚀 To use mAP reward in your training:")
    print("   1. Use existing train.parquet and val.parquet (same data format)")
    print("   2. Run: sbatch jobs/single_node_map.sh")
    print("   3. Monitor training with comprehensive mAP logging")
    
    print("\n📁 Files created:")
    print("   ✅ custom_reward/map_reward.py - mAP reward implementation")
    print("   ✅ jobs/single_node_map.sh - SLURM script for mAP training")
    print("   ✅ test_map_reward.py - This comprehensive test script")
    
    print("\n🔗 Integration with existing pipeline:")
    print("   ✅ Same data preprocessing (llava_json_to_verl_iou_robust.py)")
    print("   ✅ Same VERL training framework")
    print("   ✅ Enhanced evaluation with detailed metrics")
    print("   ✅ Drop-in replacement for basic IoU reward")

if __name__ == "__main__":
    main()