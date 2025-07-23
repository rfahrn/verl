#!/usr/bin/env python3
"""
mAP-based Model Evaluation Framework

This module provides comprehensive evaluation of models trained with different reward functions
(IoU, GIoU, Enhanced Medical, etc.) using industry-standard mAP metrics.

The key insight: 
- TRAINING: Use fast reward functions (IoU, GIoU) for efficient learning
- EVALUATION: Use comprehensive mAP metrics for rigorous performance assessment

This allows you to:
1. Train multiple models with different reward functions
2. Evaluate all models using the same mAP standard
3. Compare which training reward leads to best final performance
4. Generate publication-quality results and plots
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Any, Union
from pathlib import Path
import logging
from datetime import datetime
import pickle

# Import the mAP computation functions from our reward function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom_reward.map_reward import (
    extract_bounding_boxes_from_answer,
    compute_map_coco_style,
    compute_map_single_threshold
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class mAPEvaluator:
    """
    Comprehensive mAP-based evaluation framework for medical grounding models.
    
    This evaluator can assess models trained with any reward function using
    standardized mAP metrics, enabling fair comparison across different training approaches.
    """
    
    def __init__(self, 
                 iou_thresholds: Optional[List[float]] = None,
                 save_detailed_results: bool = True,
                 output_dir: str = "evaluation/results"):
        """
        Initialize the mAP evaluator.
        
        Args:
            iou_thresholds: IoU thresholds for evaluation (default: COCO-style)
            save_detailed_results: Whether to save detailed per-sample results
            output_dir: Directory to save evaluation results
        """
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05).tolist()
        self.save_detailed_results = save_detailed_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for results
        self.evaluation_results = {}
        self.detailed_results = []
        
    def evaluate_model_predictions(self, 
                                 predictions: List[str],
                                 ground_truths: List[List[List[float]]],
                                 model_name: str = "model",
                                 dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate a model's predictions using comprehensive mAP metrics.
        
        Args:
            predictions: List of model text outputs
            ground_truths: List of ground truth bounding boxes for each sample
            model_name: Name of the model being evaluated
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"Evaluating {model_name} on {dataset_name} dataset...")
        logger.info(f"Samples: {len(predictions)}, IoU thresholds: {len(self.iou_thresholds)}")
        
        # Storage for all metrics
        all_map_results = []
        per_threshold_aps = {f"AP@{thresh:.2f}": [] for thresh in self.iou_thresholds}
        sample_details = []
        
        # Process each sample
        for i, (prediction, gt_boxes) in enumerate(zip(predictions, ground_truths)):
            # Extract predicted boxes
            pred_boxes = extract_bounding_boxes_from_answer(prediction)
            
            # Compute comprehensive mAP results
            map_results = compute_map_coco_style(pred_boxes, gt_boxes)
            all_map_results.append(map_results)
            
            # Store per-threshold results
            for thresh in self.iou_thresholds:
                thresh_key = f"AP@{thresh:.2f}"
                if thresh_key in map_results:
                    per_threshold_aps[thresh_key].append(map_results[thresh_key])
                else:
                    # Compute for this specific threshold if not already computed
                    thresh_result = compute_map_single_threshold(pred_boxes, gt_boxes, thresh)
                    per_threshold_aps[thresh_key].append(thresh_result["ap"])
            
            # Store detailed sample information
            if self.save_detailed_results:
                sample_details.append({
                    "sample_id": i,
                    "prediction": prediction,
                    "ground_truth": gt_boxes,
                    "predicted_boxes": pred_boxes,
                    "map_score": map_results["mAP"],
                    "ap50": map_results["AP@0.50"],
                    "ap75": map_results["AP@0.75"],
                    "precision_50": map_results["precision@0.50"],
                    "recall_50": map_results["recall@0.50"],
                    "num_predictions": len(pred_boxes),
                    "num_ground_truth": len(gt_boxes)
                })
        
        # Compute aggregate statistics
        aggregate_results = self._compute_aggregate_statistics(
            all_map_results, per_threshold_aps, sample_details
        )
        
        # Store results
        evaluation_key = f"{model_name}_{dataset_name}"
        self.evaluation_results[evaluation_key] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "aggregate_results": aggregate_results,
            "per_sample_results": sample_details if self.save_detailed_results else None,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Evaluation complete. mAP: {aggregate_results['mean_mAP']:.3f}")
        
        return self.evaluation_results[evaluation_key]
    
    def _compute_aggregate_statistics(self, 
                                    all_map_results: List[Dict],
                                    per_threshold_aps: Dict[str, List[float]],
                                    sample_details: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics across all samples."""
        
        # Main mAP metrics
        map_scores = [result["mAP"] for result in all_map_results]
        ap50_scores = [result["AP@0.50"] for result in all_map_results]
        ap75_scores = [result["AP@0.75"] for result in all_map_results]
        
        # Precision and recall at IoU=0.5
        precision_scores = [result["precision@0.50"] for result in all_map_results]
        recall_scores = [result["recall@0.50"] for result in all_map_results]
        
        # Per-threshold statistics
        per_threshold_stats = {}
        for thresh_key, ap_list in per_threshold_aps.items():
            per_threshold_stats[thresh_key] = {
                "mean": np.mean(ap_list),
                "std": np.std(ap_list),
                "median": np.median(ap_list),
                "min": np.min(ap_list),
                "max": np.max(ap_list)
            }
        
        # Detection statistics
        total_predictions = sum(len(result["predicted_boxes"]) for result in sample_details)
        total_ground_truth = sum(result["num_ground_truth"] for result in sample_details)
        
        # Categorize samples by ground truth presence
        positive_samples = [s for s in sample_details if s["num_ground_truth"] > 0]
        negative_samples = [s for s in sample_details if s["num_ground_truth"] == 0]
        
        aggregate_results = {
            # Main metrics
            "mean_mAP": np.mean(map_scores),
            "std_mAP": np.std(map_scores),
            "mean_AP50": np.mean(ap50_scores),
            "std_AP50": np.std(ap50_scores),
            "mean_AP75": np.mean(ap75_scores),
            "std_AP75": np.std(ap75_scores),
            
            # Precision and Recall
            "mean_precision": np.mean(precision_scores),
            "std_precision": np.std(precision_scores),
            "mean_recall": np.mean(recall_scores),
            "std_recall": np.std(recall_scores),
            
            # Per-threshold breakdown
            "per_threshold_stats": per_threshold_stats,
            
            # Dataset statistics
            "total_samples": len(sample_details),
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples),
            "total_predictions": total_predictions,
            "total_ground_truth": total_ground_truth,
            
            # Performance on positive samples (samples with ground truth)
            "positive_sample_mAP": np.mean([s["map_score"] for s in positive_samples]) if positive_samples else 0.0,
            "positive_sample_precision": np.mean([s["precision_50"] for s in positive_samples]) if positive_samples else 0.0,
            "positive_sample_recall": np.mean([s["recall_50"] for s in positive_samples]) if positive_samples else 0.0,
            
            # Performance on negative samples (no ground truth - should predict nothing)
            "negative_sample_accuracy": np.mean([1.0 if s["num_predictions"] == 0 else 0.0 for s in negative_samples]) if negative_samples else 1.0,
        }
        
        return aggregate_results
    
    def compare_models(self, 
                      model_results: Dict[str, Dict],
                      save_plots: bool = True) -> Dict[str, Any]:
        """
        Compare multiple models trained with different reward functions.
        
        Args:
            model_results: Dictionary mapping model names to their evaluation results
            save_plots: Whether to save comparison plots
            
        Returns:
            Comprehensive comparison results
        """
        logger.info(f"Comparing {len(model_results)} models...")
        
        # Extract key metrics for comparison
        comparison_data = []
        for model_name, results in model_results.items():
            agg = results["aggregate_results"]
            comparison_data.append({
                "model": model_name,
                "mAP": agg["mean_mAP"],
                "mAP_std": agg["std_mAP"],
                "AP50": agg["mean_AP50"],
                "AP50_std": agg["std_AP50"],
                "AP75": agg["mean_AP75"],
                "AP75_std": agg["std_AP75"],
                "Precision": agg["mean_precision"],
                "Precision_std": agg["std_precision"],
                "Recall": agg["mean_recall"],
                "Recall_std": agg["std_recall"],
                "Positive_mAP": agg["positive_sample_mAP"],
                "Negative_Accuracy": agg["negative_sample_accuracy"],
                "Total_Samples": agg["total_samples"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models by mAP
        comparison_df = comparison_df.sort_values("mAP", ascending=False)
        comparison_df["Rank"] = range(1, len(comparison_df) + 1)
        
        # Statistical significance testing (if multiple models)
        significance_results = None
        if len(model_results) > 1:
            significance_results = self._compute_statistical_significance(model_results)
        
        # Generate plots if requested
        plot_paths = []
        if save_plots:
            plot_paths = self._generate_comparison_plots(comparison_df, model_results)
        
        comparison_results = {
            "comparison_table": comparison_df,
            "best_model": comparison_df.iloc[0]["model"],
            "performance_ranking": comparison_df[["Rank", "model", "mAP", "AP50", "AP75"]].to_dict("records"),
            "statistical_significance": significance_results,
            "plot_paths": plot_paths,
            "summary": self._generate_comparison_summary(comparison_df)
        }
        
        # Save results
        self._save_comparison_results(comparison_results)
        
        return comparison_results
    
    def _compute_statistical_significance(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute statistical significance between models using t-tests."""
        from scipy import stats
        
        # Extract per-sample mAP scores for each model
        model_scores = {}
        for model_name, results in model_results.items():
            if results["per_sample_results"]:
                scores = [sample["map_score"] for sample in results["per_sample_results"]]
                model_scores[model_name] = scores
        
        # Pairwise t-tests
        significance_results = {"pairwise_tests": []}
        model_names = list(model_scores.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_a, model_b = model_names[i], model_names[j]
                scores_a, scores_b = model_scores[model_a], model_scores[model_b]
                
                # Paired t-test (assuming same test set)
                if len(scores_a) == len(scores_b):
                    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
                    test_type = "paired"
                else:
                    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                    test_type = "independent"
                
                significance_results["pairwise_tests"].append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "test_type": test_type,
                    "mean_diff": np.mean(scores_a) - np.mean(scores_b)
                })
        
        return significance_results
    
    def _generate_comparison_plots(self, 
                                 comparison_df: pd.DataFrame,
                                 model_results: Dict[str, Dict]) -> List[str]:
        """Generate comprehensive comparison plots."""
        plot_paths = []
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Main metrics comparison (bar plot)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')
        
        metrics = ["mAP", "AP50", "AP75", "Precision"]
        metric_stds = ["mAP_std", "AP50_std", "AP75_std", "Precision_std"]
        
        for idx, (metric, std_metric) in enumerate(zip(metrics, metric_stds)):
            ax = axes[idx // 2, idx % 2]
            bars = ax.bar(comparison_df["model"], comparison_df[metric], 
                         yerr=comparison_df[std_metric], capsize=5, alpha=0.8)
            ax.set_title(f"{metric} Comparison", fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, comparison_df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = self.output_dir / "model_comparison_metrics.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 2. Per-threshold AP comparison (line plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        iou_thresholds = self.iou_thresholds
        for model_name, results in model_results.items():
            per_thresh_stats = results["aggregate_results"]["per_threshold_stats"]
            ap_means = [per_thresh_stats[f"AP@{thresh:.2f}"]["mean"] for thresh in iou_thresholds]
            ap_stds = [per_thresh_stats[f"AP@{thresh:.2f}"]["std"] for thresh in iou_thresholds]
            
            ax.plot(iou_thresholds, ap_means, marker='o', linewidth=2, 
                   markersize=6, label=model_name)
            ax.fill_between(iou_thresholds, 
                           np.array(ap_means) - np.array(ap_stds),
                           np.array(ap_means) + np.array(ap_stds),
                           alpha=0.2)
        
        ax.set_xlabel("IoU Threshold", fontweight='bold')
        ax.set_ylabel("Average Precision (AP)", fontweight='bold')
        ax.set_title("AP vs IoU Threshold", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "ap_vs_iou_threshold.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        # 3. Precision-Recall scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name in comparison_df["model"]:
            precision = comparison_df[comparison_df["model"] == model_name]["Precision"].iloc[0]
            recall = comparison_df[comparison_df["model"] == model_name]["Recall"].iloc[0]
            ax.scatter(recall, precision, s=100, label=model_name, alpha=0.8)
            ax.annotate(model_name, (recall, precision), 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel("Recall", fontweight='bold')
        ax.set_ylabel("Precision", fontweight='bold')
        ax.set_title("Precision vs Recall", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add diagonal line for F1 score reference
        x = np.linspace(0, 1, 100)
        for f1 in [0.5, 0.7, 0.9]:
            y = f1 * x / (2 * x - f1)
            y = np.where(y > 0, y, np.nan)
            ax.plot(x, y, '--', alpha=0.5, label=f'F1={f1}')
        
        plt.tight_layout()
        plot_path = self.output_dir / "precision_recall_scatter.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
        
        logger.info(f"Generated {len(plot_paths)} comparison plots")
        return plot_paths
    
    def _generate_comparison_summary(self, comparison_df: pd.DataFrame) -> str:
        """Generate a text summary of the comparison results."""
        best_model = comparison_df.iloc[0]
        worst_model = comparison_df.iloc[-1]
        
        summary = f"""
=== MODEL PERFORMANCE COMPARISON SUMMARY ===

🏆 BEST PERFORMING MODEL: {best_model['model']}
   mAP: {best_model['mAP']:.3f} (±{best_model['mAP_std']:.3f})
   AP@0.50: {best_model['AP50']:.3f} (±{best_model['AP50_std']:.3f})
   AP@0.75: {best_model['AP75']:.3f} (±{best_model['AP75_std']:.3f})
   Precision: {best_model['Precision']:.3f} (±{best_model['Precision_std']:.3f})
   Recall: {best_model['Recall']:.3f} (±{best_model['Recall_std']:.3f})

📊 PERFORMANCE RANKING:
"""
        
        for _, row in comparison_df.iterrows():
            summary += f"   {row['Rank']}. {row['model']}: mAP={row['mAP']:.3f}\n"
        
        improvement = ((best_model['mAP'] - worst_model['mAP']) / worst_model['mAP']) * 100
        summary += f"\n💡 IMPROVEMENT: Best model is {improvement:.1f}% better than worst model\n"
        
        return summary
    
    def _save_comparison_results(self, comparison_results: Dict[str, Any]) -> None:
        """Save comparison results to disk."""
        # Save as JSON (excluding DataFrame which isn't JSON serializable)
        json_results = {k: v for k, v in comparison_results.items() if k != "comparison_table"}
        json_results["comparison_data"] = comparison_results["comparison_table"].to_dict("records")
        
        json_path = self.output_dir / "model_comparison_results.json"
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save DataFrame as CSV
        csv_path = self.output_dir / "model_comparison_table.csv"
        comparison_results["comparison_table"].to_csv(csv_path, index=False)
        
        # Save summary as text
        summary_path = self.output_dir / "comparison_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(comparison_results["summary"])
        
        logger.info(f"Comparison results saved to {self.output_dir}")
    
    def evaluate_from_parquet(self, 
                            model_outputs_path: str,
                            model_name: str = "model",
                            answer_column: str = "answer",
                            ground_truth_column: str = "reward_model") -> Dict[str, Any]:
        """
        Evaluate model from parquet file containing predictions.
        
        Args:
            model_outputs_path: Path to parquet file with model outputs
            model_name: Name of the model
            answer_column: Column name containing model answers
            ground_truth_column: Column name containing ground truth data
            
        Returns:
            Evaluation results
        """
        import pandas as pd
        
        logger.info(f"Loading model outputs from {model_outputs_path}")
        df = pd.read_parquet(model_outputs_path)
        
        # Extract predictions and ground truths
        predictions = df[answer_column].tolist()
        
        # Handle ground truth format (assuming it's stored as dict with 'ground_truth' key)
        ground_truths = []
        for _, row in df.iterrows():
            gt_data = row[ground_truth_column]
            if isinstance(gt_data, dict) and 'ground_truth' in gt_data:
                ground_truths.append(gt_data['ground_truth'])
            elif isinstance(gt_data, list):
                ground_truths.append(gt_data)
            else:
                ground_truths.append([])  # Empty for no findings
        
        return self.evaluate_model_predictions(
            predictions, ground_truths, model_name, "parquet_data"
        )

def main():
    """Example usage of the mAP evaluation framework."""
    
    print("🎯 mAP-based Model Evaluation Framework")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = mAPEvaluator(output_dir="evaluation/results")
    
    # Example: Simulate evaluation of models trained with different reward functions
    print("\n📊 Simulating evaluation of models trained with different rewards...")
    
    # Simulate some test data
    np.random.seed(42)
    n_samples = 100
    
    # Ground truth: mix of positive and negative samples
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
    
    # Simulate predictions from different models
    models_to_evaluate = {
        "IoU_trained": {
            "description": "Model trained with basic IoU reward",
            "performance_level": 0.6  # Baseline performance
        },
        "GIoU_trained": {
            "description": "Model trained with GIoU reward", 
            "performance_level": 0.75  # Better performance
        },
        "Enhanced_trained": {
            "description": "Model trained with Enhanced Medical reward",
            "performance_level": 0.8  # Best performance
        }
    }
    
    evaluation_results = {}
    
    for model_name, model_info in models_to_evaluate.items():
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Simulate model predictions based on performance level
        predictions = []
        perf_level = model_info["performance_level"]
        
        for gt_boxes in ground_truths:
            if not gt_boxes:  # No finding case
                if np.random.random() < 0.9:  # 90% chance to correctly identify no finding
                    predictions.append("The image appears normal with no abnormalities.")
                else:  # 10% false positive
                    fake_box = np.random.uniform(0, 0.8, 4)
                    fake_box[2:] = fake_box[:2] + np.random.uniform(0.1, 0.2, 2)
                    predictions.append(f"Abnormality detected at <{fake_box[0]*1000:.0f},{fake_box[1]*1000:.0f},{fake_box[2]*1000:.0f},{fake_box[3]*1000:.0f}>")
            else:  # Positive case
                pred_boxes = []
                for gt_box in gt_boxes:
                    if np.random.random() < perf_level:  # Chance to detect based on performance
                        # Add noise to ground truth box
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
                    predictions.append("No abnormalities detected.")
        
        # Evaluate this model
        results = evaluator.evaluate_model_predictions(
            predictions, ground_truths, model_name, "simulated_test"
        )
        evaluation_results[model_name] = results
    
    # Compare all models
    print("\n🏆 Comparing all models...")
    comparison_results = evaluator.compare_models(evaluation_results, save_plots=True)
    
    # Print summary
    print(comparison_results["summary"])
    
    print(f"\n📁 Results saved to: {evaluator.output_dir}")
    print("📊 Generated plots:")
    for plot_path in comparison_results["plot_paths"]:
        print(f"   - {plot_path}")

if __name__ == "__main__":
    main()