# mAP-based Evaluation Guide

## 🎯 The Key Insight

**TRAINING vs EVALUATION** - Use different metrics for different purposes:

- **🏃 TRAINING**: Use fast reward functions (IoU, GIoU) for efficient learning
- **📊 EVALUATION**: Use comprehensive mAP metrics for rigorous performance assessment

This approach allows you to:
1. Train models efficiently with simple reward functions
2. Evaluate all models fairly using the same comprehensive standard
3. Determine which training approach produces the best final performance

## 🚀 Quick Start

### Step 1: Train Models with Different Reward Functions

```bash
# Train with different reward functions
sbatch jobs/single_node_basic.sh     # Basic IoU reward
sbatch jobs/single_node_giou.sh      # GIoU reward (recommended)
sbatch jobs/single_node_map.sh       # mAP reward (comprehensive)
sbatch jobs/single_node_enhanced.sh  # Enhanced Medical reward (clinical)
```

### Step 2: Evaluate All Models Using mAP

```bash
# Demo evaluation (no actual models needed)
python3 evaluate_trained_models.py

# Real evaluation with your trained models
python3 evaluate_trained_models.py \
  --test_data data/val_verl_iou_fast.parquet \
  --model_dirs \
    checkpoints/model_iou_reward/ \
    checkpoints/model_giou_reward/ \
    checkpoints/model_enhanced_reward/ \
  --output_dir evaluation/results
```

### Step 3: Analyze Results

The evaluation will generate:
- **Comprehensive metrics**: mAP@[0.5:0.05:0.95], AP@0.50, AP@0.75, Precision, Recall
- **Statistical significance tests**: Determine if differences are meaningful
- **Visual comparisons**: Plots showing performance across IoU thresholds
- **Detailed reports**: CSV and JSON files with all results

## 📊 Understanding the Results

### Key Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **mAP@[0.5:0.05:0.95]** | Mean AP across 10 IoU thresholds | Overall detection quality |
| **AP@0.50** | AP at IoU threshold 0.5 | Performance at standard detection threshold |
| **AP@0.75** | AP at IoU threshold 0.75 | Performance at strict localization threshold |
| **Precision@0.50** | Precision at IoU 0.5 | How many detections are correct |
| **Recall@0.50** | Recall at IoU 0.5 | How many ground truths are found |

### Performance Ranking

Models are ranked by **mAP@[0.5:0.05:0.95]** - the industry standard metric used in:
- COCO Detection Challenge
- PASCAL VOC Challenge  
- Major computer vision conferences
- Medical imaging competitions

## 🔬 Expected Results

Based on the reward function design, you should expect:

### 1. **Basic IoU Training** (Baseline)
- **Expected mAP**: ~0.45-0.55
- **Strengths**: Fast training, simple implementation
- **Weaknesses**: No gradient for non-overlapping boxes, limited learning

### 2. **GIoU Training** (Recommended)  
- **Expected mAP**: ~0.55-0.70
- **Strengths**: Better gradients, improved localization
- **Why better**: Meaningful scores for non-overlapping boxes guide learning

### 3. **mAP Training** (Comprehensive)
- **Expected mAP**: ~0.60-0.75  
- **Strengths**: Directly optimizes evaluation metric
- **Trade-off**: Slower training due to complex reward computation

### 4. **Enhanced Medical Training** (Clinical)
- **Expected mAP**: ~0.65-0.80
- **Strengths**: Domain knowledge, clinical reasoning
- **Best for**: Medical applications requiring clinical accuracy

## 📈 Interpreting Differences

### Statistical Significance
- **p < 0.05**: Difference is statistically significant
- **p ≥ 0.05**: Difference could be due to random variation

### Practical Significance
- **ΔmAP < 0.02**: Small difference, may not be practically important
- **ΔmAP 0.02-0.05**: Moderate improvement worth considering
- **ΔmAP > 0.05**: Substantial improvement, definitely use better method

### Per-Threshold Analysis
- **High performance at IoU=0.5, low at IoU=0.9**: Good detection, poor localization
- **Consistent performance across thresholds**: Well-calibrated model
- **Sharp drop after IoU=0.7**: Model struggles with precise localization

## 🛠️ Advanced Usage

### Custom IoU Thresholds

```python
from evaluation.map_evaluator import mAPEvaluator

# Custom thresholds for medical imaging
evaluator = mAPEvaluator(
    iou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Lower thresholds for medical
    output_dir="evaluation/medical_results"
)
```

### Evaluating from Parquet Files

```python
# If you have model outputs saved in parquet format
results = evaluator.evaluate_from_parquet(
    model_outputs_path="model_predictions.parquet",
    model_name="my_model",
    answer_column="model_answer",
    ground_truth_column="reward_model"
)
```

### Domain-Specific Analysis

```python
# Analyze performance on specific subsets
positive_samples = [s for s in results["per_sample_results"] 
                   if s["num_ground_truth"] > 0]
negative_samples = [s for s in results["per_sample_results"] 
                   if s["num_ground_truth"] == 0]

print(f"Performance on positive samples: {np.mean([s['map_score'] for s in positive_samples]):.3f}")
print(f"Accuracy on negative samples: {np.mean([1.0 if s['num_predictions']==0 else 0.0 for s in negative_samples]):.3f}")
```

## 📋 Best Practices

### 1. **Consistent Test Set**
- Use the same test set for all models
- Ensure test set is representative of your target domain
- Keep test set separate from training/validation data

### 2. **Multiple Metrics**
- Don't rely solely on mAP - consider precision, recall, and domain-specific metrics
- Analyze performance across different IoU thresholds
- Consider computational cost vs performance trade-offs

### 3. **Statistical Rigor**
- Use statistical significance tests for comparisons
- Report confidence intervals when possible
- Consider multiple random seeds if training is stochastic

### 4. **Domain Considerations**
- For medical imaging: Lower IoU thresholds might be more appropriate
- For safety-critical applications: Prioritize recall over precision
- For high-throughput applications: Consider inference speed

## 🎯 Real-World Example

```bash
# 1. Train models (this takes hours/days)
sbatch jobs/single_node_giou.sh      # GIoU training
sbatch jobs/single_node_enhanced.sh  # Enhanced medical training

# 2. Wait for training to complete...

# 3. Evaluate both models (this takes minutes)
python3 evaluate_trained_models.py \
  --test_data data/val_verl_iou_fast.parquet \
  --model_dirs \
    checkpoints/verl_iou_grounding_giou/global_step_1000/ \
    checkpoints/verl_iou_grounding_enhanced/global_step_1000/ \
  --output_dir evaluation/giou_vs_enhanced

# 4. Analyze results
cat evaluation/giou_vs_enhanced/comparison_summary.txt
```

## 🔍 Troubleshooting

### Common Issues

1. **Low mAP scores across all models**
   - Check ground truth quality and format
   - Verify bounding box coordinate system
   - Ensure test set difficulty is appropriate

2. **No significant differences between models**
   - Increase test set size
   - Check if models are actually different
   - Consider domain-specific metrics

3. **Inconsistent results**
   - Use multiple random seeds
   - Check for data leakage between train/test
   - Verify evaluation implementation

### Performance Debugging

```python
# Check individual sample performance
for sample in results["per_sample_results"][:5]:
    print(f"Sample {sample['sample_id']}: mAP={sample['map_score']:.3f}")
    print(f"  Prediction: {sample['prediction'][:100]}...")
    print(f"  Ground truth: {sample['ground_truth']}")
    print(f"  Predicted boxes: {len(sample['predicted_boxes'])}")
```

## 📚 References

1. **COCO Evaluation**: Lin et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
2. **mAP Definition**: Everingham et al. "The Pascal Visual Object Classes (VOC) Challenge." IJCV 2010.
3. **Medical Imaging Evaluation**: Rajpurkar et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays." arXiv 2017.

## 🤝 Contributing

To extend the evaluation framework:

1. **Add new metrics**: Extend `mAPEvaluator` class
2. **Add domain-specific analysis**: Create specialized evaluator subclasses  
3. **Add visualization**: Extend `_generate_comparison_plots` method
4. **Add statistical tests**: Extend `_compute_statistical_significance` method

The evaluation framework is designed to be modular and extensible for different domains and use cases.