## Data parquet creation
Using your llava dataset created with RadVLM repo, you can create the parquet data files by executing the following command (change as you like):

### For general multimodal tasks (BLEU reward):
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic
```

### For grounding tasks (IOU reward):

**Option 1: Process directly from LLaVA JSON (RECOMMENDED)**
```
# Process your all_train_llava.json directly to VERL format (ROBUST VERSION):
python examples/data_preprocess/llava_json_to_verl_iou_robust.py /path/to/all_train_llava.json --local_dir SCRATCH/<username>/data/verl_grounding_iou

# Filter for specific datasets only (e.g., VinDr-CXR):
python examples/data_preprocess/llava_json_to_verl_iou_robust.py /path/to/all_train_llava.json --local_dir SCRATCH/<username>/data/verl_grounding_iou --dataset_filter vindr-cxr vindr-cxr-mono

# Adjust batch size if you have memory issues:
python examples/data_preprocess/llava_json_to_verl_iou_robust.py /path/to/all_train_llava.json --local_dir SCRATCH/<username>/data/verl_grounding_iou --batch_size 5000

**Important**: Make sure your SLURM script uses the correct configuration:
- `data.image_key=images` (not `image`)
- `custom_reward_function.path=$WORK_DIR/custom_reward/iou_reward.py` (note: `custom_reward` not `custom_rewards`)

# Alternative scripts for other formats:
python examples/data_preprocess/grounding_iou.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/grounding_iou
python examples/data_preprocess/vindr_grounding_iou.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/vindr_grounding_iou
```

**Option 2: Convert existing LLaVA-format parquet files to VERL format**
```
# If you already have LLaVA format parquet files (like train_vindr_grounding_iou.parquet):
python examples/data_preprocess/convert_llava_to_verl_iou.py /path/to/train_vindr_grounding_iou.parquet --output_dir SCRATCH/<username>/data/verl_grounding --split_name train

# Also convert validation set if you have it:
python examples/data_preprocess/convert_llava_to_verl_iou.py /path/to/val_vindr_grounding_iou.parquet --output_dir SCRATCH/<username>/data/verl_grounding --split_name val
```

## Reward function 
The reward function that worked for me using BLEU metric was created here: `custom_rewards/bleu_reward.py`. For grounding tasks with bounding boxes, an IOU (Intersection over Union) reward function is available at `custom_reward/iou_reward.py`. You can create other reward files, always specifying the name of the function that calls it (currently `compute_score` in the training scripts).

### Reward Functions for Medical Grounding

We provide multiple reward functions for different evaluation needs:

#### 📊 Basic IOU Reward (`custom_reward/iou_reward.py`)
Simple spatial accuracy evaluation:
- Ground truth: List of bounding boxes in format `[[x1, y1, x2, y2], ...]` or empty list `[]` for "no finding" cases
- Model output: Text containing bounding boxes in the `<answer>` section with format `[x1, y1, x2, y2]`

**Scoring logic:**
- **Perfect box match**: 1.0 score
- **Partial box overlap**: IOU score (0.0-1.0 based on overlap)
- **Correct "no finding"**: 0.8 score (when ground truth is empty and model says "no abnormalities")
- **Incorrect predictions**: 0.0 score (false positives or false negatives)
- **⚠️ Limitation**: Zero gradient for non-overlapping boxes (no learning signal)

#### 📐 GIoU Reward (`custom_reward/giou_reward.py`) - **Recommended**
**Based on "Generalized Intersection over Union" (Stanford, CVPR 2019)**

Addresses IoU's main weakness by providing meaningful scores for non-overlapping boxes:

**Key Improvements:**
- **Always differentiable**: Provides gradient even when boxes don't overlap
- **Distance-aware**: Closer non-overlapping boxes get better scores than distant ones
- **Scale-invariant**: Maintains IoU's desirable properties
- **Proven effective**: Used successfully in YOLO v3, Faster R-CNN, Mask R-CNN

**GIoU Formula:** `GIoU = IoU - |C \ (A ∪ B)| / |C|`
Where C is the smallest box enclosing both predicted and ground truth boxes.

**Example advantage:**
- IoU: Both `[0.1,0.1,0.3,0.3]` and `[0.4,0.4,0.6,0.6]` vs GT `[0.7,0.7,0.9,0.9]` → 0.0
- GIoU: Closer box gets -0.680, distant box gets -0.875 (provides learning direction!)

#### 🎯 mAP Reward (`custom_reward/map_reward.py`) - **Industry Standard**
**Based on COCO evaluation methodology - comprehensive multi-threshold assessment**

Implements mean Average Precision (mAP) across multiple IoU thresholds for robust evaluation:

**Key Features:**
- **Multi-threshold evaluation**: mAP@[0.5:0.05:0.95] (10 IoU thresholds from 0.5 to 0.95)
- **Precision-Recall curves**: Full AP calculation with 101-point interpolation (COCO standard)
- **Industry standard**: Same metric used in COCO, PASCAL VOC, and major detection competitions
- **Comprehensive metrics**: AP@0.50, AP@0.75, precision, recall, and detailed breakdowns
- **Robust assessment**: Handles varying detection quality gracefully

**Scoring Breakdown:**
- **Base Score**: mAP@[0.5:0.05:0.95] (primary metric)
- **Precision Bonus**: 0.1 × Precision@0.50 (clinical relevance - avoid false positives)
- **Recall Bonus**: 0.1 × Recall@0.50 (don't miss findings)
- **False Positive Penalty**: 0.05 × FP_rate (reduce noise)

**Best for:** Research, benchmarking, publication-quality evaluation, detailed performance analysis

#### 🧠 Enhanced Medical Reward (`custom_reward/enhanced_medical_reward.py`)
**Inspired by "Enhancing Abnormality Grounding for Vision Language Models with Knowledge Descriptions" (arXiv:2503.03278)**

Multi-criteria evaluation combining:
1. **Spatial Accuracy** (40%/30%): IOU with anatomical importance weighting
2. **Semantic Understanding** (30%/40%): Medical terminology and anatomical awareness
3. **Clinical Reasoning** (20%): Quality of diagnostic thinking process
4. **Clinical Relevance** (10%): Actionability and specificity of findings

**Key Features:**
- **Knowledge Decomposition**: Breaks down medical concepts into fundamental attributes
- **Anatomical Context**: Weights findings by clinical importance (heart=1.0, lung=0.9, etc.)
- **Severity Weighting**: Different abnormalities have different clinical significance
- **Reasoning Evaluation**: Analyzes `<think>` content for systematic approach, differential diagnosis
- **Fallback Safety**: Reverts to basic IOU if enhanced scoring fails

**Medical Knowledge Base:**
- 5 anatomical regions with importance weights
- 7 abnormality types with severity scores
- Visual descriptors (size, shape, density, location)
- Clinical reasoning quality indicators 

## Data parquet creation

### Option 1: Fast Pre-filtering (Recommended for large datasets)

For datasets with >100k samples, use the fast version that pre-filters based on text length:

```bash
cd examples/data_preprocess
python3 llava_json_to_verl_iou_fast.py \
  --json_path /path/to/your/all_train_llava.json \
  --local_dir ./data \
  --dataset_filter vindr-cxr \
  --batch_size 5000
```

This script pre-filters samples with >8192 characters (~2048 tokens) during data creation, eliminating the need for slow multimodal filtering during training.

### Option 2: Standard Processing

For smaller datasets or when you need exact token counting:

```bash
cd examples/data_preprocess
python3 llava_json_to_verl_iou_robust.py \
  --json_path /path/to/your/all_train_llava.json \
  --local_dir ./data \
  --dataset_filter vindr-cxr \
  --batch_size 5000
```

Both scripts will:
- Filter for grounding tasks (samples containing bounding boxes)
- Extract ground truth bounding boxes from GPT responses
- Handle "no finding" cases appropriately
- Create proper VERL format with `prompt`, `images`, `reward_model`, etc.
- Process data in batches to avoid memory issues

## Slurm script 

### Choosing Your Reward Function

#### Quick Setup (Choose Your Reward Function)
```bash
# Generate SLURM scripts for all reward functions
python3 custom_reward/reward_config.py

# Recommended: GIoU reward (simple improvement over IoU)
sbatch jobs/single_node_giou.sh

# Industry Standard: mAP reward (comprehensive evaluation)
sbatch jobs/single_node_map.sh

# Advanced: Enhanced medical reward (best for clinical applications)
sbatch jobs/single_node_enhanced.sh

# Baseline: Basic IOU reward (for comparison)
sbatch jobs/single_node_basic.sh
```

#### Manual Configuration
In your SLURM script, set:
```bash
# For GIoU reward (recommended for general use)
custom_reward_function.path=$WORK_DIR/custom_reward/giou_reward.py

# For mAP reward (industry standard evaluation)
custom_reward_function.path=$WORK_DIR/custom_reward/map_reward.py

# For enhanced medical reward (clinical applications)
custom_reward_function.path=$WORK_DIR/custom_reward/enhanced_medical_reward.py

# For basic IOU reward (baseline)
custom_reward_function.path=$WORK_DIR/custom_reward/iou_reward.py
```

#### Fast Training Options
For faster startup, all scripts disable slow multimodal filtering:
- `data.filter_overlong_prompts=false`
- Uses pre-filtered data from `llava_json_to_verl_iou_fast.py`

### Legacy Scripts
You will find the original scripts in `jobs` folder. 

## Checkpoint conversion to HF
```
python -m verl.model_merger merge --backend fsdp --local_dir /your/checkpoint/path/global_step_X/actor --target_dir /your/checkpoint/path/huggingface/
```


