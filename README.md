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

#### 📊 Basic IOU Reward (`custom_reward/iou_reward.py`)
Simple spatial accuracy evaluation:
- Ground truth: List of bounding boxes in format `[[x1, y1, x2, y2], ...]` or empty list `[]` for "no finding" cases
- Model output: Text containing bounding boxes in the `<answer>` section with format `[x1, y1, x2, y2]`

**Scoring logic:**
- **Perfect box match**: 1.0 score
- **Partial box overlap**: IOU score (0.0-1.0 based on overlap)
- **Correct "no finding"**: 0.8 score (when ground truth is empty and model says "no abnormalities")
- **Incorrect predictions**: 0.0 score (false positives or false negatives)

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

#### Quick Setup with Enhanced Medical Reward (Recommended)
```bash
# Generate SLURM scripts for both reward functions
python3 custom_reward/reward_config.py

# Use enhanced medical reward (best for clinical applications)
sbatch jobs/single_node_enhanced.sh

# Or use basic IOU reward (faster, good for testing)
sbatch jobs/single_node_basic.sh
```

#### Manual Configuration
In your SLURM script, set:
```bash
# For enhanced medical reward
custom_reward_function.path=$WORK_DIR/custom_reward/enhanced_medical_reward.py

# For basic IOU reward  
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


