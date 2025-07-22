## Data parquet creation
Using your llava dataset created with RadVLM repo, you can create the parquet data files by executing the following command (change as you like):

### For general multimodal tasks (BLEU reward):
```
python examples/data_preprocess/mimic_mm.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/mimic
```

### For grounding tasks (IOU reward):

**Option 1: Process directly from LLaVA JSON**
```
# For all grounding tasks in the dataset:
python examples/data_preprocess/grounding_iou.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/grounding_iou

# For specific dataset prefixes (e.g., VinDr-CXR only):
python examples/data_preprocess/grounding_iou.py /path/to/dataset.json --local_dir SCRATCH/<username>/data/vindr_grounding --dataset_filter "vindr-cxr"

# Alternative VinDr-specific script:
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

### IOU Reward for Grounding Tasks
The IOU reward function computes the Intersection over Union between predicted and ground truth bounding boxes. It expects:
- Ground truth: List of bounding boxes in format `[[x1, y1, x2, y2], ...]`
- Model output: Text containing bounding boxes in the `<answer>` section with format `[x1, y1, x2, y2]` 

## Slurm script 
You will find what worked for me until now for running GRPO in `jobs` folder. 

## Checkpoint conversion to HF
```
python -m verl.model_merger merge --backend fsdp --local_dir /your/checkpoint/path/global_step_X/actor --target_dir /your/checkpoint/path/huggingface/
```


