#!/bin/bash
#SBATCH --job-name=verl_ppo_enhanced_medical
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --output=logs/verl_ppo_%j.out
#SBATCH --error=logs/verl_ppo_%j.err

# Set working directory
export WORK_DIR=/workspace

# Create logs directory if it doesn't exist
mkdir -p $WORK_DIR/logs

# Load necessary modules and activate environment
source ~/.bashrc

cd $WORK_DIR

# Run VERL PPO training with enhanced_medical reward function
python3 -m verl.trainer.main_ppo \
actor_rollout_ref.model.path=Qwen/Qwen2-VL-2B-Instruct \
actor_rollout_ref.model.enable_prefix_caching=false \
critic.model.path=Qwen/Qwen2-VL-2B-Instruct \
critic.model.enable_prefix_caching=false \
data.train_files=$WORK_DIR/data/train_verl_iou_fast.parquet \
data.val_files=$WORK_DIR/data/val_verl_iou_fast.parquet \
data.image_key=images \
data.prompt_key=prompt \
data.max_prompt_length=2048 \
data.filter_overlong_prompts=false \
data.filter_overlong_prompts_workers=1 \
trainer.default_hdfs_dir=null \
trainer.project_name=verl_iou_grounding_enhanced_medical \
trainer.experiment_name=vindr_grounding_enhanced_medical \
trainer.total_epochs=1 \
trainer.save_freq=1 \
trainer.logging.use_wandb=false \
actor.strategy.name=fsdp \
critic.strategy.name=fsdp \
actor.strategy.kwargs.param_dtype=bfloat16 \
critic.strategy.kwargs.param_dtype=bfloat16 \
custom_reward_function.enable=true \
custom_reward_function.path=$WORK_DIR/custom_reward/enhanced_medical_reward.py
