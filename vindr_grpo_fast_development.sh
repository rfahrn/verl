#!/bin/bash
#SBATCH --job-name=vindr_fast_dev
#SBATCH --account=a135
#SBATCH --time=02:00:00
#SBATCH --environment=verl
#SBATCH --output=job_outputs/%x_%j.txt
#SBATCH --error=job_outputs/%x_%j.err

unset ROCR_VISIBLE_DEVICES
set -x

ENGINE=${1:-vllm}
export DEBUG_REWARD=true

# Environment setup
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1
export WORK_DIR=$SCRATCH/code/verl

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export VLLM_USE_MODELSCOPE=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# WandB configuration
export WANDB_API_KEY=15b5344c70fad59908246ded2a98fdef6a4e9eda
export WANDB_PROJECT=vindr_grpo_map
export WANDB_MODE=online

cd $WORK_DIR

echo "🚀 FAST DEVELOPMENT MODE - AGGRESSIVE OPTIMIZATIONS"
echo "⚡ Optimizations applied:"
echo "   • Smaller images (256x256 instead of 512x512)"
echo "   • Shorter sequences (256 tokens max)"
echo "   • Smaller batches for faster iteration"
echo "   • Single rollout (n=1) for speed"
echo "   • Reduced dataset size"
echo "   • More frequent validation"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$SCRATCH/dataset/vindr_all_new_labels/train.parquet \
    data.val_files=$SCRATCH/dataset/vindr_all_new_labels/val.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    data.train_max_samples=1000 \
    data.val_max_samples=100 \
    actor_rollout_ref.model.path=/capstor/store/cscs/swissai/a135/RadVLM_project/models_a135/Qwen2.5-VL-7B-CS-FULL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=$WORK_DIR/custom_rewards/vindr_reward_hybrid_simplified.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name='vindr_grpo_map' \
    trainer.experiment_name='qwen_fast_dev' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=2

echo "⚡ FAST DEVELOPMENT TRAINING COMPLETED!"
echo "📊 Speed Optimizations Applied:"
echo "   • 🖼️  Image size: 256x256 (was 512x512)"
echo "   • 📝 Max tokens: 256+256=512 (was 1024+)"
echo "   • 🎯 Batch size: 8 (was 24)"
echo "   • 🔄 Rollouts: n=1 (was n=2)"
echo "   • 📊 Dataset: 1000 train + 100 val samples"
echo "   • ⚡ Expected: ~10x faster training"