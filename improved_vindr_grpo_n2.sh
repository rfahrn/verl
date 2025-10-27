#!/bin/bash
#SBATCH --job-name=vindr_clean_grpo_n2
#SBATCH --account=a135
#SBATCH --time=10:00:00
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

echo "🔧 Installing required packages..."

# Install NLTK
pip install nltk

# Install spaCy and English model
echo "📦 Installing spaCy..."
pip install spacy

echo "🧠 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Verify spaCy installation
echo "✅ Verifying spaCy installation..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy en_core_web_sm model loaded successfully!')
    # Test on medical text
    doc = nlp('The cardiomegaly is positioned at coordinates.')
    chunks = [chunk.text for chunk in doc.noun_chunks]
    print(f'✅ Test extraction: {chunks}')
except Exception as e:
    print(f'❌ spaCy installation failed: {e}')
    print('⚠️  Will fall back to regex-based extraction')
"

# Install other useful packages for medical NLP (optional)
echo "📚 Installing additional NLP packages..."
pip install --quiet regex
pip install --quiet scikit-learn

echo "🚀 Starting Clean VinDR GRPO Training with n=2..."
echo "Dataset: VinDR-CXR (coordinates as-is, simple approach)"
echo "Reward: Clean mAP with spaCy-based label extraction bonus"
echo "GRPO: n=2 responses per prompt for relative ranking"
echo "spaCy Model: en_core_web_sm for intelligent medical entity extraction"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$SCRATCH/dataset/vindr_all_new_labels/train.parquet \
    data.val_files=$SCRATCH/dataset/vindr_all_new_labels/val.parquet \
    data.train_batch_size=24 \
    data.val_batch_size=12 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.image_key=images \
    actor_rollout_ref.model.path=/capstor/store/cscs/swissai/a135/RadVLM_project/models_a135/Qwen2.5-VL-7B-CS-FULL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.clip_ratio=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=$WORK_DIR/custom_rewards/vindr_reward_final_clean.py \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='[console,wandb]' \
    trainer.project_name='vindr_grpo_map' \
    trainer.experiment_name='qwen_map_reward_spacy_n2' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=5

echo "🎉 Clean VinDR GRPO Training with n=2 Completed!"
echo "📊 Results:"
echo "   • Multi-box fuzzy mAP scoring: ✅ Preserved"
echo "   • spaCy medical label extraction: ✅ Added as bonus"
echo "   • GRPO n=2: ✅ Optimized for faster training"
echo "   • Training log: job_outputs/vindr_clean_grpo_n2_%j.txt"
echo "   • Error log: job_outputs/vindr_clean_grpo_n2_%j.err"