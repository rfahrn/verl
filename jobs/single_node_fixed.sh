#!/bin/bash
#SBATCH --job-name=test_grpo_grounding_iou   # %x will expand to "interactive_job"
#SBATCH --account=a135               # -A a135
#SBATCH --time=10:00:00              # --time=01:00:00  --environment=verl
#SBATCH --output=job_outputs_grounding2/%x.txt  # redirect stdout to job_outputs/<job-name>.txt
#SBATCH --environment=verl


unset ROCR_VISIBLE_DEVICES
set -x
ENGINE=${1:-vllm}

export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_MULTIPROC=1

export WORK_DIR=$SCRATCH/code/verl
cd $WORK_DIR

export WANDB_API_KEY=15b5344c70fad59908246ded2a98fdef6a4e9eda
export WANDB_PROJECT=GRPO

pip install nltk

# ---------- JOB STEP INSIDE CONTAINER ----------
#srun --environment=verl \
     python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$SCRATCH/dataset/vindr_grounding_dataset/train.parquet \
        data.val_files=$SCRATCH/dataset/vindr_grounding_dataset/val.parquet \
        data.train_batch_size=64 \
        data.max_prompt_length=2048 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.image_key=images \
        actor_rollout_ref.model.path=/capstor/store/cscs/swissai/a135/RadVLM_project/models/Qwen2.5-VL-7B-CS \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.00 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=$ENGINE \
        actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        custom_reward_function.path=$WORK_DIR/custom_reward/iou_reward.py \
        custom_reward_function.name=compute_score \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='grounding_grpo_iou2' \
        trainer.experiment_name='qwen2_5_vl_7b_grounding_iou2' \
        trainer.n_gpus_per_node=4 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=5 \
        trainer.total_epochs=1 $@