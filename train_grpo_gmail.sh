set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

WAND_PROJECT=${WAND_PROJECT:-'OpenRL-GRPO-GMAIL'}
export AGENTGYM_ENV_NAME=${AGENTGYM_ENV_NAME:-'gmail'}

n_gpu=${1:-1}
export MODEL_NUM=${2:-''}
export BASE_MODEL="/data/models/QWEN1_5B_0815_A$MODEL_NUM"

# Set CUDA_VISIBLE_DEVICES based on n_gpu
if [ "$n_gpu" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
elif [ "$n_gpu" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
elif [ "$n_gpu" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
elif [ "$n_gpu" -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
else
    # For custom GPU sizes, use sequential numbering
    gpu_list=""
    for i in $(seq 0 $((n_gpu-1))); do
        if [ -n "$gpu_list" ]; then
            gpu_list="$gpu_list,$i"
        else
            gpu_list="$i"
        fi
    done
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$gpu_list}
fi

echo "[Config] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export AGENTGYM_SERVER_BASE="http://127.0.0.1"
export AGENTGYM_PORTS_STR="8000"

export DATA_DIR=${DATA_DIR_OVERRIDE:-"/workspace/OpenRL2/data/$AGENTGYM_ENV_NAME"} # Default data dir based on env name
export EXPERIMENT_NAME="OpenRL-GRPO-${BASE_MODEL##*/}-${AGENTGYM_ENV_NAME}"

echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
TRAIN_FILE="/workspace/OpenRL2/data/_grpo/train1.parquet"
TEST_FILE="/workspace/OpenRL2/data/_grpo/val1.parquet"


PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun --nproc_per_node=1 verl/trainer/main_ppo.py \

#NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALL NCCL_P2P_DISABLE=0 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
TORCH_COMPILE=0 TORCHINDUCTOR_MAX_AUTOTUNE=0 TORCHINDUCTOR_COMPILE_THREADS=10 TORCHINDUCTOR_NUM_WORKERS=1 \
TORCH_LOGS=-dynamo TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor_cache \
TORCH_DISABLE_COMPILE=1 TORCH_DISABLE_INDUCTOR=1 \
OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5 \
SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=true \
SGLANG_ATTENTION_BACKEND=XFORMERS \
TORCH_DISTRIBUTED_USE_SPAWN=1 \
GLANG_DISABLE_PIDFD=1 \
DEBUGGY_LOCAL=True WANDB_DISABLE_ARTIFACTS=True WANDB_DISABLE_CODE=True WANDB_CONSOLE=off WANDB_START_METHOD='thread' \
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gmail_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$TEST_FILE \
    data.train_batch_size=4 \
    data.val_batch_size=1 \
    data.prompt_key='raw_prompt' \
    data.max_prompt_length=6000 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.dataloader_num_workers=1 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((n_gpu * 2)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$n_gpu \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    reward_model.strategy=fsdp2 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.ppo_mini_batch_size=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$n_gpu \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/data/verl_checkpoints/$EXPERIMENT_NAME \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=False \
    2>&1 | tee "/output/logs/$(date +%Y%m%d_%H%M%S).log"
