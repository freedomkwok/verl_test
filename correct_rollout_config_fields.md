# RolloutConfig Valid Fields Documentation

## ❌ Invalid Fields (Not in RolloutConfig)
- `max_response_length` - This is a **data** field, not a rollout field
- `max_prompt_length` - This is a **data** field, not a rollout field
- `nnodes` - Not a valid RolloutConfig field
- `node_rank` - Not a valid RolloutConfig field
- `frequency_penalty` - Not a valid RolloutConfig field
- `presence_penalty` - Not a valid RolloutConfig field
- `repetition_penalty` - Not a valid RolloutConfig field
- `stop` - Not a valid RolloutConfig field

## ✅ Valid RolloutConfig Fields

### Core Fields
- `name`: str - Rollout engine name (e.g., "sglang", "vllm", "hf")
- `mode`: str - "sync" or "async" (default: "sync")

### Sampling Parameters
- `temperature`: float (default: 1.0)
- `top_k`: int (default: -1)
- `top_p`: float (default: 1.0)
- `do_sample`: bool (default: True)
- `n`: int (default: 1)

### Length Parameters
- `prompt_length`: int (default: 512) - **This is the rollout field**
- `response_length`: int (default: 512) - **This is the rollout field**

### Model Parameters
- `dtype`: str (default: "bfloat16")
- `gpu_memory_utilization`: float (default: 0.5)
- `ignore_eos`: bool (default: False)
- `enforce_eager`: bool (default: True)
- `free_cache_engine`: bool (default: True)
- `tensor_model_parallel_size`: int (default: 2)
- `max_num_batched_tokens`: int (default: 8192)
- `max_num_seqs`: int (default: 1024)

### Advanced Parameters
- `over_sample_rate`: float (default: 0.0)
- `max_model_len`: Optional[int] (default: None)
- `calculate_log_probs`: bool (default: False)
- `load_format`: str (default: "dummy_dtensor")

### Multi-turn Configuration
- `multi_turn`: MultiTurnConfig object with fields:
  - `enable`: bool (default: False)
  - `max_assistant_turns`: Optional[int] (default: None)
  - `max_user_turns`: Optional[int] (default: None)
  - `tool_config_path`: Optional[str] (default: None)
  - `interaction_config_path`: Optional[str] (default: None)
  - `use_inference_chat_template`: bool (default: False)
  - `tokenization_sanity_check_mode`: str (default: "strict")

### Validation Parameters
- `val_kwargs`: SamplingConfig object

### Other Fields
- `engine_kwargs`: dict (default: {})
- `agent`: AgentLoopConfig object
- `trace`: TraceConfig object
- `server`: ServerConfig object
- `profiler`: Optional[ProfilerConfig] (default: None)

## 📋 Corrected Configuration

Here's how your config should look:

```yaml
actor_rollout_ref:
  hybrid_engine: True
  rollout:
    name: sglang
    mode: async
    load_format: dummy_dtensor
    enforce_eager: False
    free_cache_engine: True
    
    # Model and distributed configs
    dtype: float16
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.5
    ignore_eos: False
    max_num_batched_tokens: 8192

    # Length configs (these are the correct rollout fields)
    prompt_length: 6000
    response_length: 1000
    max_model_len: 7000  # prompt_length + response_length
    
    # Multi-turn configs
    multi_turn:
      enable: True
      max_assistant_turns: 3
      max_user_turns: 2
      tool_config_path: null
      interaction_config_path: examples/sglang_multiturn/config/interaction_config/gmail_interaction_config.yaml
      use_inference_chat_template: false
      tokenization_sanity_check_mode: disable

    # Sampling configs
    calculate_log_probs: true
    temperature: 0.7
    top_p: 0.9
    top_k: -1
    n: 1
    do_sample: true
```

## 🔍 Key Differences

1. **Length Fields**: Use `prompt_length` and `response_length` in rollout, not `max_prompt_length` and `max_response_length`
2. **Data vs Rollout**: `max_prompt_length` and `max_response_length` belong in the `data` section
3. **Removed Invalid Fields**: Removed `nnodes`, `node_rank`, `frequency_penalty`, `presence_penalty`, `repetition_penalty`, `stop`
4. **Correct Load Format**: Use `dummy_dtensor` instead of `dummy`
