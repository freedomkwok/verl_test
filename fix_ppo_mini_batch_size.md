# Fix for ppo_mini_batch_size = 0 Error

## 🔍 Root Cause Analysis

The error occurs because `ppo_mini_batch_size` becomes 0 after normalization. Here's the formula:

```python
# From verl/workers/fsdp_workers.py lines 222-223
self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
self.config.actor.ppo_mini_batch_size //= self.device_mesh.size() // self.ulysses_sequence_parallel_size
```

### 📊 The Math:
1. **Initial**: `ppo_mini_batch_size = 256` (default)
2. **Multiply by n**: `256 * 1 = 256` (your rollout.n = 1)
3. **Divide by world_size**: `256 // (world_size // ulysses_sequence_parallel_size)`

### 🚨 The Problem:
If you have many GPUs, the division results in 0:
- 4 GPUs: `256 // 4 = 64` ✅
- 8 GPUs: `256 // 8 = 32` ✅  
- 16 GPUs: `256 // 16 = 16` ✅
- 32 GPUs: `256 // 32 = 8` ✅
- 64 GPUs: `256 // 64 = 4` ✅
- 128 GPUs: `256 // 128 = 2` ✅
- 256+ GPUs: `256 // 256 = 1` ✅
- 512+ GPUs: `256 // 512 = 0` ❌

## ✅ Solution

You need to increase `ppo_mini_batch_size` in your config. Add this to your `gmail_multiturn_grpo.yaml`:

```yaml
actor_rollout_ref:
  actor:
    ppo_mini_batch_size: 1024  # Increase from default 256
    ppo_micro_batch_size_per_gpu: 8  # Set a reasonable micro batch size
```

## 📋 Complete Fix

Add this section to your config file:

```yaml
actor_rollout_ref:
  hybrid_engine: True
  actor:
    # Increase mini batch size to handle large GPU counts
    ppo_mini_batch_size: 1024
    ppo_micro_batch_size_per_gpu: 8
    use_dynamic_bsz: false
    ppo_max_token_len_per_gpu: 16384
  rollout:
    # ... your existing rollout config
```

## 🎯 Recommended Values

Based on your setup:

| GPU Count | Recommended ppo_mini_batch_size |
|-----------|--------------------------------|
| 1-4       | 256                           |
| 4-8       | 512                           |
| 8-16      | 1024                          |
| 16-32     | 2048                          |
| 32+       | 4096                          |

## 🔧 Alternative: Use Dynamic Batch Size

You can also enable dynamic batch sizing:

```yaml
actor_rollout_ref:
  actor:
    use_dynamic_bsz: true
    ppo_max_token_len_per_gpu: 16384
```

This automatically adjusts batch sizes based on token length rather than fixed sizes.
