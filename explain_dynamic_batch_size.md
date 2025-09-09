# Understanding `use_dynamic_bsz` (Dynamic Batch Size)

## 🎯 What `use_dynamic_bsz` Does

**No, it doesn't match 400 tokens to 8000 tokens.** Instead, it dynamically groups samples based on their **total token count** to optimize GPU memory usage.

## 📊 How It Works

### **Without Dynamic Batch Size (`use_dynamic_bsz: false`):**
```python
# Fixed micro batch size
micro_batches = mini_batch.split(ppo_micro_batch_size_per_gpu)  # e.g., split into groups of 2
```

### **With Dynamic Batch Size (`use_dynamic_bsz: true`):**
```python
# Dynamic grouping based on token count
max_token_len = ppo_max_token_len_per_gpu * ulysses_sequence_parallel_size  # e.g., 16384
micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
```

## 🔍 The Algorithm (from `seqlen_balancing.py`)

1. **Calculate effective sequence length** for each sample:
   ```python
   seq_len_effective = attention_mask.sum(dim=1)  # Actual token count per sample
   ```

2. **Group samples** so total tokens ≤ `max_token_len`:
   ```python
   # Example with max_token_len = 16384:
   # Sample 1: 400 tokens  → Group 1: 400 tokens
   # Sample 2: 8000 tokens → Group 2: 8000 tokens  
   # Sample 3: 2000 tokens → Group 2: 8000 + 2000 = 10000 tokens
   # Sample 4: 7000 tokens → Group 3: 7000 tokens
   ```

3. **Balance workload** using sum of squared sequence lengths:
   ```python
   # Approximates attention computation cost
   workload = sum(seq_len ** 2 for seq_len in group)
   ```

## 📋 Your Configuration

```yaml
actor:
  use_dynamic_bsz: true
  ppo_max_token_len_per_gpu: 16384  # Max tokens per GPU
```

### **What This Means:**
- **Max tokens per micro-batch**: 16384 tokens
- **Your samples**: Up to 8000 tokens each (from `max_prompt_length: 8000`)
- **Grouping**: Samples are grouped so total tokens ≤ 16384

## 🎯 Examples

### **Scenario 1: Short Samples**
```
Sample 1: 400 tokens
Sample 2: 600 tokens  
Sample 3: 500 tokens
Sample 4: 300 tokens
```
**Result**: All 4 samples in one micro-batch (total: 1800 tokens < 16384)

### **Scenario 2: Long Samples**
```
Sample 1: 8000 tokens
Sample 2: 7000 tokens
Sample 3: 6000 tokens
```
**Result**: 
- Micro-batch 1: Sample 1 (8000 tokens)
- Micro-batch 2: Sample 2 (7000 tokens) 
- Micro-batch 3: Sample 3 (6000 tokens)

### **Scenario 3: Mixed Lengths**
```
Sample 1: 8000 tokens
Sample 2: 4000 tokens
Sample 3: 3000 tokens
Sample 4: 2000 tokens
```
**Result**:
- Micro-batch 1: Sample 1 (8000 tokens)
- Micro-batch 2: Sample 2 + Sample 3 (4000 + 3000 = 7000 tokens)
- Micro-batch 3: Sample 4 (2000 tokens)

## ✅ Benefits

1. **Memory Efficiency**: No wasted GPU memory on padding
2. **Better Throughput**: More samples per batch when possible
3. **Automatic Optimization**: Adapts to your actual data distribution
4. **Workload Balancing**: Groups samples by computational complexity

## 🎯 Key Point

**Dynamic batch size groups samples by total token count, not by individual sequence length.** It's about optimizing GPU memory usage across the entire batch, not matching individual samples to a target length.
