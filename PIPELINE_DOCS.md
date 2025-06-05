# Pipeline Parallelism Documentation

This document explains the pipeline parallelism implementation in MLX-LM and provides guidance on how to implement pipeline support for new models.

## Overview

Pipeline parallelism enables distributing transformer layers across multiple devices/ranks, allowing inference of large models that don't fit in the memory of a single device. MLX-LM implements pipeline parallelism using MLX's distributed communication primitives.

## Current Pipeline-Enabled Models

The following models currently support pipeline parallelism:

- **Llama** (`llama.py`) - Including Meta-Llama-3-8B-Instruct and variants
- **DeepSeek V2** (`deepseek_v2.py`) - DeepSeek V2 architecture
- **DeepSeek V3** (`deepseek_v3.py`) - DeepSeek V3 architecture with MoE support
- **Qwen3** (`qwen3.py`) - Qwen3 architecture

## Pipeline Implementation Pattern

All pipeline-enabled models follow (roughly) the same implementation pattern:

### 1. Pipeline Method

Each model's main class (e.g., `LlamaModel`, `DeepseekV2Model`, `DeepseekV3Model`, `Qwen3Model`) implements a `pipeline(self, group)` method:

```python
def pipeline(self, group):
    # Split layers in reverse so rank=0 gets the last layers and
    # rank=pipeline_size-1 gets the first
    # eg layers = 10
    self.pipeline_rank = group.rank() # 0
    self.pipeline_size = group.size() # 3
    layers_per_rank = len(self.layers) // self.pipeline_size # 10 // 3 = 3
    extra = len(self.layers) - layers_per_rank * self.pipeline_size # 10 - 3 * 3 = 1
    if self.pipeline_rank < extra:
        layers_per_rank += 1

    self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank # (3 - 0 - 1) * 3 = 6
    self.end_idx = self.start_idx + layers_per_rank # 6 + 3 = 9
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx
    self.num_layers = len(self.layers) - self.start_idx
```

**Key aspects:**
- Layers are distributed in **reverse order**: rank 0 gets the last layers, highest rank gets the first layers
- Handles uneven distribution when `num_layers % pipeline_size != 0`
- Sets `None` for layers not owned by the current rank
- Updates `num_layers` to reflect only locally owned layers

### 2. Forward Pass Communication

The forward pass in the model's `__call__` method implements the communication pattern:

```python
def __call__(self, inputs, mask=None, cache=None):
    h = self.embed_tokens(inputs)
    
    pipeline_rank = self.pipeline_rank
    pipeline_size = self.pipeline_size
    if mask is None:
        mask = create_attention_mask(h, cache)

    if cache is None:
        cache = [None] * self.num_layers

    # Receive from the previous process in the pipeline
    if pipeline_rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (pipeline_rank + 1))

    for i in range(self.num_layers):
        h = self.layers[self.start_idx + i](h, mask, cache[i])

    # Send to the next process in the pipeline
    if pipeline_rank != 0:
        h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)

    # Broadcast h while keeping it in the graph
    h = mx.distributed.all_gather(h)[: h.shape[0]]

    return self.norm(h)
```

**Communication flow:**
1. **Receive**: Each rank (except the first) receives hidden states from the previous rank
2. **Process**: Apply the local transformer layers
3. **Send**: Each rank (except the last) sends hidden states to the next rank
4. **Broadcast**: All ranks receive the final hidden states via `all_gather`

### 3. Top-Level Model Integration

The top-level `Model` class remains unchanged and calls the pipelined model normally:

```python
class Model(nn.Module):
    def __call__(self, inputs, mask=None, cache=None, input_embeddings=None):
        out = self.model(inputs, mask, cache, input_embeddings)
        # Apply language modeling head
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out
```

## Usage Examples

### Running Pipeline Inference

Use `mlx.launch` to run pipeline inference across multiple hosts:

```bash
# Llama model
mlx.launch \
 --hostfile /path/to/hosts.json \
 /path/to/llama_pipeline.py \
 --model meta-llama/Meta-Llama-3-8B-Instruct \
 --prompt "Write a quicksort in Python."

# DeepSeek model  
mlx.launch \
 --hostfile /path/to/hosts.json \
 /path/to/pipeline_generate.py \
 --model mlx-community/DeepSeek-R1-3bit \
 --prompt "Write a quicksort in C++."
```

### Example Pipeline Script Structure

Both example scripts (`llama_pipeline.py` and `pipeline_generate.py`) follow this pattern:

1. **Download metadata**: Download config files, tokenizer, but not weights
2. **Initialize distributed group**: `group = mx.distributed.init()`
3. **Shard model**: Call `model.model.pipeline(group)` to set up layer distribution
4. **Determine local weights**: Use weight index to find which weight files are needed locally
5. **Download and load weights**: Download only the weights needed for local layers
6. **Generate**: Use `stream_generate()` normally - the pipelining is transparent

## Adding Pipeline Support to New Models

To add pipeline parallelism support to a new model, follow these steps:

### Step 1: Implement the Pipeline Method

Add a `pipeline(self, group)` method to your model's main class (the one containing transformer layers):

```python
def pipeline(self, group):
    # Split layers in reverse so rank=0 gets the last layers and
    # rank=pipeline_size-1 gets the first
    self.pipeline_rank = group.rank()
    self.pipeline_size = group.size()
    layers_per_rank = len(self.layers) // self.pipeline_size
    extra = len(self.layers) - layers_per_rank * self.pipeline_size
    if self.pipeline_rank < extra:
        layers_per_rank += 1

    self.start_idx = (self.pipeline_size - self.pipeline_rank - 1) * layers_per_rank
    self.end_idx = self.start_idx + layers_per_rank
    self.layers = self.layers[: self.end_idx]
    self.layers[: self.start_idx] = [None] * self.start_idx
    self.num_layers = len(self.layers) - self.start_idx
```

### Step 2: Update the Forward Pass

Modify your model's `__call__` method to handle pipeline communication:

```python
def __call__(self, inputs, mask=None, cache=None):
    h = self.embed_tokens(inputs)
    
    # Add pipeline attributes check
    pipeline_rank = getattr(self, 'pipeline_rank', 0)
    pipeline_size = getattr(self, 'pipeline_size', 1)
    
    # Handle mask and cache
    if mask is None:
        mask = create_attention_mask(h, cache)
    if cache is None:
        cache = [None] * self.num_layers

    # Pipeline communication - receive
    if pipeline_rank < pipeline_size - 1:
        h = mx.distributed.recv_like(h, (pipeline_rank + 1))

    # Process local layers
    for i in range(self.num_layers):
        h = self.layers[self.start_idx + i](h, mask, cache[i])

    # Pipeline communication - send
    if pipeline_rank != 0:
        h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)

    # Broadcast final result
    h = mx.distributed.all_gather(h)[: h.shape[0]]

    return self.norm(h)
```

### Step 3: Add Properties for Layer Access

Ensure your model has a `layers` property that can be accessed for pipeline setup:

```python
@property
def layers(self):
    return self.model.layers[self.model.start_idx : self.model.end_idx]
```

### Step 4: Test with Example Scripts

Create a test script based on the existing pipeline examples, or modify an existing one to use your new model.

## Technical Considerations

### Layer Distribution Strategy

- **Reverse distribution**: The last layers are placed on rank 0, first layers on the highest rank
- **Rationale**: This ensures rank 0 (which typically handles I/O) gets the final layers and can output results
- **Load balancing**: Extra layers are distributed to lower-ranked processes first

### Memory Optimization

The pipeline examples implement selective weight downloading:

1. Only download metadata initially
2. Determine which layers are local after pipeline setup
3. Download only the weight files needed for local layers
4. This minimizes memory usage and download time

### Communication Patterns

- **Point-to-point**: `recv_like` and `send` for passing hidden states between adjacent ranks
- **Collective**: `all_gather` to broadcast final results to all ranks
- **Synchronization**: `all_sum` used for process synchronization before generation

### Caching Considerations

- Cache arrays are still sized for the full model (`[None] * self.num_layers`)
- Only cache entries for local layers are actually used
- This maintains compatibility with generation functions

## Performance Notes

- Pipeline parallelism provides memory scaling but may not improve throughput for small batch sizes
- Communication overhead increases with more pipeline stages
- Best suited for large models that don't fit on a single device
- Can be combined with other parallelization strategies (data parallel, tensor parallel)

## Requirements

- MLX with distributed support
- MPI setup across multiple hosts
- Network connectivity between hosts for MPI communication
- Model weights available via Hugging Face or locally

For more information on MLX distributed computing, see: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
