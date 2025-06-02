# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
# Run all tests
python -m unittest discover tests/

# Run a specific test file
python -m unittest tests/test_models.py

# Run a specific test class
python -m unittest tests.test_models.TestModels

# Run a specific test method
python -m unittest tests.test_models.TestModels.test_llama
```

### Code Formatting and Linting
```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Run formatters on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files file1.py file2.py

# Manual formatting
black file.py  # Python formatting
```

### Installation
```bash
# Install for development
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with evaluation dependencies
pip install -e ".[evaluate]"
```

## Architecture Overview

### Core Module Structure

**mlx_lm/** - Main package directory
- **models/** - Model implementations for different architectures
  - Each model file (e.g., `llama.py`, `mistral.py`) contains:
    - `ModelArgs` dataclass for configuration
    - Core components: `Attention`, `MLP`, `TransformerBlock`
    - Main model class inheriting from `nn.Module`
  - `base.py` - Base classes and utilities shared across models
  - Models are dynamically loaded based on `model_type` in config.json

- **tuner/** - Fine-tuning functionality
  - `lora.py` - LoRA implementation
  - `trainer.py` - Training logic
  - `datasets.py` - Dataset loading and processing
  - `utils.py` - LoRA layer mappings for each model type

- **Main API Functions:**
  - `load()` - Load models and tokenizers from local path or Hugging Face
  - `generate()` / `stream_generate()` - Text generation
  - `convert()` - Model quantization and format conversion

### Model Loading Flow

1. Models are loaded via `utils.py:load()` which:
   - Downloads from Hugging Face if needed
   - Reads `config.json` to determine model type
   - Dynamically imports the appropriate model class from `models/`
   - Loads weights from safetensors format
   - Applies quantization if specified

2. Model type mapping in `utils.py:MODEL_REMAPPING` handles aliases (e.g., mistral → llama)

3. Each model must implement the standard interface expected by generation functions

### Key Design Patterns

- **Quantization**: Supports 4-bit and 8-bit quantization via AWQ and DWQ methods
- **Caching**: Implements KV-cache for efficient generation, including quantized cache
- **Adapters**: LoRA adapters can be loaded on top of base or quantized models
- **Distributed**: Uses `mx.distributed` for multi-GPU inference and training

### Adding New Models

1. Create a new file in `mlx_lm/models/` matching the `model_type` from config.json
2. Implement required classes: `ModelArgs`, model components, and main `Model` class
3. Add LoRA mappings in `mlx_lm/tuner/utils.py`
4. Add tests in `tests/test_models.py`

### Important Files for Context

- `mlx_lm/utils.py` - Model loading and management utilities
- `mlx_lm/generate.py` - Core generation logic
- `mlx_lm/models/base.py` - Shared model utilities
- `mlx_lm/sample_utils.py` - Sampling strategies for generation