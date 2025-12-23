# Polysome Integration Tests

This directory contains integration tests for validating the Polysome framework across different inference engines (HuggingFace, vLLM, and llama.cpp).

## Test Structure

- `data/`: Contains test datasets (e.g., `test_questions.json`).
- `workflows/`: Engine-specific workflow configurations.
- `scripts/`: Test execution and validation scripts.
- `expected_outputs/`: Success criteria and expected results.

## Prerequisites

1. **GPU Availability**: Most tests require a CUDA-capable GPU.
2. **Model Weights**:
   - HuggingFace models are downloaded automatically.
   - vLLM models should be available on HuggingFace Hub.
   - **llama.cpp** requires a GGUF model file located at `/models/gemma-3-4b-it-qat-q4_0-gguf` or as specified in `workflows/test_llamacpp.json`.

## Running Tests

### 1. CLI Method
Run workflows using the `polysome` command line tool:
```bash
bash tests/integration/scripts/test_cli.sh
```

### 2. API Method
Run workflows programmatically using the Python API:
```bash
python3 tests/integration/scripts/test_api.py
```

### 3. Docker Method
Run workflows inside the standardized Docker container:
```bash
bash tests/integration/scripts/test_docker.sh
```

## Validation
After running any test method, you can validate the outputs using:
```bash
python3 tests/integration/scripts/validate_outputs.py
```

## Adding New Tests
1. Add input data to `data/`.
2. Create a new workflow in `workflows/`.
3. Update `expected_outputs/validation_criteria.json` with the new expected output paths.
