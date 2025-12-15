# Docker Container Guide

Generic workflow runner for Polysome. Two variants available:

- **GPU-enabled (x86_64)**: CUDA support for NVIDIA GPUs
- **CPU-only (ARM64)**: For ARM64 systems

## Build

**GPU container:**

```bash
docker build -t polysome-runner .
```

**ARM64 container:**

```bash
docker build -f Dockerfile.arm64 -t polysome-runner-arm64 .
```

## Run

**GPU container:**

```bash
docker run --rm --gpus all \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  -v $(pwd)/workflows:/workflows \
  -v $(pwd)/prompts:/prompts \
  -e WORKFLOW_PATH=/workflows/my_workflow.json \
  polysome-runner
```

**ARM64 container:**

```bash
docker run --rm \
  -v $(pwd)/models:/models \
  -v $(pwd)/data:/data \
  -v $(pwd)/output:/output \
  -v $(pwd)/workflows:/workflows \
  -v $(pwd)/prompts:/prompts \
  -e WORKFLOW_PATH=/workflows/my_workflow.json \
  polysome-runner-arm64
```

## Mount Points

| Path | Purpose |
|------|---------|
| `/models/` | Model files (e.g. .gguf) |
| `/data/` | Input data |
| `/output/` | Results |
| `/workflows/` | Workflow configs |
| `/prompts/` | Prompt templates |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKFLOW_PATH` | Workflow JSON path | `/workflows/default.json` |

## Container Directory Structure

```
/app/                          # Application code
├── inference.py               # Main entrypoint
├── src/                       # Polysome source
└── pyproject.toml            # Dependencies

/models/                       # Model files (mounted)
├── llava-med-v1.5-mistral-7b.gguf
└── ...

/data/                         # Input data (mounted)
├── test_case.json
└── ...

/output/                       # Results (mounted)
├── workflow_name/
│   ├── node_outputs/
│   └── final_results.json
└── ...

/workflows/                    # Workflow configs (mounted)
├── simple_translation_test.json
├── my_workflow.json
└── ...

/prompts/                      # Prompt templates (mounted)
├── task_translation_dutch/
│   ├── system_prompt.txt
│   ├── user_prompt.txt
│   └── few_shot.jsonl
├── task_histai_detailed_description/
│   ├── system_prompt.txt
│   ├── user_prompt.txt
│   └── few_shot.jsonl
└── ...
```

## Simplified Workflow Configuration

Reference container paths in your workflow JSON:

```json
{
  "name": "example_workflow",
  "data_dir": "/data",
  "output_dir": "/output",
  "nodes": [
    {
      "id": "load_data",
      "type": "load",
      "params": {
        "input_data_path": "/data/case_data.json"
      }
    },
    {
      "id": "generate_text",
      "type": "text_prompt",
      "params": {
        "name": "task_translation_dutch",
        "model_name": "/models/model.gguf"
      },
      "dependencies": ["load_data"]
    }
  ]
}
```
