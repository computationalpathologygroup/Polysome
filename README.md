# Polysome

Polysome is a **generic data generation framework** designed for transforming text attributes using Large Language Models. It serves as a powerful workflow builder for chaining prompts to generate synthetic data at scale.

While originally developed for computational pathology, Polysome is domain-agnostic and focuses purely on the **data generation** aspect of the pipeline. It allows you to define complex, multi-step text transformation workflows using a node-based architecture.

**Key Features:**

* **Workflow Engine**: Define DAG-based pipelines to load data, process it with LLMs, and structure the output.
* **Synthetic Data Generation**: Ideal for creating instruction tuning datasets, rewriting reports, or extracting structured information from unstructured text.
* **Prompt Chaining**: distinct nodes for complex reasoning tasks, summarization, and translation.
* **High Performance**: Supports batch processing and Data Parallelism via vLLM.

## 🚀 Quick Start

### Installation

Install Polysome from PyPI (minimal install, HuggingFace only):

```bash
pip install polysome
```

**For specific engines:**

```bash
# vLLM (Recommended for Linux + NVIDIA GPU)
pip install "polysome[vllm]"

# llama.cpp (Recommended for CPU or Apple Silicon)
pip install "polysome[llama-cpp]"

# Install everything (for development/testing)
pip install "polysome[all]"
```

**Convenience aliases:**
* `pip install "polysome[gpu]"` → installs `vllm` stack
* `pip install "polysome[cpu]"` → installs `llama-cpp` stack

### Create Your First Project

Initialize a new project with example workflows and prompts:

```bash
polysome init my-analysis
cd my-analysis
```

This creates:
* `workflows/` - Example workflow configurations
* `prompts/` - Prompt templates for your tasks
* `data/` - Directory for input data (with example file)
* `output/` - Results will be saved here

### Run a Workflow

```bash
# Run the example workflow
polysome run workflows/basic_text_generation.json

# Run with custom settings
polysome run workflows/my_workflow.json --log-level DEBUG
```

### Customize for Your Use Case

1. **Edit your workflow** (`workflows/basic_text_generation.json`):
   * Change the model name
   * Adjust generation parameters
   * Add or remove processing nodes

2. **Customize prompts** (`prompts/simple_qa/`):
   * `system_prompt.txt`: Define the AI's role
   * `user_prompt.txt`: Template with variables like `{question}`
   * `few_shot.jsonl`: Example inputs and outputs

3. **Prepare your data** (`data/input.json`):
   * Format as JSON with fields matching your prompt variables

4. **Run and iterate**:

   ```bash
   polysome run workflows/basic_text_generation.json
   ```

### Using Docker (Alternative)

For reproducible environments or deployment:

```bash
# Pull the image
docker pull ghcr.io/computationalpathologygroup/polysome:latest

# Run with Docker
docker run --rm --gpus all \
  -v ./data:/data \
  -v ./output:/output \
  -v ./workflows:/workflows \
  -v ./prompts:/prompts \
  -e WORKFLOW_PATH=/workflows/basic_text_generation.json \
  ghcr.io/computationalpathologygroup/polysome:latest
```

For detailed Docker usage, see [docs/docker_container.md](docs/docker_container.md).

## 🐍 Programmatic API

For advanced users, you can also use Polysome programmatically:

```python
from polysome.workflow import Workflow

# Load and run a workflow
workflow = Workflow("workflows/my_workflow.json")
success = workflow.run()

# Access results
print(f"Results saved to: {workflow.output_dir}")
```

## 🛠️ Contributing & Development

Want to contribute or modify Polysome? See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Local Development Setup

```bash
git clone https://github.com/computationalpathologygroup/Polysome.git
cd Polysome

# Install in development mode with dependencies
pip install -e .[gpu-dev]  # For GPU development
# OR
pip install -e .[cpu-dev]  # For CPU development
```

### Running Tests

```bash
pytest
pytest --cov=polysome  # With coverage
```

## 🧩 Workflow Configuration

Workflows are defined in JSON files (DAGs) located in the `workflows/` directory. They control how data is loaded, processed by LLMs, and saved.

For a detailed guide on creating nodes and configuring JSONs, see **[docs/text_preprocessing.md](docs/text_preprocessing.md)**.

## 🎨 Prompt Engineering

Polysome includes a **Streamlit-based Prompt Editor** to help you design, manage, and test Jinja2 templates for your LLM tasks.

```bash
# Run the editor
streamlit run prompt_editor.py
```

For a user guide on managing templates and few-shot examples, see **[docs/prompt_editor.md](docs/prompt_editor.md)**.

## ⚡ High Performance Inference

For large-scale processing, Polysome supports **Data Parallelism** using vLLM to distribute batches across multiple GPUs.

To enable this, use the `vllm_dp` engine in your workflow configuration. See **[docs/data_parallelism.md](docs/data_parallelism.md)** for setup instructions and performance tuning.

## 📚 Documentation Index

* [Text Preprocessing & Workflows](docs/text_preprocessing.md)
* [Docker Container Guide](docs/docker_container.md)
* [Prompt Editor Guide](docs/prompt_editor.md)
* [Data Parallelism Guide](docs/data_parallelism.md)

## 📄 Citation

This framework was originally developed to support visual instruction tuning. If you use this code to generate data for such models, please consider citing the following paper:

```bibtex
@inproceedings{moonemans2025open,
  title={Democratizing Pathology Co-Pilots: An Open Pipeline and Dataset for Whole-Slide Vision-Language Modeling},
  author={Sander Moonemans and Sebastiaan Ram and Fr{\'e}d{\'e}rique Meeuwsen and Carlijn Lems and Jeroen van der Laak and Geert Litjens and Francesco Ciompi},
  booktitle={Submitted to Medical Imaging with Deep Learning},
  year={2025},
  url={https://openreview.net/forum?id=aGPowreqPi},
  note={under review}
}
```
