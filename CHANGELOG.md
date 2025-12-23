# Changelog

All notable changes to Polysome will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-12-23

### Changed
- **Renamed Package**: Canonical name changed from `Polysome` to `polysome` (lowercase) for consistency with Python standards.
- **GUI Integration**: Prompt Editor is now integrated into the CLI via `polysome-gui`.
- **Dependency Structure**: Introduced `[ui]` extra for minimal GUI install and `[all]` for full engine support.
- **Default Model**: Updated example templates to use `google/gemma-3-4b-it`.

### Fixed
- **Jinja2 Syntax**: Corrected variable interpolation syntax in prompt templates.
- **Workflow Error Handling**: Improved `Workflow.run` to correctly report node failures.
- **CLI Scripting**: Fixed Bash arithmetic evaluation in integration test scripts.

## [0.1.0] - 2025-12-23

### Added
- **Initial Release**: Polysome v0.1.0 is now available!
- **Workflow Engine**: DAG-based pipeline orchestration for data processing and generation.
- **Inference Engines**: Support for multiple LLM backends:
    - **HuggingFace Transformers**: Standard backend for PyTorch models.
    - **vLLM**: High-performance inference engine with quantization support.
    - **llama.cpp**: Efficient CPU/Apple Silicon inference using GGUF models.
    - **vLLM Data Parallel**: Multi-GPU batch processing support.
- **Node Types**:
    - `load`: Flexible data loading from JSON/JSONL/CSV.
    - `text_prompt`: Core LLM processing node with Jinja2 templating and few-shot support.
    - `combine_intermediate_outputs`: Merge workflow branches.
    - Utility nodes: `regex_split`, `sentence_split`, `deduplication`, `row_concatenation`, `column_concatenation`.
- **CLI Tool**: `polysome` command-line interface for:
    - `init`: Scaffolding new projects with templates.
    - `run`: Executing workflows with validation.
- **Docker Support**: Ready-to-use Docker images for GPU (CUDA) and CPU (ARM64) environments.
- **Prompt Editor**: Streamlit-based UI for designing and testing prompts (`polysome-prompt-editor`).
- **Documentation**: Comprehensive guides for installation, workflows, prompt engineering, and docker usage.

[0.1.1]: https://github.com/computationalpathologygroup/Polysome/releases/tag/v0.1.1
[0.1.0]: https://github.com/computationalpathologygroup/Polysome/releases/tag/v0.1.0
