# Text Preprocessing

This repo contains a framework for preprocessing text data, including synthetic text generation using LLMs.
The way it achieves this is using a workflow-like approach, where each step in the workflow can be configured using predefined nodes.

In general, the workflows are configured using json files which is explained below.

## Workflow

A workflow defines the control flow of the nodes.
A workflow is structured like a DAG and therefore does not allow for cycles.

A workflow is defined as a json file under the `workflows` folder and has the following structure:

```json
{
  "name": "",
  "data_dir" : "",
  "output_dir": "",
  "prompts_dir": "",
  "workflow_settings": {
    "optimize_for_engines": true
  },
  "nodes": [ ]
}
```

- `data_dir` - str: absolute path to the base data directory for this workflow.
- `output_dir` - str: absolute path to the base output directory
- `prompts_dir` - str: absolute path to the directory containing prompt templates
- `workflow_settings` - Dict | Optional: Global workflow configuration options.
  - `optimize_for_engines` - bool | Optional: Whether to optimize node execution order to minimize model loading/unloading. Defaults to `true`. Groups nodes with identical model configurations together to improve performance through engine sharing.

## Nodes

Nodes are the elementary building blocks of the framework.
Nodes are functions that can be chained but where the boilerplate code is hidden.
There are various types of nodes, each having their own inputs and outputs, and processing logic.
As it currently stands, each node reads from a json file (except for the data loading node) and writes to a json file.

The main node types are:

1. Load Node (`load_node`)
2. Text Prompt Node (`text_prompt_node`)
3. Combine Intermediate Outputs Node (`combine_intermediate_outputs`)

Each node comes with a base set of parameters, common to all nodes:

```json
    {
      "id": "",
      "type": "",
      "params": {
        "output_file_name": "<>.jsonl"
      },
      "dependencies": []
    },
```

- `id` - str: The id of the node. This is used to identify the node in the workflow, and will determine the name of the output file, except for the data loading node.
- `type` - str (enum, see below): The type of the node. This is used to determine which node to use in the workflow. These can only be one of the predefined types.
- `params` - Dict: The parameters of the node. This is a dictionary of parameters that are specific to the node type.
  - `output_file_name` - str | Optional: The name of the output file. This is optional, and if not provided the file name will be equals to the node id (.jsonl). Will be stored in the workflow output directory.
- `dependencies` - List\[id\]: The dependencies of the node. This is a list of node ids that this node depends on. This is used to determine the order in which the nodes should be executed. The dependencies are not used for the data loading node, as it is the first node in the workflow.

### Data Loading Node

The most basic node is the data loading node.
The main purpose of this node is to load data from a basic file, and transform it into a json format, which is the standard format for all nodes.

The json format is as follows:

```json
    {
      "id": "",
      "type": "load_node", 
      "params": {
        "name": "",
        "input_data_path": "",
        "primary_key": "",
        "data_attributes": [
          "",
          ""
        ]
      },
      "dependencies": []
    }
```

- `input_data_path` - str: The path to the input data file relative to the workflow data_dir. This is the file that will be loaded by the node.
- `data_attributes` - List\[str\] | Optional: The attributes of the data that will be loaded. This is optional and if not provided, all attributes will be loaded.
- `primary_key` - str : The column name that represents the primary identifier of the dataset.

### Text Prompt Node

The text prompt node is used to generate text using LLMs.
The node takes a single text attribute from an input or dependency output file.

The prompts for the LLM are located in the `prompts` folder, with each directory containing a different task that the user can define themselves.

Here, there should be a `system_prompt.txt` file, which contains the system prompt for the LLM, and a `user_prompt.txt` file, which contains the user prompt for the LLM.
Both files can be Jinja2 templates, which means that they can contain placeholders for the input data.
For more information on Jinja2 templates, see the [Jinja2 documentation](https://jinja.palletsprojects.com/en/3.0.x/).

A json file named `few-shot.jsonl` can be added to the directory, which contains a list of few-shot examples that will be used to generate the text. The few-shot examples are used to provide context to the LLM and are used in the same way as the user prompt. The few-shot examples are optional and if not provided, no few-shot examples will be used. The basic format of the json objects in this file is:

```json
{
  "context": {
    "variable1_in_template": "value for this example's variable1",
    "another_variable_in_template": "another value for this example"
  },
  "assistant": "The LLM's ideal response for this few-shot example's context."
}
```

In short a task for the LLM is defined using the following structure:

```txt
Polysome/
├─ prompts/
│  ├─ <your_task_name>/         (Corresponds to the 'name' parameter in the node config)
│  │  ├─ system_prompt.txt     (Or filename specified by 'system_prompt_file')
│  │  ├─ user_prompt.txt       (Jinja2 template, or filename specified by 'user_prompt_file')
│  │  ├─ few_shot.jsonl      (Or filename specified by 'few_shot_lines_file')
```

For an easy stars, please see the [Documentation](../docs/prompt_editor.md) about the [Prompt Editor](../prompt_editor.py).
The node comes with some options for customizing the LLM set-up:

```json

{
      "id": "llm_processing_step",
      "type": "text_prompt_node",
      "params": {
        "name": "my_summarization_task",
        "template_context_map": {
          "template_patient_id": "input_study_id",
          "full_text_content": "input_report_text"
        },
        "system_prompt_file": "system_instructions.txt",
        "user_prompt_file": "user_query_template.j2",
        "few_shot_lines_file": "examples.jsonl",
        "num_few_shots": 3,
        "model_name": "google/gemma-1.1-7b-it",
        "inference_engine": "huggingface",
        "engine_options": {
          "torch_dtype": "bfloat16",
          "device_map": "auto"
        },
        "generation_options": {
          "temperature": 0.7,
          "top_p": 0.9,
          "max_new_tokens": 1024
        },
        "output_data_attribute": "llm_summary",
        "resume": true,
        "parse_json": false
      },
      "dependencies": [
        "previous_data_loading_node_id"
      ]
    }```

- `name` - str: The name of the node. This should correspond to the name of the folder in the `prompts` directory. It will also determine the output file name.
- `template_context_map` - Dict | Optional: A dictionary to map keys from the input data (e.g., Excel column headers or keys from a preceding node's JSONL output) to the variable names expected by your Jinja2 templates.
    - Example: {"template_var_name": "input_data_key_name"}. If omitted, input data keys are used directly as variable names in templates.
- `system_prompt_file` - str | Optional: The filename of the system prompt file within the task's prompt directory. Defaults to "system_prompt.txt".
- `user_prompt_file` - str | Optional: The filename of the user prompt Jinja2 template file within the task's prompt directory. Defaults to `"user_prompt.txt"`. Consider using a `.j2` extension for clarity (e.g., `"user_template.j2"`).
- `few_shot_lines_file` - str | Optional: The filename of the JSONL file containing few-shot examples, located within the task's prompt directory. Defaults to `"few_shot.jsonl"`.
- `num_few_shots` - int | Optional: The number of few-shot examples to use. This is optional and if not provided, no few-shot examples will be used.
- `model_name` - str: The name of the model to use. This should be the name of the model in the Hugging Face model hub.
- `inference_engine` - str (enum: "huggingface", "llama_cpp", "vllm"): The type of inference engine to use. Currently there are three options: the standard Huggingface transformers ("huggingface") backend, the llama.cpp ("llama_cpp") inference engine, or the vLLM ("vllm") engine for optimized inference.
- `engine_options` - Dict | Optional: The options for the inference engine. This is optional and if not provided, the default options will be used.
  - For a full list of options, see the [Huggingface Transformers documentation](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained), [VLLM documentation (LLM class)](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html), or [llama-cpp documentation (Llama)](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/)
- `generation_options` - Dict | Optional: The options for the generation.
  - The options depend on the inference engine and can be found in the [Huggingface Transformers documentation](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.generate), [VLLM documentation (LLM class)](https://docs.vllm.ai/en/latest/api/offline_inference/llm.html#vllm.LLM.chat), or [llama-cpp documentation (Llama)](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama.create_chat_completion)
- `resume` - bool | Optional: Whether to resume from a previous workflow run for this node. It will read the output file (if it exists) and determines if it should resume based on the primary keys existing in thi file.
- `parse_json` - bool | Optional: Whether to parse the output of the LLM as json. This is optional and if not provided, the output will be returned as a string. If set to true, the output will be parsed as json and stored in the output json file as a json object.
- `batch_size` - int | Optional: The number of items to process in a single batch. Defaults to `1`. When greater than 1, enables batch processing for improved performance. Note: llama_cpp backend does not support batch inference and will fall back to sequential processing.
- `use_shared_engines` - bool | Optional: Whether to use shared engine instances across nodes. Defaults to `true` for optimal performance. When enabled, nodes with identical model configurations share the same loaded model instance, reducing memory usage and loading time. Set to `false` only if nodes require isolated model state.

### Combine Intermediate Outputs Node

The combine intermediate outputs node merges outputs from multiple dependency nodes into a single dataset.

```json
{
  "id": "combine_results",
  "type": "combine_intermediate_outputs",
  "params": {
    "name": "combine_outputs",
    "join_strategy": "inner",
    "column_mapping": {
      "node1_id": "summary_1",
      "node2_id": "summary_2"
    },
    "handle_conflicts": "prefix_source",
    "retain_original_attributes": true,
    "additional_output_formats": ["excel", "json"],
    "output_format_options": {
      "excel": {
        "sheet_name": "Combined_Results",
        "index": false
      },
      "json": {
        "indent": 2,
        "orient": "records"
      }
    }
  },
  "dependencies": [
    "dependency_node1",
    "dependency_node2"
  ]
}
```

- `name` - str: The name of the combine operation.
- `join_strategy` - str (enum: "inner", "left", "outer") | Optional: The type of join to perform. Defaults to "inner".
- `column_mapping` - Dict | Optional: Maps dependency node output attributes to new column names. Format: `{"dependency_node_id": "new_column_name"}`.
- `handle_conflicts` - str (enum: "prefix_source", "keep_first", "keep_last") | Optional: How to handle column name conflicts. Defaults to "prefix_source".
- `retain_original_attributes` - bool | Optional: Whether to include original attributes from source data. Defaults to false.
- `additional_output_formats` - List[str] | Optional: Additional output formats to generate. Supported: `["excel", "json", "parquet"]`.
- `output_format_options` - Dict | Optional: Format-specific options for additional output formats.

## Example Workflow Snippet

The following JSON snippet shows a Data Loading Node followed by a Text Prompt Node configured to use Jinja2 templating.

```json
{
  "name": "clinical_summary_workflow",
  "data_dir": "/path/to/workflow_data/",
  "output_dir": "/path/to/workflow_output/",
  "nodes": [
    {
      "id": "step0_excel_loader",
      "type": "load_node",
      "params": {
        "name": "load_patient_records",
        "input_data_path": "patient_data.xlsx",
        "primary_key": "RecordID",
        "data_attributes": [
          "RecordID",
          "PatientAge",
          "PatientGender",
          "ChiefComplaintText",
          "MedicalHistorySummary"
        ]
      },
      "dependencies": []
    },
    {
      "id": "step1_generate_summaries", 
      "type": "text_prompt_node",
      "params": {
        "name": "clinical_summarizer_task",
        "template_context_map": {
          "patient_id": "RecordID",
          "age": "PatientAge",
          "gender": "PatientGender",
          "complaint": "ChiefComplaintText",
          "history": "MedicalHistorySummary"
        },
        "user_prompt_file": "summary_user_template.j2",
        "num_few_shots": 5,
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "inference_engine": "huggingface",
        "engine_options": {
          "torch_dtype": "bfloat16",
          "device_map": "auto"
        },
        "generation_options": {
          "temperature": 0.2,
          "max_new_tokens": 512
        },
        "output_data_attribute": "generated_summary",
        "resume": false,
        "parse_json": false,
        "use_shared_engines": true
      },
      "dependencies": [
        "step0_excel_loader"
      ]
    }
  ]
}```
