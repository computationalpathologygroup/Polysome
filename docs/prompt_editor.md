# Jinja Prompt & Example Manager: User Guide

## Overview

The Jinja Prompt & Example Manager is a Streamlit application designed to help you create, manage, and test prompts for Large Language Models (LLMs). It's particularly useful when working with a framework that uses Jinja2 templating for user prompts and supports few-shot examples. The app allows you to:

* Organize prompts into "tasks."
* Edit system prompts, user prompt Jinja2 templates, and few-shot examples.
* Define context variables for your templates and examples.
* Inspect Excel data and use its columns to guide context variable creation or pre-fill examples.
* Test your prompt configurations with various LLMs and inference engines using sample data from an Excel file.
* View detailed test results, including the fully formatted prompt sent to the LLM and its output.
* Generate a copyable JSON configuration snippet for a "Text Prompt Node" based on your test settings, ready for integration into a workflow.

## Getting Started

### Prerequisites

1.  **Python Environment:** Ensure you have Python 3.10+ installed.
2.  **Installation:** Install Polysome with the `ui` extra to include Streamlit and other necessary libraries:
    ```bash
    pip install "polysome[ui]"
    ```

### Launching the App

Once installed, you can launch the Prompt Editor from any directory using the built-in CLI command:

```bash
polysome-gui
```

This will automatically find the editor script within the package and open the application in your web browser.

## Main Interface

The application is divided into two main sections:

* **Sidebar (Left):** Contains controls for Task Management, the Excel Data Inspector, and Task Testing.
* **Main Area (Right):** Displays content for editing the selected task, managing few-shot examples, and viewing test results.

## Task Management (Sidebar)

At the top of the sidebar, you'll find the task management section.

### Creating a New Task

1.  From the "Select Task" dropdown, ensure no task is selected (it should show "✨ Create New Task ✨").
2.  A "New Task Name" input field will appear. Enter a descriptive name for your new task (e.g., `clinical_summarizer`).
3.  Click the "Create Task" button.
4.  The new task will be created with default system and user prompts, and it will become the currently selected task.

### Selecting an Existing Task

1.  Click the "Select Task" dropdown.
2.  Choose from the list of existing tasks found in your `prompts` directory.
3.  The main area will update to show the selected task's configuration.

### Duplicating a Task

1.  Select the task you wish to duplicate from the "Select Task" dropdown.
2.  Under "Actions for: [Your Task Name]", click the "Duplicate Task" button.
3.  A copy of the task will be created (e.g., `[Your Task Name]_copy`) and automatically selected.

### Deleting a Task

1.  Select the task you wish to delete.
2.  Click the "🗑️ Delete Task" button.
3.  Confirm the action. The task directory and its contents will be removed from the `prompts` folder.

## Editing a Task (Main Area)

When a task is selected or newly created, its details are displayed in the main area for editing.

### System Prompt

* The "System Prompt Content" text area allows you to define the instructions or role for the LLM.

### User Prompt Jinja2 Template

* The "User Prompt Template Content" text area is where you define your Jinja2 template.
* Use Jinja2 syntax (e.g., `{{ variable_name }}`) to include dynamic content.
* Variables used here should correspond to the "Few-shot Context Dictionary Keys" you define.

### Few-shot Context Dictionary Keys

These are the variable names that your User Prompt Template will expect and that will structure the `context` part of your few-shot examples.

* **Adding a New Key:**
    1.  Enter a valid key name (letters, numbers, underscores) in the "New Context Key Name" input field.
    2.  Click "Add Context Key." The key will be added to the list and to the context of all existing few-shot examples (with an empty string value).
* **Using Excel Columns as Keys (from Sidebar):**
    1.  Upload an Excel file in the "Excel Data Inspector" (see below).
    2.  Click "Use Excel Columns as Context Keys." This will add all column headers from the Excel file to your current task's context keys.
* **Removing Keys:**
    * Click "Remove All Context Keys" to clear all defined keys and their values from all few-shot examples.

### Few-shot Examples

Few-shot examples provide the LLM with demonstrations of the desired input/output behavior.

* **Adding a New Example:**
    1.  If you have defined "Context Dictionary Keys," corresponding text areas will appear under "Add New Few-shot Example" for each key.
    2.  Fill in the context values for your example.
    3.  Fill in the desired "Assistant Response."
    4.  **(Optional) Pre-fill from Excel:**
        * If an Excel file is loaded, you can use the "Pre-fill from Excel Row" expander.
        * Enter the Excel row index (0-based).
        * Click "Load Data from Excel Row." The context fields will be populated with data from the specified row, matching context key names to Excel column headers. Review and adjust as needed.
    5.  Click the "➕ Add Example" button within the form. The example will be added to the list.
* **Editing an Existing Example:**
    1.  Under "Current Few-shot Examples," find the example you want to edit.
    2.  Click the "✏️ Edit" button for that example.
    3.  The example's details will load into the editing form above the list.
    4.  Make your changes and click "✅ Update Example." To cancel, click "❌ Cancel Editing."
* **Deleting an Example:**
    1.  Click the "🗑️ Delete" button next to the example you wish to remove.

### Saving the Task

* After making any changes to the system prompt, user prompt template, or few-shot examples, click the "💾 Save Task" button at the bottom of the main area.
* This saves the current configuration to the respective files (`system_prompt.txt`, `user_prompt.txt`, `few_shot.jsonl`) in the task's directory under `prompts/`.

## Excel Data Inspector (Sidebar)

This section helps you inspect sample data, which is crucial for testing your prompts.

### Uploading an Excel File

1.  Click "Browse files" under "Upload Excel File (for reference)" or drag and drop an `.xlsx` or `.xls` file.
2.  Once loaded, the file name and a success message will appear.

### Viewing Available Columns

* The column headers from the uploaded Excel file will be listed.
* You can click "Use Excel Columns as Context Keys" to add these column names as context keys for the currently selected task.

### Data Preview Controls

* **Number of rows for preview:** Set how many rows to display in the preview.
* **📊 Show Random Sample:** Displays a random sample of N rows from your Excel data.
* **📋 Show First N Rows:** Displays the first N rows of your Excel data (default view).
* A preview of the data (either random sample or first N rows) will be shown directly in the sidebar.

## Testing a Task (Sidebar)

Once a task is defined and an Excel file is uploaded, you can test your prompt configuration.

### Prerequisites for Testing

* A task must be selected.
* An Excel file must be uploaded via the "Excel Data Inspector."

### Engine Configuration Preset

* **Preset Selection:** Choose a pre-defined engine configuration from the "Engine Configuration Preset" dropdown (e.g., "Hugging Face (gemma-3-4b-it)", "Llama.cpp (...)"). This will populate the model name, inference engine, and options fields.
* **"Custom":** If you modify any of the preset fields, the preset will automatically switch to "Custom."

### Test Configuration Fields

* **Inference Engine:** Select the backend for running the LLM (e.g., `huggingface`, `llama_cpp`).
* **Model for Test Run:** Specify the Hugging Face model ID or the local path to your model file (e.g., a GGUF file for Llama.cpp).
* **Primary Key Column for Test Data:** Select the column from your uploaded Excel file that serves as the unique identifier for each row. This is important for tracking results.
* **Number of sample rows for test:** Specify how many rows from the *top* of your Excel file should be used for this test run.
* **Engine Options (JSON string):** Provide a JSON string for options specific to the chosen inference engine and model loading (e.g., `{"torch_dtype": "bfloat16", "device_map": "auto"}`).
* **Generation Options (JSON string):** Provide a JSON string for options controlling the LLM's text generation (e.g., `{"max_new_tokens": 256, "temperature": 0.7}`).

### Running the Test

* Click the "🚀 Run Test with Sample Data" button.
* A spinner will indicate that the test is in progress.

## Test Run Results (Main Area)

After a test run is complete, the results will be displayed in the main area.

### Viewing Results per Item

For each row processed from your sample data, an expander will show:

* **Input Data:** The raw data from the Excel row used for this test item.
* **Formatted Prompt (JSON for LLM):** The actual prompt (often a list of messages like system, user, few-shot assistant) that was constructed and sent to the LLM. This is crucial for debugging your Jinja2 template and few-shot formatting.
* **LLM Output:**
    * If the LLM produced text or structured JSON, it will be displayed here.
    * If an error occurred during this item's processing, the error message will be shown.
    * If no output was generated (but no error occurred), this will be indicated.

### General Test Run Errors

* Any errors that occurred during the test run setup or that were not specific to a single item will be displayed at the top of the results section.

### Clearing Test Results

* Click the "Clear Test Results and Cleanup Files" button to remove the displayed results and any temporary files created during the test run.

## Generating Node Configuration (Main Area)

After a successful test run, a new section titled "📋 Generated Text Prompt Node Configuration" will appear below the test results.

* This section displays a JSON object representing the configuration for a "Text Prompt Node" based on:
    * The current task name.
    * The model, inference engine, engine options, and generation options used in the test.
    * The number of few-shot examples currently defined for the task.
* You can directly copy this JSON snippet.
* **Note:** You may need to adjust fields like `"id"` and `"dependencies"` to fit your specific workflow structure. The `"template_context_map"` will be empty by default in the generated config, reflecting the test setup; if your input data keys differ from your template variable names, you'll need to populate this map.

## Directory Structure for Prompts

The application organizes tasks on the filesystem as follows:

```
prompts/
└── <your_task_name>/
    ├── system_prompt.txt      # Contains the system prompt content
    ├── user_prompt.txt        # Contains the Jinja2 user prompt template
    └── few_shot.jsonl         # Contains few-shot examples, one JSON object per line
```

This structure is automatically managed by the application when you create, save, or delete tasks.

---

Happy Prompting!
