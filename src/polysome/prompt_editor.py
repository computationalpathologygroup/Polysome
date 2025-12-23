import streamlit as st
import os
import json
import shutil
import re
import pandas as pd
from typing import Dict, Any, List
from polysome.nodes.text_prompt_node import TextPromptNode
from polysome.utils.data_loader import DataFileLoader
import tempfile  # For temporary directories
from pathlib import Path
import logging  # For logging within the test function
from polysome.prompt_formatter import PromptFormatter

logger = logging.getLogger(__name__)

# --- Presets Definition ---
ENGINE_PRESETS = {
    "Hugging Face (gemma-3-4b-it)": {
        "model_name": "google/gemma-3-4b-it",
        "inference_engine": "huggingface",
        "engine_options": {
            "torch_dtype": "bfloat16",
            "device_map": "auto",
        },  # Added device_map as a common default
        "generation_options": {"max_new_tokens": 256},
    },
    "Llama.cpp (gemma-3-27b-it-q4_0)": {
        "model_name": os.path.join(os.getenv("MODEL_PATH", "./models"), "gemma-3-27b-it-q4_0.gguf"),
        "inference_engine": "llama_cpp",
        "engine_options": {
            "chat_format": "gemma",
            "n_gpu_layers": -1,
            "n_ctx": 4096,
            "verbose": False,
        },
        "generation_options": {"max_tokens": 150, "temperature": 0.7, "top_p": 0.9},
    },
    "Custom": {  # Allows users to start from scratch or modify
        "model_name": "",
        "inference_engine": "huggingface",  # Default engine for custom
        "engine_options": {},
        "generation_options": {},
    },
}
#

st.set_page_config(page_title="Jinja Prompt Manager", layout="wide")


def ensure_prompts_dir(prompts_dir):
    if not os.path.exists(prompts_dir):
        os.makedirs(prompts_dir)


def get_task_directories(prompts_dir):
    ensure_prompts_dir(prompts_dir)
    return [
        d
        for d in os.listdir(prompts_dir)
        if os.path.isdir(os.path.join(prompts_dir, d))
    ]


def save_task(task_name, system_prompt, user_prompt_template, few_shots, prompts_dir):
    task_dir = os.path.join(prompts_dir, task_name)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)

    with open(os.path.join(task_dir, "system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system_prompt)
    with open(os.path.join(task_dir, "user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(user_prompt_template)

    with open(os.path.join(task_dir, "few_shot.jsonl"), "w", encoding="utf-8") as f:
        for example in few_shots:
            if "context" not in example or not isinstance(example["context"], dict):
                example["context"] = {}
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def load_task(task_name, prompts_dir):
    task_dir = os.path.join(prompts_dir, task_name)
    system_prompt = ""
    system_prompt_path = os.path.join(task_dir, "system_prompt.txt")
    if os.path.exists(system_prompt_path):
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    user_prompt_template = ""
    user_prompt_template_path = os.path.join(task_dir, "user_prompt.txt")
    if os.path.exists(user_prompt_template_path):
        with open(user_prompt_template_path, "r", encoding="utf-8") as f:
            user_prompt_template = f.read()

    few_shots = []
    context_dict_keys = []
    few_shot_path = os.path.join(task_dir, "few_shot.jsonl")
    if os.path.exists(few_shot_path):
        with open(few_shot_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        example = json.loads(line)
                        if "context" not in example or not isinstance(
                            example["context"], dict
                        ):
                            st.warning(
                                f"Task '{task_name}', few_shot.jsonl line {line_num + 1}: 'context' field missing/invalid. Init as empty."
                            )
                            example["context"] = {}
                        if not context_dict_keys and example["context"]:
                            context_dict_keys = list(example["context"].keys())
                        few_shots.append(example)
                    except json.JSONDecodeError as e:
                        st.error(
                            f"Error decoding JSON in {few_shot_path} line {line_num + 1}: {e}"
                        )

    return system_prompt, user_prompt_template, few_shots, context_dict_keys


def duplicate_task(source_task, new_task_name, prompts_dir):
    source_dir = os.path.join(prompts_dir, source_task)
    target_dir = os.path.join(prompts_dir, new_task_name)
    if os.path.exists(target_dir):
        return False, f"Task '{new_task_name}' already exists"
    try:
        shutil.copytree(source_dir, target_dir)
        return True, f"Task '{source_task}' duplicated as '{new_task_name}'"
    except Exception as e:
        return False, f"Error duplicating task: {e}"


def delete_task(task_name, prompts_dir):
    task_dir = os.path.join(prompts_dir, task_name)
    if os.path.exists(task_dir):
        try:
            shutil.rmtree(task_dir)
            return True
        except Exception as e:
            st.error(f"Error deleting task directory: {e}")
            return False
    return False


def rename_task(
    old_task_name: str, new_task_name: str, prompts_dir: str
) -> tuple[bool, str]:
    """
    Renames a task directory.

    Args:
        old_task_name: The current name of the task.
        new_task_name: The desired new name for the task.
        prompts_dir: The base directory where task folders are stored.

    Returns:
        A tuple containing a boolean indicating success and a message string.
    """
    if not old_task_name:
        return False, "Original task name is missing or invalid."
    if not new_task_name or not new_task_name.strip():
        return False, "New task name cannot be empty or just whitespace."

    # Sanitize new_task_name by stripping leading/trailing whitespace
    new_task_name = new_task_name.strip()

    if old_task_name == new_task_name:
        return False, "New task name is the same as the old one. No change made."

    # Validate new task name for allowed characters (letters, numbers, underscore, hyphen, period, space)
    if not re.match(r"^[a-zA-Z0-9_.\- ]+$", new_task_name):
        return (
            False,
            "New task name contains invalid characters. Please use only letters, numbers, underscores, hyphens, periods, or spaces.",
        )

    old_task_dir = os.path.join(prompts_dir, old_task_name)
    new_task_dir = os.path.join(prompts_dir, new_task_name)

    if not os.path.exists(old_task_dir) or not os.path.isdir(old_task_dir):
        return False, f"Task '{old_task_name}' not found or is not a directory."

    if os.path.exists(new_task_dir):
        return (
            False,
            f"A task or directory with the name '{new_task_name}' already exists. Please choose a different name.",
        )

    try:
        os.rename(old_task_dir, new_task_dir)
        return (
            True,
            f"Task '{old_task_name}' successfully renamed to '{new_task_name}'.",
        )
    except OSError as e:
        # More specific error messages could be helpful here if needed
        return (
            False,
            f"Error renaming task directory: {e}. Check permissions or if the directory is in use.",
        )


# --- Callbacks for Prompt Text Areas ---
def system_prompt_changed():
    widget_key = f"sys_prompt_{st.session_state.current_task}"
    if widget_key in st.session_state:
        st.session_state.system_prompt = st.session_state[widget_key]


def user_prompt_template_changed():
    widget_key = f"usr_tmpl_{st.session_state.current_task}"
    if widget_key in st.session_state:
        st.session_state.user_prompt_template = st.session_state[widget_key]


def display_data_inspector():
    st.sidebar.subheader("Data Inspector")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Data File (for reference)",
        type=["xlsx", "xls", "json", "jsonl", "csv"],
        key="data_uploader",
    )

    # Initialize session state variables for data display if they don't exist
    if "data_df" not in st.session_state:
        st.session_state.data_df = None
    if "data_columns" not in st.session_state:
        st.session_state.data_columns = []
    if "last_uploaded_data_name" not in st.session_state:
        st.session_state.last_uploaded_data_name = None
    if "data_sample_n" not in st.session_state:
        st.session_state.data_sample_n = 5
    if "data_current_sample_df" not in st.session_state:
        st.session_state.data_current_sample_df = None
    if "data_file_type" not in st.session_state:
        st.session_state.data_file_type = None

    def load_json_file(file):
        """Load JSON file - handles both JSON array and JSONL formats, converting nested objects to strings"""

        def convert_nested_to_string(obj):
            """Recursively convert nested objects/arrays to JSON strings"""
            if isinstance(obj, dict):
                result = {}
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        # Convert nested structures to JSON strings
                        result[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        result[key] = value
                return result
            elif isinstance(obj, list):
                return [convert_nested_to_string(item) for item in obj]
            else:
                return obj

        try:
            # Read file content as text
            content = file.read().decode("utf-8")

            # Try to parse as regular JSON first
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    # Convert nested structures in each item
                    processed_data = [convert_nested_to_string(item) for item in data]
                    return pd.DataFrame(processed_data)
                elif isinstance(data, dict):
                    # If it's a single object, wrap it in a list and convert nested structures
                    processed_data = convert_nested_to_string(data)
                    return pd.DataFrame([processed_data])
                else:
                    raise ValueError(
                        "JSON must be an array of objects or a single object"
                    )
            except json.JSONDecodeError:
                # If regular JSON parsing fails, try JSONL format
                lines = content.strip().split("\n")
                json_objects = []
                for line_num, line in enumerate(lines, 1):
                    if line.strip():  # Skip empty lines
                        try:
                            parsed_line = json.loads(line)
                            processed_line = convert_nested_to_string(parsed_line)
                            json_objects.append(processed_line)
                        except json.JSONDecodeError as e:
                            raise ValueError(f"Invalid JSON on line {line_num}: {e}")

                if not json_objects:
                    raise ValueError("No valid JSON objects found in file")

                return pd.DataFrame(json_objects)

        except Exception as e:
            raise Exception(f"Error parsing JSON file: {str(e)}")

    if uploaded_file is not None:
        # Process only if it's a new file or a re-upload of a different file
        if st.session_state.last_uploaded_data_name != uploaded_file.name:
            try:
                file_extension = uploaded_file.name.lower().split(".")[-1]

                if file_extension in ["xlsx", "xls"]:
                    df = pd.read_excel(uploaded_file)
                    st.session_state.data_file_type = "Excel"
                elif file_extension in ["json", "jsonl"]:
                    df = load_json_file(uploaded_file)
                    st.session_state.data_file_type = (
                        "JSON" if file_extension == "json" else "JSONL"
                    )
                elif file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data_file_type = "CSV"
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                st.session_state.excel_df = df
                st.session_state.excel_columns = list(df.columns)
                st.session_state.last_uploaded_excel_name = uploaded_file.name
                st.session_state.excel_current_sample_df = (
                    None  # Reset to show head for new file
                )
                st.sidebar.success("Excel file loaded!")

            except Exception as e:
                st.sidebar.error(f"Error loading Excel: {e}")
                # Reset all Excel related states on error
                st.session_state.excel_df = None
                st.session_state.excel_columns = []
                st.session_state.last_uploaded_excel_name = None
                st.session_state.excel_current_sample_df = None

    if st.session_state.excel_df is not None:
        st.sidebar.write("Available Columns:", st.session_state.excel_columns)

        if st.sidebar.button(
            "Use Excel Columns as Context Keys", key="use_excel_cols_btn"
        ):
            if st.session_state.current_task:
                existing_keys = set(st.session_state.get("context_dict_keys", []))
                new_keys = set(st.session_state.excel_columns)
                all_keys = sorted(list(existing_keys.union(new_keys)))

                st.session_state.context_dict_keys = all_keys
                for example in st.session_state.few_shots:  # Update existing few-shots
                    if "context" not in example or not isinstance(
                        example["context"], dict
                    ):
                        example["context"] = {}
                    for key_name in st.session_state.context_dict_keys:
                        if key_name not in example["context"]:
                            example["context"][key_name] = ""
                st.sidebar.success("Context keys updated from Excel columns.")
                st.rerun()
            else:
                st.sidebar.warning(
                    "Please select or create a task first to apply context keys."
                )

        st.sidebar.markdown("---")
        st.sidebar.write("**Data Preview Controls:**")

        # Number input for sample/head size
        # Ensure max_value is at least 1 even if df is empty or None temporarily
        df_len = (
            len(st.session_state.excel_df)
            if st.session_state.excel_df is not None
            else 0
        )
        max_rows_for_input = max(1, df_len)

        # Update session state directly from number_input's value if it changes
        st.session_state.excel_sample_n = st.sidebar.number_input(
            "Number of rows for preview:",
            min_value=1,
            max_value=max_rows_for_input,
            value=st.session_state.excel_sample_n,  # Controlled component
            step=1,
            key="excel_sample_n_input_controller",  # Unique key
        )

        col1, col2 = st.sidebar.columns(2)
        if col1.button("📊 Show Random Sample", key="show_random_sample_btn"):
            n_to_sample = st.session_state.excel_sample_n
            if df_len > 0:
                actual_n = min(n_to_sample, df_len)
                st.session_state.excel_current_sample_df = (
                    st.session_state.excel_df.sample(n=actual_n)
                )
            else:  # No data to sample
                st.session_state.excel_current_sample_df = (
                    pd.DataFrame()
                )  # Show empty dataframe
                st.sidebar.caption("No data to sample.")

        if col2.button("📋 Show First N Rows", key="show_head_sample_btn"):
            st.session_state.excel_current_sample_df = (
                None  # Signal to show df.head() with N rows
            )

        st.sidebar.markdown("**Sample Data Preview:**")
        if st.session_state.excel_current_sample_df is not None:
            # Displaying a previously generated random sample
            st.sidebar.caption(
                f"Displaying {len(st.session_state.excel_current_sample_df)} random rows:"
            )
            st.sidebar.dataframe(st.session_state.excel_current_sample_df)
        elif st.session_state.excel_df is not None:
            # Default: Displaying top N rows
            n_for_head = st.session_state.excel_sample_n
            actual_n_head = min(n_for_head, df_len)
            if actual_n_head > 0:
                st.sidebar.caption(f"Displaying first {actual_n_head} rows:")
                st.sidebar.dataframe(st.session_state.excel_df.head(actual_n_head))
            else:
                st.sidebar.caption("Excel file loaded, but it's empty.")
        else:
            st.sidebar.caption("Upload an Excel file to see data preview.")
    else:  # No Excel file loaded at all
        st.sidebar.caption(
            "Upload an Excel file to inspect its data and use its columns."
        )


def generate_text_prompt_node_config(
    current_task_name: str,
    model_name: str,
    inference_engine: str,
    engine_options: Dict[str, Any],
    generation_options: Dict[str, Any],
    num_few_shots: int,
    template_context_map: Dict[str, str],  # Added to allow future flexibility
    output_data_attribute: str = "generated_llm_output",
    node_id_prefix: str = "text_prompt_step",
) -> str:
    """
    Generates a JSON configuration string for a Text Prompt Node
    based on the provided settings.
    """
    node_id = f"{node_id_prefix}_{current_task_name.lower().replace(' ', '_')}"
    if not current_task_name:
        node_id = f"{node_id_prefix}_untitled"  # Fallback if task name is empty

    config = {
        "id": node_id,
        "type": "text_prompt",
        "params": {
            "name": current_task_name,  # Corresponds to the prompt folder name
            "template_context_map": template_context_map,  # Use the passed map
            "system_prompt_file": "system_prompt.txt",  # Default, as saved by app
            "user_prompt_file": "user_prompt.txt",  # Default, as saved by app
            "few_shot_lines_file": "few_shot.jsonl",  # Default, as saved by app
            "num_few_shots": num_few_shots,
            "model_name": model_name,
            "inference_engine": inference_engine,
            "engine_options": engine_options,
            "generation_options": generation_options,
            "output_data_attribute": output_data_attribute,
            "resume": False,  # Default value
            "parse_json": False,  # Default value
        },
        "dependencies": ["id_of_previous_data_loading_node"],  # Placeholder
    }
    return json.dumps(config, indent=2)


def run_test_workflow_directly(
    task_name,
    system_prompt_content,  # Actual system prompt string
    user_prompt_content,  # Actual user prompt template string
    few_shots_list,  # Actual list of few-shot example dicts
    sample_df,
    data_primary_key,
    model_name_for_test,
    inference_engine_for_test,
    template_context_map_for_test,  # The map like {"template_var": "excel_col_key"}
    num_few_shots_for_test,  # Number of few-shots to actually use for this run
    engine_options_for_test,
    generation_options_for_test,
):
    """
    Runs a simplified test of the TextPromptNode with current UI settings
    and includes the formatted prompt in the results.
    """
    # Temporary directory for the TextPromptNode's own files (prompts, data, output)
    temp_root_dir_obj = tempfile.TemporaryDirectory(
        prefix=f"streamlit_test_run_{task_name}_"
    )
    temp_root_path = Path(temp_root_dir_obj.name)

    test_run_results = []
    test_run_errors = []

    # --- Setup PromptFormatter for Display Purposes ---
    # This formatter uses the direct content from the UI to show what *would* be formatted.
    display_formatter_temp_dir_obj = tempfile.TemporaryDirectory(
        prefix=f"display_formatter_{task_name}_"
    )
    display_formatter_prompts_base = Path(display_formatter_temp_dir_obj.name)
    # PromptFormatter might expect prompts within a task-named subdirectory from its base PROMPT_DIR
    display_formatter_task_prompts_dir = display_formatter_prompts_base / task_name
    display_formatter_task_prompts_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary prompt files for the display_formatter
    display_sys_prompt_file = display_formatter_task_prompts_dir / "system_prompt.txt"
    display_user_prompt_file = (
        display_formatter_task_prompts_dir / "user_prompt.txt"
    )  # Ensure extension matches PromptFormatter expectation if any
    display_few_shot_file = display_formatter_task_prompts_dir / "few_shot.jsonl"

    with open(display_sys_prompt_file, "w", encoding="utf-8") as f:
        f.write(system_prompt_content)
    with open(display_user_prompt_file, "w", encoding="utf-8") as f:
        f.write(user_prompt_content)  # This is the template string

    # Use only the selected number of few-shot examples for formatting
    actual_few_shots_for_display = (
        few_shots_list[:num_few_shots_for_test] if few_shots_list else []
    )
    if actual_few_shots_for_display:
        with open(display_few_shot_file, "w", encoding="utf-8") as f:
            for example in actual_few_shots_for_display:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
    else:  # Create an empty file if no few-shots, as PromptFormatter might expect it
        with open(display_few_shot_file, "w", encoding="utf-8") as f:
            pass

    display_formatter = None
    try:
        display_formatter = PromptFormatter(
            system_prompt_path=display_sys_prompt_file,  # Needs to be a Path object
            user_prompt_template_path=display_user_prompt_file,  # Needs to be a Path object
            few_shot_examples_path=display_few_shot_file,  # Needs to be a Path object
            num_few_shots=len(
                actual_few_shots_for_display
            ),  # Number of examples written to its temp file
            # Add other PromptFormatter params if your version requires them (e.g., few_shot_context_key)
        )
        logger.info("Display Formatter initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize display_formatter: {e}", exc_info=True)
        # Optionally, add to test_run_errors if this failure is critical for display
    # display_formatter_temp_dir_obj will be cleaned up at the end of this function in the main finally block.

    # --- Main Test Execution Logic (for TextPromptNode) ---
    try:
        logger.info(
            f"Starting test run in TextPromptNode's temporary directory: {temp_root_path}"
        )
        # 1. Prepare directories for TextPromptNode
        temp_prompts_base_for_node = temp_root_path / "prompts_base_for_test"
        temp_task_specific_prompts_dir_for_node = temp_prompts_base_for_node / task_name
        temp_task_specific_prompts_dir_for_node.mkdir(parents=True, exist_ok=True)
        temp_data_dir = temp_root_path / "data_for_test"
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        temp_output_dir = temp_root_path / "output_for_test"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. Save prompts to TextPromptNode's temporary directory
        # These files are for the actual TextPromptNode instance.
        with open(
            temp_task_specific_prompts_dir_for_node / "system_prompt.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(system_prompt_content)
        with open(
            temp_task_specific_prompts_dir_for_node / "user_prompt.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(user_prompt_content)

        # Use num_few_shots_for_test to control how many few-shots the TextPromptNode itself loads
        actual_few_shots_for_node = (
            few_shots_list[:num_few_shots_for_test] if few_shots_list else []
        )
        if actual_few_shots_for_node:
            with open(
                temp_task_specific_prompts_dir_for_node / "few_shot.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                for example in actual_few_shots_for_node:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
        else:
            with open(
                temp_task_specific_prompts_dir_for_node / "few_shot.jsonl",
                "w",
                encoding="utf-8",
            ) as f:
                pass

        # 3. Save sample DataFrame
        temp_excel_path = temp_data_dir / "sample_input_data.xlsx"
        logger.info(f"Saving temporary sample data to {temp_excel_path}")
        sample_df.to_excel(temp_excel_path, index=False)

        data_loader = DataFileLoader(
            input_data_path=temp_excel_path, primary_key=data_primary_key
        )
        loaded_sample_data_dict = data_loader.load_input_data()

        if not loaded_sample_data_dict:
            test_run_errors.append(
                "Failed to load sample data or sample data is empty."
            )
        else:
            logger.info(
                f"Loaded {len(loaded_sample_data_dict)} items for test processing."
            )
            # 5. Configure and run TextPromptNode
            node_params = {
                "name": task_name,  # TextPromptNode will look for prompts in PROMPT_DIR/task_name
                "primary_key": data_primary_key,
                "model_name": model_name_for_test,
                "output_data_attribute": "test_llm_output",
                "system_prompt_file": "system_prompt.txt",  # Relative to PROMPT_DIR/task_name
                "user_prompt_file": "user_prompt.txt",  # Relative to PROMPT_DIR/task_name
                "few_shot_lines_file": "few_shot.jsonl",  # Relative to PROMPT_DIR/task_name
                "num_few_shots": len(
                    actual_few_shots_for_node
                ),  # Number of examples TextPromptNode should load
                "template_context_map": template_context_map_for_test or {},
                "inference_engine": inference_engine_for_test,
                "engine_options": engine_options_for_test or {},
                "generation_options": generation_options_for_test or {},
            }
            test_node = TextPromptNode(
                node_id="streamlit_test_node",
                node_type="text_prompt",
                parent_wf_name="streamlit_test_wf",
                data_dir=temp_data_dir,  # TextPromptNode's data dir
                output_dir=temp_output_dir,  # TextPromptNode's output dir
                prompts_dir=temp_prompts_base_for_node,  # TextPromptNode's prompts dir
                params=node_params,
            )
            test_node.input_data_path = temp_excel_path
            test_node.primary_key = data_primary_key
            logger.info("Initializing TextPromptNode dependencies for test...")
            test_node.setup_processing()  # This initializes test_node.prompt_formatter

            logger.info("Dependencies initialized. Starting item processing...")
            for item_pk, item_row_data in loaded_sample_data_dict.items():
                # Initialize with a default error message or structure in case of failure
                raw_formatted_prompt_for_display: Any = {
                    "error": "Display formatter was not available or prompt generation failed."
                }
                llm_output_value = None
                error_detail_value = None

                try:
                    # --- Prepare context for template rendering (for both display and actual node) ---
                    # This context is based on the item_row_data and template_context_map_for_test
                    # Start with all row data as template variables (automatic matching)
                    template_vars_for_item: Dict[str, Any] = item_row_data.copy()

                    # Apply explicit mappings, which can override automatic matches
                    current_context_map = template_context_map_for_test or {}
                    if current_context_map:
                        for template_var, data_key in current_context_map.items():
                            template_vars_for_item[template_var] = item_row_data.get(
                                data_key, ""
                            )  # Use .get for safety

                    # --- Generate formatted prompt for display using display_formatter ---
                    if (
                        display_formatter
                    ):  # Check if display_formatter was successfully initialized
                        try:
                            # `create_messages` typically takes the dictionary of template variables
                            # Its output is expected to be a JSON-serializable object (e.g., list of dicts)
                            messages_for_display = display_formatter.create_messages(
                                template_vars_for_item
                            )
                            raw_formatted_prompt_for_display = (
                                messages_for_display  # Store the direct object
                            )
                        except Exception as fmt_e:
                            logger.error(
                                f"Error formatting prompt for display for item {item_pk}: {fmt_e}",
                                exc_info=True,
                            )
                            # Store a dictionary with error info, which is JSON serializable
                            raw_formatted_prompt_for_display = {
                                "error": "Prompt formatting for display failed.",
                                "detail": str(fmt_e),
                            }
                    else:  # display_formatter itself was not initialized
                        raw_formatted_prompt_for_display = {
                            "error": "Display formatter instance was not available."
                        }

                    # --- Actual processing by TextPromptNode ---
                    # _process_item uses its own internal prompt_formatter and the item_row_data
                    # to construct its context based on its own template_context_map.
                    llm_output_value = test_node.process_item(
                        str(item_pk), item_row_data
                    )

                except Exception as proc_e:
                    logger.error(
                        f"Error processing item {item_pk} in test run: {proc_e}",
                        exc_info=True,
                    )
                    error_detail_value = str(proc_e)
                    test_run_errors.append(f"Error on item {item_pk}: {proc_e}")
                    # If processing fails, raw_formatted_prompt_for_display might still have a value from above.
                    # If the error was before prompt formatting, it will retain its initial error state.

                test_run_results.append(
                    {
                        "primary_key": item_pk,
                        "input_data": item_row_data,
                        "formatted_prompt": raw_formatted_prompt_for_display,  # Store the raw object
                        "llm_output": llm_output_value,
                        "error_detail": error_detail_value,
                    }
                )
            logger.info("Test item processing finished.")
    except Exception as e:
        logger.error(
            f"Critical error during test run setup or execution: {e}", exc_info=True
        )
        test_run_errors.append(f"Test run failed critically: {e}")
    finally:
        display_formatter_temp_dir_obj.cleanup()  # Clean up temp dir for display_formatter
        # temp_root_dir_obj (for TextPromptNode) is returned and cleaned up by Streamlit as before.

    return test_run_results, test_run_errors, temp_root_dir_obj


def display_test_run_controls_in_sidebar():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 Test Current Task")

    # --- Preset Selector --- (This part remains the same)
    preset_options = list(ENGINE_PRESETS.keys())
    current_preset_name = st.session_state.get(
        "test_selected_engine_preset", preset_options[0]
    )
    if current_preset_name not in preset_options:
        current_preset_name = preset_options[0]
        st.session_state.test_selected_engine_preset = current_preset_name
    selected_preset_name = st.sidebar.selectbox(
        "Engine Configuration Preset",
        options=preset_options,
        index=preset_options.index(current_preset_name),
        key="test_preset_selector_sidebar",
    )
    if selected_preset_name != st.session_state.get("test_selected_engine_preset"):
        st.session_state.test_selected_engine_preset = selected_preset_name
        preset_data = ENGINE_PRESETS[selected_preset_name]
        st.session_state.test_inference_engine = preset_data["inference_engine"]
        st.session_state.test_model_name_cache = preset_data["model_name"]
        st.session_state.test_engine_options_str_cache = json.dumps(
            preset_data.get("engine_options", {})
        )
        st.session_state.test_generation_options_str_cache = json.dumps(
            preset_data.get("generation_options", {})
        )
        st.rerun()

    # --- Inference Engine Selector --- (This part remains the same)
    current_engine_in_state = st.session_state.get(
        "test_inference_engine", "huggingface"
    )
    engine_options_list = ["huggingface", "llama_cpp"]
    try:
        current_engine_index = engine_options_list.index(current_engine_in_state)
    except ValueError:
        current_engine_index = 0
        st.session_state.test_inference_engine = engine_options_list[0]
    test_inference_engine = st.sidebar.selectbox(
        "Inference Engine",
        options=engine_options_list,
        index=current_engine_index,
        key="test_inference_engine_selector_sidebar",
    )
    if test_inference_engine != st.session_state.get("test_inference_engine"):
        st.session_state.test_inference_engine = test_inference_engine
        if (
            st.session_state.test_selected_engine_preset != "Custom"
            and ENGINE_PRESETS[st.session_state.test_selected_engine_preset][
                "inference_engine"
            ]
            != test_inference_engine
        ):
            st.session_state.test_selected_engine_preset = "Custom"
            st.rerun()

    # Model for testing - populated by preset or custom input (This part remains the same)
    test_model_name = st.sidebar.text_input(
        "Model for Test Run (Path or HF ID)",
        value=st.session_state.get("test_model_name_cache", ""),
        key="test_model_name_input_sidebar",
    )
    if test_model_name != st.session_state.get("test_model_name_cache"):
        st.session_state.test_model_name_cache = test_model_name
        if (
            st.session_state.test_selected_engine_preset != "Custom"
            and ENGINE_PRESETS[st.session_state.test_selected_engine_preset][
                "model_name"
            ]
            != test_model_name
        ):
            st.session_state.test_selected_engine_preset = "Custom"
            st.rerun()

    # Primary key from Excel data for the test (This part remains the same)
    excel_df_exists = st.session_state.get("excel_df") is not None  # Keep this check
    if not st.session_state.get("excel_columns"):
        st.sidebar.caption("Upload Excel to select Primary Key.")
        test_primary_key = ""
    else:
        excel_cols = st.session_state.excel_columns
        possible_pks = [
            col for col in excel_cols if "id" in col.lower() or "key" in col.lower()
        ]
        default_pk_index = 0
        if possible_pks:
            try:
                default_pk_index = excel_cols.index(possible_pks[0])
            except ValueError:
                if excel_cols:
                    default_pk_index = 0
        elif excel_cols:
            default_pk_index = 0
        test_primary_key = st.sidebar.selectbox(
            "Primary Key Column for Test Data",
            options=excel_cols,
            index=default_pk_index,
            key="test_primary_key_selector_sidebar",
        )

    # --- REMOVED: Number of rows for testing input ---
    # The number of rows will now be derived from the current Excel preview settings

    # Engine options - populated by preset or custom input (This part remains the same)
    test_engine_options_str = st.sidebar.text_area(
        "Engine Options (JSON string)",
        value=st.session_state.get("test_engine_options_str_cache", "{}"),
        height=100,
        key="test_engine_options_str_sidebar",
    )
    if test_engine_options_str != st.session_state.get("test_engine_options_str_cache"):
        st.session_state.test_engine_options_str_cache = test_engine_options_str
        try:
            current_opts = json.loads(test_engine_options_str)
            if (
                st.session_state.test_selected_engine_preset != "Custom"
                and ENGINE_PRESETS[st.session_state.test_selected_engine_preset].get(
                    "engine_options", {}
                )
                != current_opts
            ):
                st.session_state.test_selected_engine_preset = "Custom"
                st.rerun()
        except json.JSONDecodeError:
            pass

    # Generation options - populated by preset or custom input (This part remains the same)
    test_generation_options_str = st.sidebar.text_area(
        "Generation Options (JSON string)",
        value=st.session_state.get("test_generation_options_str_cache", "{}"),
        height=100,
        key="test_generation_options_str_sidebar",
    )
    if test_generation_options_str != st.session_state.get(
        "test_generation_options_str_cache"
    ):
        st.session_state.test_generation_options_str_cache = test_generation_options_str
        try:
            current_gen_opts = json.loads(test_generation_options_str)
            if (
                st.session_state.test_selected_engine_preset != "Custom"
                and ENGINE_PRESETS[st.session_state.test_selected_engine_preset].get(
                    "generation_options", {}
                )
                != current_gen_opts
            ):
                st.session_state.test_selected_engine_preset = "Custom"
                st.rerun()
        except json.JSONDecodeError:
            pass

    if st.sidebar.button(
        "🚀 Run Test with Sample Data",
        key="run_test_button_sidebar",
        disabled=(not excel_df_exists or not test_primary_key),
    ):
        actual_model_name = st.session_state.test_model_name_cache
        actual_inference_engine = st.session_state.test_inference_engine
        actual_engine_options_str = st.session_state.test_engine_options_str_cache
        actual_generation_options_str = (
            st.session_state.test_generation_options_str_cache
        )

        # --- NEW LOGIC: Determine sample_df_for_test from current preview ---
        sample_df_for_test = None
        num_samples_for_test_run = 0
        data_source_message = ""

        # Check if a random sample is currently displayed and is not empty
        if (
            st.session_state.get("excel_current_sample_df") is not None
            and not st.session_state.excel_current_sample_df.empty
        ):
            sample_df_for_test = (
                st.session_state.excel_current_sample_df.copy()
            )  # Use a copy
            num_samples_for_test_run = len(sample_df_for_test)
            data_source_message = f"Using the displayed random sample of {num_samples_for_test_run} row(s) for the test."
        # Else, check if the main excel_df is available for taking a head (as per preview settings)
        elif st.session_state.get("excel_df") is not None:
            if (
                not st.session_state.excel_df.empty
            ):  # Ensure main DataFrame is not empty
                # Get the 'N' from "Number of rows for preview" input
                n_for_head_test = st.session_state.get(
                    "excel_sample_n", 1
                )  # Default to 1 if somehow not set
                # df.head(N) handles N > len(df) gracefully by returning all rows.
                sample_df_for_test = st.session_state.excel_df.head(
                    n_for_head_test
                ).copy()  # Use a copy
                num_samples_for_test_run = len(
                    sample_df_for_test
                )  # Actual number of rows in the head

                if num_samples_for_test_run > 0:
                    data_source_message = f"Using the first {num_samples_for_test_run} row(s) (from preview settings) for the test."
                # If head is empty (e.g. excel_sample_n was 0 or excel_df was empty), num_samples_for_test_run will be 0
            else:  # excel_df exists but is empty
                num_samples_for_test_run = 0  # Explicitly set to 0
        # If neither condition met, sample_df_for_test remains None or empty, and num_samples_for_test_run remains 0

        if (
            sample_df_for_test is None
            or sample_df_for_test.empty
            or num_samples_for_test_run == 0
        ):
            st.sidebar.error(
                "The data snippet for testing is empty. "
                "Please check Excel preview settings or upload a non-empty file."
            )
            return  # Stop the test run

        if data_source_message:  # Briefly show user where the data is coming from
            st.sidebar.info(data_source_message)

        # --- Existing checks ---
        if not test_primary_key:
            st.sidebar.error("Please select a primary key for the test data.")
            return
        elif not actual_model_name:
            st.sidebar.error("Please specify a model name for the test run.")
            return
        # No need to check excel_df_exists explicitly here, as it's covered by sample_df_for_test logic

        # --- Spinner and Test Execution ---
        # num_rows_for_test is replaced by num_samples_for_test_run
        spinner_message = (
            f"Running test with {actual_model_name} using {actual_inference_engine} "
            f"on {num_samples_for_test_run} sample(s) from current preview..."
        )
        with st.spinner(spinner_message):
            # Use the configured template context map
            current_template_context_map = st.session_state.get(
                "template_context_map", {}
            )
            num_few_shots_to_use = len(st.session_state.get("few_shots", []))

            engine_opts_for_test = {}
            try:
                parsed_opts = json.loads(actual_engine_options_str)
                if isinstance(parsed_opts, dict):
                    engine_opts_for_test = parsed_opts
                else:
                    st.sidebar.error("Engine Options must be a valid JSON object.")
                    return
            except json.JSONDecodeError as e:
                st.sidebar.error(f"Invalid JSON in Engine Options: {e}")
                return

            generation_opts_for_test = {}
            try:
                parsed_gen_opts = json.loads(actual_generation_options_str)
                if isinstance(parsed_gen_opts, dict):
                    generation_opts_for_test = parsed_gen_opts
                else:
                    st.sidebar.error("Generation Options must be a valid JSON object.")
                    return
            except json.JSONDecodeError as e:
                st.sidebar.error(f"Invalid JSON in Generation Options: {e}")
                return

            results, errors, temp_dir_obj = run_test_workflow_directly(
                st.session_state.current_task,
                st.session_state.system_prompt,
                st.session_state.user_prompt_template,
                st.session_state.get("few_shots", []),
                sample_df_for_test,  # Use the determined sample DataFrame
                test_primary_key,
                actual_model_name,
                actual_inference_engine,
                current_template_context_map,
                num_few_shots_to_use,
                engine_opts_for_test,
                generation_opts_for_test,
            )
            st.session_state.last_test_results = results
            st.session_state.last_test_errors = errors
            if (
                "temp_dir_to_clean" in st.session_state
                and st.session_state.temp_dir_to_clean
            ):
                try:
                    st.session_state.temp_dir_to_clean.cleanup()
                except Exception as e:
                    logger.warning(f"Could not cleanup previous temp directory: {e}")
            st.session_state.temp_dir_to_clean = temp_dir_obj

        if st.session_state.last_test_errors:
            for error_msg in st.session_state.last_test_errors:
                st.toast(f"Test Error: {error_msg}", icon="🚨")
        if st.session_state.last_test_results:
            st.toast(
                f"Test run finished for {len(st.session_state.last_test_results)} items!",
                icon="✅",
            )
        st.rerun()


def display_test_run_results_main_area():
    if "last_test_results" in st.session_state and st.session_state.last_test_results:
        st.markdown("---")
        st.subheader("🔬 Test Run Results")

        # Display general errors if any (errors not tied to a specific item)
        if st.session_state.get("last_test_errors"):
            # Filter out errors that are already displayed with items
            item_specific_error_details = set()
            for res_item_check in st.session_state.last_test_results:
                if res_item_check.get("error_detail"):
                    item_specific_error_details.add(
                        str(res_item_check.get("error_detail"))
                    )

            general_errors_to_display = []
            for error_msg in st.session_state.last_test_errors:
                is_general = True
                for item_err_detail in item_specific_error_details:
                    # Basic check if the general error message is part of an item's detailed error
                    if str(error_msg) in item_err_detail or item_err_detail in str(
                        error_msg
                    ):
                        is_general = False
                        break
                if is_general:
                    general_errors_to_display.append(error_msg)

            if general_errors_to_display:
                st.error("Encountered General Test Run Errors:")
                for error_msg in list(
                    dict.fromkeys(general_errors_to_display)
                ):  # Deduplicate
                    st.caption(f"- {error_msg}")

        for res_item in st.session_state.last_test_results:
            pk_display = res_item.get("primary_key", "N/A")
            # Default expander to open if there's an error for this item, or always if preferred.
            expanded_by_default = True

            with st.expander(
                f"Test Result for Item (PK: {pk_display})", expanded=expanded_by_default
            ):
                st.markdown("**Input Data:**")
                # Input data is typically a dict, st.json is good here.
                st.json(res_item.get("input_data", {}), expanded=False)

                st.markdown("**Formatted Prompt (JSON for LLM):**")
                formatted_prompt_data = res_item.get("formatted_prompt")

                if formatted_prompt_data is not None:
                    # formatted_prompt_data is expected to be a list of dicts (messages)
                    # or a dict (e.g., if an error occurred during its generation).
                    if isinstance(formatted_prompt_data, (list, dict)):
                        # st.json will pretty-print the list/dictionary.
                        # The `expanded` parameter in st.json controls the initial expansion of levels *within* the JSON object.
                        # For a list of messages, it's usually clear enough.
                        st.json(formatted_prompt_data, expanded=True)
                    else:
                        # Fallback if it's an unexpected simple string (e.g., older error message format)
                        st.text(f"Formatted prompt (raw): {str(formatted_prompt_data)}")
                else:
                    st.caption("Formatted prompt data not available for this item.")

                st.markdown("**LLM Output:**")
                if res_item.get(
                    "error_detail"
                ):  # Specific error during this item's processing
                    st.error(f"Processing Error: {res_item['error_detail']}")

                llm_output = res_item.get("llm_output")
                if llm_output is not None:
                    if isinstance(
                        llm_output, (dict, list)
                    ):  # If LLM output is structured JSON
                        st.json(llm_output, expanded=True)
                    else:  # If LLM output is a string
                        st.text_area(
                            "LLM Response Text:",
                            value=str(llm_output),
                            height=150,
                            disabled=True,
                            key=f"llm_output_text_area_{pk_display}",
                        )
                elif not res_item.get("error_detail"):  # No error, but also no output
                    st.info(
                        "No output was generated by the LLM for this item (output was None)."
                    )
                # If there was an error_detail, it's already displayed, so no need for another message if llm_output is None.

        st.markdown("---")
        st.subheader("📋 Generated Text Prompt Node Configuration")
        st.markdown(
            "Below is a JSON configuration for a Text Prompt Node based on the settings "
            "used in the last test run. You can copy this for your workflow files."
        )

        # Retrieve the settings used for the test run
        # These should match what was passed to run_test_workflow_directly
        task_name_for_config = st.session_state.get("current_task", "my_task")

        # Get values from session state cache (source of truth for test run inputs)
        model_name_for_config = st.session_state.get("test_model_name_cache", "")
        inference_engine_for_config = st.session_state.get(
            "test_inference_engine", "huggingface"
        )

        engine_options_str_for_config = st.session_state.get(
            "test_engine_options_str_cache", "{}"
        )
        try:
            engine_options_for_config = json.loads(engine_options_str_for_config)
            if not isinstance(engine_options_for_config, dict):  # Ensure it's a dict
                engine_options_for_config = {}
                st.warning(
                    "Engine options from test run were not a valid JSON object; using empty for config."
                )
        except json.JSONDecodeError:
            engine_options_for_config = {}  # Fallback if JSON is invalid
            st.warning(
                "Could not parse engine options from test run; using empty for config."
            )

        generation_options_str_for_config = st.session_state.get(
            "test_generation_options_str_cache", "{}"
        )
        try:
            generation_options_for_config = json.loads(
                generation_options_str_for_config
            )
            if not isinstance(
                generation_options_for_config, dict
            ):  # Ensure it's a dict
                generation_options_for_config = {}
                st.warning(
                    "Generation options from test run were not a valid JSON object; using empty for config."
                )
        except json.JSONDecodeError:
            generation_options_for_config = {}  # Fallback if JSON is invalid
            st.warning(
                "Could not parse generation options from test run; using empty for config."
            )

        num_few_shots_for_config = len(st.session_state.get("few_shots", []))

        # Use the configured template context map from the UI
        template_context_map_for_config = st.session_state.get(
            "template_context_map", {}
        )

        generated_json_config = generate_text_prompt_node_config(
            current_task_name=task_name_for_config,
            model_name=model_name_for_config,
            inference_engine=inference_engine_for_config,
            engine_options=engine_options_for_config,
            generation_options=generation_options_for_config,
            num_few_shots=num_few_shots_for_config,
            template_context_map=template_context_map_for_config,
            # You can customize the output_data_attribute and node_id_prefix if needed
            # output_data_attribute="my_llm_output_field",
            # node_id_prefix="llm_node"
        )

        st.code(generated_json_config, language="json")
        st.caption(
            "Note: `dependencies` and `template_context_map` might need to be adjusted to fit your specific workflow."
        )
        # Button to clear results
        if st.button(
            "Clear Test Results and Cleanup Files", key="clear_test_results_btn_main"
        ):
            st.session_state.last_test_results = None
            st.session_state.last_test_errors = None
            if (
                "temp_dir_to_clean" in st.session_state
                and st.session_state.temp_dir_to_clean
            ):
                try:
                    st.session_state.temp_dir_to_clean.cleanup()
                    logger.info(
                        "Successfully cleaned up temporary directory for TextPromptNode."
                    )
                    st.toast("Temporary test files (for node) cleaned up.", icon="🧹")
                    del st.session_state.temp_dir_to_clean
                except Exception as e:
                    logger.warning(
                        f"Could not cleanup temporary directory for TextPromptNode: {e}"
                    )
                    st.warning(
                        f"Could not cleanup temporary test files (for node): {e}"
                    )
            else:
                logger.info(
                    "No 'temp_dir_to_clean' found in session state for TextPromptNode files."
                )
            st.rerun()


def add_context_key_callback():
    """Callback to add a new context key and clear the input field."""
    key_to_add = (
        st.session_state.new_key_name_input
    )  # Read from the text input's session state

    if key_to_add:
        if not re.match(r"^[a-zA-Z0-9_]+$", key_to_add):
            st.error("Key name must consist of letters, numbers, and underscores only.")
        elif key_to_add in st.session_state.get("context_dict_keys", []):
            st.error(f"Key '{key_to_add}' already exists.")
        else:
            if "context_dict_keys" not in st.session_state:
                st.session_state.context_dict_keys = []
            st.session_state.context_dict_keys.append(key_to_add)

            # Ensure all few_shot examples have the new key
            if "few_shots" not in st.session_state:
                st.session_state.few_shots = []
            for example in st.session_state.few_shots:
                if "context" not in example or not isinstance(example["context"], dict):
                    example["context"] = {}
                if key_to_add not in example["context"]:
                    example["context"][key_to_add] = ""

            st.success(f"Added context key: {key_to_add}")
            st.session_state.new_key_name_input = (
                ""  # Clear the input field for the next render
            )
            # No explicit st.rerun() needed here; Streamlit reruns after a callback.
    else:
        st.warning("Please enter a key name.")


def main():
    st.title("📝 Jinja Prompt & Example Manager")

    # Initialize session state variables ONCE
    if "initialized" not in st.session_state:
        st.session_state.few_shots = []
        st.session_state.editing_index = -1
        st.session_state.current_task = ""
        st.session_state.context_dict_keys = []
        st.session_state.system_prompt = ""
        st.session_state.user_prompt_template = ""
        st.session_state.new_asst_val = ""
        st.session_state.new_key_name_input_field = (
            ""  # For the 'Add Context Key' input
        )
        st.session_state.prompts_dir = "prompts/"  # Default prompts directory
        st.session_state.template_context_map = {}  # Template variable mapping

        # Excel related state
        st.session_state.excel_df = None
        st.session_state.excel_columns = []
        st.session_state.last_uploaded_excel_name = None
        st.session_state.excel_sample_n = 5
        st.session_state.excel_current_sample_df = None

        # For test runs
        st.session_state.last_test_results = None
        st.session_state.last_test_errors = None
        st.session_state.temp_dir_to_clean = (
            None  # For managing TemporaryDirectory object
        )
        st.session_state.test_model_name_cache = (
            "google/gemma-3-12b-it"  # Default test model
        )
        st.session_state.initialized = True

        st.session_state.test_selected_engine_preset = (
            "Hugging Face (gemma-3-4b-it)"  # Default preset
        )
        default_preset_data = ENGINE_PRESETS[
            st.session_state.test_selected_engine_preset
        ]
        st.session_state.test_inference_engine = default_preset_data["inference_engine"]
        st.session_state.test_model_name_cache = default_preset_data["model_name"]
        st.session_state.test_engine_options_str_cache = json.dumps(
            default_preset_data.get("engine_options", {})
        )
        st.session_state.test_generation_options_str_cache = json.dumps(
            default_preset_data.get("generation_options", {})
        )

        st.session_state.initialized = True  # Mark as initialized

    if "few_shots" not in st.session_state:
        st.session_state.few_shots = []
    if "editing_index" not in st.session_state:
        st.session_state.editing_index = -1
    if "current_task" not in st.session_state:
        st.session_state.current_task = ""
    if "context_dict_keys" not in st.session_state:
        st.session_state.context_dict_keys = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "user_prompt_template" not in st.session_state:
        st.session_state.user_prompt_template = ""
    # For clearing new example form inputs
    if "new_asst" not in st.session_state:
        st.session_state.new_asst = ""
    if "clear_new_example_form_fields" not in st.session_state:
        st.session_state.clear_new_example_form_fields = False

    with st.sidebar:
        st.header("📁 Prompts Directory")

        # Directory selector with validation
        new_prompts_dir = st.text_input(
            "Prompts Directory Path:",
            value=st.session_state.get("prompts_dir", "prompts/"),
            help="Path to the directory containing prompt templates",
            key="prompts_dir_input",
        )

        # Update prompts directory if changed
        if new_prompts_dir != st.session_state.get("prompts_dir"):
            st.session_state.prompts_dir = new_prompts_dir
            # Clear current task selection when directory changes
            st.session_state.current_task = ""
            st.rerun()

        # Display current directory status
        if os.path.exists(st.session_state.prompts_dir):
            st.success(f"✅ Using: `{st.session_state.prompts_dir}`")
        else:
            st.warning(f"⚠️ Directory will be created: `{st.session_state.prompts_dir}`")

        st.markdown("---")
        st.header("Task Management")
        task_dirs = get_task_directories(st.session_state.prompts_dir)
        task_options = [""] + sorted(task_dirs)

        current_task_val = st.session_state.get("current_task", "")
        try:
            select_box_index = task_options.index(current_task_val)
        except ValueError:
            select_box_index = 0
            if (
                current_task_val != ""
            ):  # If current_task was set but not found, reset it
                st.session_state.current_task = ""

        selected_task_name_from_widget = st.selectbox(
            "Select Task",
            task_options,
            index=select_box_index,
            format_func=lambda x: "✨ Create New Task ✨" if x == "" else x,
            key="task_selector_main",
        )

        if selected_task_name_from_widget != st.session_state.current_task:
            # The on_change callbacks for system_prompt and user_prompt_template text areas
            # would have already fired if the user typed in them and then clicked the selectbox,
            # updating st.session_state.system_prompt and st.session_state.user_prompt_template
            # with the content from the *old* task.

            st.session_state.current_task = (
                selected_task_name_from_widget  # Update current_task
            )

            if st.session_state.current_task:  # A task is selected
                (
                    system_prompt_loaded,
                    user_prompt_template_loaded,
                    few_shots_loaded,
                    context_keys_loaded,
                ) = load_task(
                    st.session_state.current_task, st.session_state.prompts_dir
                )
                st.session_state.system_prompt = system_prompt_loaded
                st.session_state.user_prompt_template = user_prompt_template_loaded
                st.session_state.few_shots = few_shots_loaded
                st.session_state.context_dict_keys = context_keys_loaded
                # Clear template context map when switching tasks
                st.session_state.template_context_map = {}

                # IMPORTANT: Update widget-specific state for the NEW task's text areas
                # This ensures they display the newly loaded content.
                sys_widget_key_new_task = f"sys_prompt_{st.session_state.current_task}"
                st.session_state[sys_widget_key_new_task] = system_prompt_loaded

                usr_widget_key_new_task = f"usr_tmpl_{st.session_state.current_task}"
                st.session_state[usr_widget_key_new_task] = user_prompt_template_loaded

                st.session_state.editing_index = -1
                st.session_state.excel_current_sample_df = None
            else:  # No task selected (e.g., "Create New Task" was selected)
                st.session_state.system_prompt = (
                    "You are a helpful assistant."  # Default for new
                )
                st.session_state.user_prompt_template = "{# Your Jinja2 template for the user prompt #}\n"  # Default for new
                st.session_state.few_shots = []
                st.session_state.context_dict_keys = []
                # Clear template context map for new task
                st.session_state.template_context_map = {}

                # Also update/initialize widget keys for this "no task" state if needed
                sys_widget_key_no_task = (
                    f"sys_prompt_{st.session_state.current_task}"  # current_task is ""
                )
                st.session_state[sys_widget_key_no_task] = (
                    st.session_state.system_prompt
                )

                usr_widget_key_no_task = (
                    f"usr_tmpl_{st.session_state.current_task}"  # current_task is ""
                )
                st.session_state[usr_widget_key_no_task] = (
                    st.session_state.user_prompt_template
                )

                st.session_state.editing_index = -1
                st.session_state.excel_current_sample_df = None
            st.rerun()

        if not st.session_state.current_task:
            with st.form("new_task_form"):
                new_task_name_input_val = st.text_input(
                    "New Task Name", key="new_task_name_form_input"
                )
                submitted_new_task = st.form_submit_button("Create Task")
                if submitted_new_task and new_task_name_input_val:
                    if new_task_name_input_val in task_dirs:
                        st.error(f"Task '{new_task_name_input_val}' already exists.")
                    else:
                        st.session_state.current_task = new_task_name_input_val
                        st.session_state.system_prompt = "You are a helpful assistant."
                        st.session_state.user_prompt_template = (
                            "{# Your Jinja2 template #}\n"
                        )
                        st.session_state.few_shots = []
                        st.session_state.context_dict_keys = []
                        save_task(
                            st.session_state.current_task,
                            st.session_state.system_prompt,
                            st.session_state.user_prompt_template,
                            st.session_state.few_shots,
                            st.session_state.prompts_dir,
                        )
                        st.success(f"Created new task: {st.session_state.current_task}")
                        st.rerun()

        if st.session_state.current_task:
            st.subheader(f"Actions for: {st.session_state.current_task}")
            if st.button("Duplicate Task", key="duplicate_btn_simple"):
                dupe_name_candidate = f"{st.session_state.current_task}_copy"
                i = 1
                final_dupe_name = dupe_name_candidate
                while final_dupe_name in get_task_directories(
                    st.session_state.prompts_dir
                ):
                    final_dupe_name = f"{dupe_name_candidate}_{i}"
                    i += 1
                success, message = duplicate_task(
                    st.session_state.current_task,
                    final_dupe_name,
                    st.session_state.prompts_dir,
                )
                if success:
                    st.success(message + f" as '{final_dupe_name}'")
                    st.session_state.current_task = final_dupe_name
                    st.rerun()
                else:
                    st.error(message)

            with st.expander("✏️ Rename Task"):
                # Using a dynamic key for the text input ensures it resets if the task selection changes,
                # and it helps manage its state correctly.
                new_task_name_for_rename = st.text_input(
                    "New Task Name:",
                    value=st.session_state.current_task,  # Pre-fill with the current name
                    key=f"rename_task_input_{st.session_state.current_task}",
                )
                if st.button(
                    "Confirm Rename",
                    key=f"confirm_rename_btn_{st.session_state.current_task}",
                ):
                    if new_task_name_for_rename and new_task_name_for_rename.strip():
                        old_name_for_rename = st.session_state.current_task
                        potential_new_name = new_task_name_for_rename.strip()

                        success, message = rename_task(
                            old_name_for_rename,
                            potential_new_name,
                            st.session_state.prompts_dir,
                        )
                        if success:
                            st.success(message)
                            # Clean up old dynamic widget keys for text areas
                            # These keys are tied to the old task name.
                            old_sys_widget_key = f"sys_prompt_{old_name_for_rename}"
                            if old_sys_widget_key in st.session_state:
                                del st.session_state[old_sys_widget_key]

                            old_usr_widget_key = f"usr_tmpl_{old_name_for_rename}"
                            if old_usr_widget_key in st.session_state:
                                del st.session_state[old_usr_widget_key]

                            # Update the current task name in session state
                            st.session_state.current_task = potential_new_name

                            # The main text area content (system_prompt, user_prompt_template)
                            # is already stored in st.session_state.system_prompt and
                            # st.session_state.user_prompt_template due to their on_change callbacks.
                            # When st.rerun() occurs, the UI will use the new current_task name,
                            # and the text area widgets will be re-initialized using these
                            # general state variables if their new dynamic keys are not found.
                            st.rerun()
                        else:
                            st.error(message)
                    else:
                        st.warning("New task name cannot be empty.")

            if st.button("🗑️ Delete Task", type="primary", key="delete_btn"):
                task_to_delete = st.session_state.current_task
                if delete_task(task_to_delete, st.session_state.prompts_dir):
                    st.success(f"Task '{task_to_delete}' deleted.")
                    st.session_state.current_task = ""
                    st.rerun()
                else:
                    st.error(f"Failed to delete task '{task_to_delete}'.")

        # Add test run controls to sidebar if a task is active and Excel data is loaded
        if st.session_state.current_task and st.session_state.excel_df is not None:
            display_test_run_controls_in_sidebar()
        elif st.session_state.current_task:
            st.sidebar.caption("Upload an Excel file to enable task testing.")

        display_data_inspector()  # Call the updated Excel display function

    if st.session_state.current_task:
        st.header(f"Editing Task: {st.session_state.current_task}")

        # System Prompt Text Area
        sys_widget_dynamic_key = f"sys_prompt_{st.session_state.current_task}"
        # Initialize widget state if it doesn't exist (e.g., first time for this task or after clearing state)
        if sys_widget_dynamic_key not in st.session_state:
            st.session_state[sys_widget_dynamic_key] = st.session_state.system_prompt

        st.text_area(
            "System Prompt Content:",
            value=st.session_state[
                sys_widget_dynamic_key
            ],  # Display from widget-specific key
            height=150,
            key=sys_widget_dynamic_key,
            on_change=system_prompt_changed,  # Add the callback
        )

        st.subheader("User Prompt Jinja2 Template")
        st.markdown(
            "Use Jinja2 (e.g., `{{ variable }}`). Variables defined as 'Context Dictionary Keys' are available."
        )

        # User Prompt Template Text Area
        usr_widget_dynamic_key = f"usr_tmpl_{st.session_state.current_task}"
        # Initialize widget state if it doesn't exist
        if usr_widget_dynamic_key not in st.session_state:
            st.session_state[usr_widget_dynamic_key] = (
                st.session_state.user_prompt_template
            )

        st.text_area(  # THE ORIGINAL TEXT AREA IN QUESTION
            "User Prompt Template Content:",
            value=st.session_state[
                usr_widget_dynamic_key
            ],  # Display from widget-specific key
            height=200,
            key=usr_widget_dynamic_key,
            on_change=user_prompt_template_changed,  # Add the callback
        )

        st.subheader("🔗 Template Context Mapping")
        st.markdown(
            "Map Excel column names (or input data keys) to template variables used in your Jinja2 templates. "
            "If no mapping is provided, input data keys are used directly as template variables."
        )

        # Display current mappings
        if st.session_state.template_context_map:
            st.write("**Current Mappings:**")
            for template_var, data_key in st.session_state.template_context_map.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.code(f"{{{{ {template_var} }}}} ← {data_key}")
                if col3.button(
                    "🗑️", key=f"remove_mapping_{template_var}", help="Remove mapping"
                ):
                    del st.session_state.template_context_map[template_var]
                    st.rerun()
        else:
            st.info(
                "No template context mappings configured. Input data keys will be used directly as template variables."
            )

        # Add new mapping
        with st.expander("➕ Add Template Context Mapping", expanded=False):
            col1, col2, col3 = st.columns([3, 3, 2])

            with col1:
                new_template_var = st.text_input(
                    "Template Variable Name",
                    placeholder="e.g., patient_age",
                    help="Variable name to use in Jinja2 template (e.g., {{ patient_age }})",
                    key="new_template_var_input",
                )

            with col2:
                # Show available Excel columns if data is loaded
                if st.session_state.get("excel_columns"):
                    available_keys = st.session_state.excel_columns
                    new_data_key = st.selectbox(
                        "Excel Column",
                        options=[""] + available_keys,
                        help="Excel column or input data key to map from",
                        key="new_data_key_select",
                    )
                else:
                    new_data_key = st.text_input(
                        "Input Data Key",
                        placeholder="e.g., PatientAge",
                        help="Excel column or input data key to map from",
                        key="new_data_key_input",
                    )

            with col3:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("Add Mapping", key="add_mapping_btn"):
                    if new_template_var and new_data_key:
                        if new_template_var in st.session_state.template_context_map:
                            st.error(
                                f"Template variable '{new_template_var}' already exists."
                            )
                        else:
                            st.session_state.template_context_map[new_template_var] = (
                                new_data_key
                            )
                            st.success(
                                f"Added mapping: {new_template_var} ← {new_data_key}"
                            )
                            st.rerun()
                    else:
                        st.error(
                            "Please provide both template variable name and data key."
                        )

        st.subheader("Few-shot Context Dictionary Keys")
        st.markdown(
            "Define keys for the `context` dictionary in few-shot examples. These should match template variables."
        )

        if st.session_state.context_dict_keys:
            st.write("Current Context Keys:")
            key_cols = st.columns(
                min(
                    4,
                    len(st.session_state.context_dict_keys)
                    if st.session_state.context_dict_keys
                    else 1,
                )
            )
            for i, key_name in enumerate(st.session_state.context_dict_keys):
                key_cols[i % 4].code(key_name)
            if st.button("Remove All Context Keys", key="remove_all_keys_btn"):
                st.session_state.context_dict_keys = []
                for example in st.session_state.few_shots:
                    example["context"] = {}
                st.success("All context keys removed. Examples' contexts cleared.")
                st.rerun()

        col1, col2 = st.columns([3, 1])
        with col1:
            # The text_input widget. Its state is stored in st.session_state.new_key_name_input.
            # The callback will update this session state variable to clear the input.
            st.text_input(
                "New Context Key Name",
                placeholder="e.g., patient_age",
                key="new_key_name_input",  # This key links the widget to st.session_state.new_key_name_input
            )
        with col2:
            # The button now uses the on_click callback.
            # The logic is moved into the add_context_key_callback function.
            st.button(
                "Add Context Key",
                key="add_key_btn",
                on_click=add_context_key_callback,  # Call the callback function on click
            )

        st.subheader("Few-shot Examples")
        if st.session_state.editing_index >= 0:
            st.markdown(f"**Editing Example {st.session_state.editing_index + 1}**")
            current_example = st.session_state.few_shots[st.session_state.editing_index]
            example_context_dict = current_example.get("context", {})
            edited_context_values = {}
            if not st.session_state.context_dict_keys:
                st.info("Define 'Context Dictionary Keys' to structure examples.")
            for key_name in st.session_state.context_dict_keys:
                edited_context_values[key_name] = st.text_area(
                    f"Context: `{key_name}`",
                    value=example_context_dict.get(key_name, ""),
                    height=80,
                    key=f"edit_ctx_{key_name}_{st.session_state.editing_index}",
                )
            edited_assistant_response = st.text_area(
                "Assistant Response",
                value=current_example.get("assistant", ""),
                height=100,
                key=f"edit_asst_{st.session_state.editing_index}",
            )
            col1, col2 = st.columns(2)
            if col1.button("✅ Update Example", key="update_ex_btn"):
                updated_context = {
                    key: edited_context_values[key]
                    for key in st.session_state.context_dict_keys
                }
                st.session_state.few_shots[st.session_state.editing_index][
                    "context"
                ] = updated_context
                st.session_state.few_shots[st.session_state.editing_index][
                    "assistant"
                ] = edited_assistant_response
                st.session_state.editing_index = -1
                st.success("Example updated.")
                st.rerun()
            if col2.button("❌ Cancel Editing", key="cancel_edit_btn"):
                st.session_state.editing_index = -1
                st.rerun()
        else:  # Adding a new example
            st.markdown("**Add New Few-shot Example**")
            if not st.session_state.context_dict_keys:
                st.info("Define 'Context Dictionary Keys' to structure new examples.")

            # --- Excel Pre-fill Section ---
            if (
                st.session_state.excel_df is not None
                and st.session_state.context_dict_keys
            ):
                with st.expander("Pre-fill from Excel Row (Optional)", expanded=False):
                    max_row_idx = len(st.session_state.excel_df) - 1
                    excel_row_idx_to_load = st.number_input(
                        f"Excel row index to load (0 to {max_row_idx})",
                        min_value=0,
                        max_value=max_row_idx,
                        step=1,
                        key="excel_prefill_row_idx_input",
                    )
                    if st.button("Load Data from Excel Row", key="load_from_excel_btn"):
                        if 0 <= excel_row_idx_to_load <= max_row_idx:
                            row_data_series = st.session_state.excel_df.iloc[
                                excel_row_idx_to_load
                            ]
                            for key_name in st.session_state.context_dict_keys:
                                widget_session_key = f"new_ctx_val_{key_name}"  # Session state key for the widget
                                if key_name in row_data_series:
                                    st.session_state[widget_session_key] = str(
                                        row_data_series[key_name]
                                    )
                                else:
                                    st.session_state[widget_session_key] = (
                                        ""  # Clear if not in Excel
                                    )
                                    st.warning(
                                        f"Excel column '{key_name}' not found. Field left blank/cleared."
                                    )
                            st.success(
                                f"Input fields populated from Excel row {excel_row_idx_to_load}. Review below."
                            )
                            # No rerun, let Streamlit update widgets from their session state keys
                        else:
                            st.error("Invalid Excel row index.")
            # --- End Excel Pre-fill Section ---

            # Check if form fields need to be cleared from a previous successful submission
            if st.session_state.get("clear_new_example_form_fields", False):
                # Safely get context_dict_keys, defaulting to an empty list if not found
                context_keys_for_clearing = st.session_state.get(
                    "context_dict_keys", []
                )
                for key_name_to_clear in context_keys_for_clearing:
                    widget_session_key_to_clear = f"new_ctx_val_{key_name_to_clear}"
                    if (
                        widget_session_key_to_clear in st.session_state
                    ):  # Check if key exists before clearing
                        st.session_state[widget_session_key_to_clear] = ""

                if "new_asst_val" in st.session_state:  # Check if key exists
                    st.session_state.new_asst_val = ""

                st.session_state.clear_new_example_form_fields = False  # Reset the flag

            with st.form("new_example_form"):
                # Ensure context_dict_keys exists and is a list
                current_context_keys_form = st.session_state.get(
                    "context_dict_keys", []
                )
                if not isinstance(current_context_keys_form, list):
                    current_context_keys_form = []

                for key_name in current_context_keys_form:
                    # Initialize session state for each text_area if not already set
                    widget_session_key = f"new_ctx_val_{key_name}"
                    if widget_session_key not in st.session_state:
                        st.session_state[widget_session_key] = ""
                    # Text area directly uses and updates its session state variable
                    st.text_area(
                        f"Context: `{key_name}`", height=80, key=widget_session_key
                    )

                # Initialize session state for assistant response if not set
                if "new_asst_val" not in st.session_state:
                    st.session_state.new_asst_val = ""
                st.text_area("Assistant Response", height=100, key="new_asst_val")

                submitted_new_example = st.form_submit_button("➕ Add Example")
                if submitted_new_example:
                    new_context_dict = {}
                    has_content = False

                    # Read values using .get for safety
                    for key_name in current_context_keys_form:
                        value = st.session_state.get(f"new_ctx_val_{key_name}", "")
                        new_context_dict[key_name] = value
                        if value.strip():
                            has_content = True

                    assistant_text = st.session_state.get("new_asst_val", "")
                    if assistant_text.strip():
                        has_content = True

                    if not has_content:
                        st.error("Cannot add an empty example.")
                    else:
                        if (
                            "few_shots" not in st.session_state
                        ):  # Ensure few_shots list exists
                            st.session_state.few_shots = []
                        st.session_state.few_shots.append(
                            {"context": new_context_dict, "assistant": assistant_text}
                        )
                        st.success("New example added.")

                        # --- CRITICAL CHANGE HERE ---
                        # DO NOT directly modify st.session_state variables tied to widget keys here.
                        # Instead, set the flag to clear them on the next run.
                        # REMOVE: st.session_state[f"new_ctx_val_{key_name}"] = "" (if you had a loop here)
                        # REMOVE: st.session_state.new_asst_val = ""

                        st.session_state.clear_new_example_form_fields = (
                            True  # Set the flag
                        )
                        st.rerun()

        if st.session_state.few_shots:
            st.markdown("---")
            st.write("**Current Few-shot Examples:**")
            for i, example in enumerate(st.session_state.few_shots):
                with st.expander(
                    f"Example {i + 1} (ID: {example.get('id', 'N/A')})", expanded=False
                ):
                    st.markdown("**Context:**")
                    if isinstance(example.get("context"), dict) and example["context"]:
                        st.json(example["context"])
                    elif isinstance(example.get("context"), dict):
                        st.caption("Context dictionary is empty.")
                    else:
                        st.code(str(example.get("context", "")))
                    st.markdown("**Assistant Response:**")
                    st.code(example.get("assistant", ""))
                    other_keys = {
                        k: v
                        for k, v in example.items()
                        if k not in ["context", "assistant"]
                    }
                    if other_keys:
                        st.markdown("**Other Metadata:**")
                        st.json(other_keys)
                    col1, _ = st.columns(
                        [1, 5]
                    )  # Edit/Delete buttons in a narrow column
                    if col1.button(f"✏️ Edit", key=f"edit_ex_btn_{i}"):
                        st.session_state.editing_index = i
                        st.rerun()
                    if col1.button(f"🗑️ Delete", key=f"del_ex_btn_{i}"):
                        st.session_state.few_shots.pop(i)
                        if st.session_state.editing_index == i:
                            st.session_state.editing_index = -1
                        elif st.session_state.editing_index > i:
                            st.session_state.editing_index -= 1
                        st.success(f"Example {i + 1} deleted.")
                        st.rerun()

        # Display test run results in the main area if they exist
        display_test_run_results_main_area()

        st.markdown("---")
        if st.button("💾 Save Task", type="primary", key="save_task_btn_main"):
            save_task(
                st.session_state.current_task,
                st.session_state.system_prompt,
                st.session_state.user_prompt_template,
                st.session_state.few_shots,
                st.session_state.prompts_dir,
            )
            st.success(f"Task '{st.session_state.current_task}' saved successfully!")

    else:
        st.info(
            "👋 Welcome! Select a task from the sidebar or create a new one to get started."
        )


if __name__ == "__main__":
    main()
