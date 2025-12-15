import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class PromptFormatter:
    """
    Loads prompt templates and few-shot examples using Jinja2, and formats messages.
    """

    def __init__(
        self,
        system_prompt_path: Union[str, Path],
        user_prompt_template_path: Union[str, Path],
        few_shot_examples_path: Optional[Union[str, Path]] = None,
        num_few_shots: int = 0,
        # Key names within the few-shot JSONL file:
        few_shot_context_key: str = "context",
        few_shot_assistant_key: str = "assistant",
        few_shot_id_key: str = "id",
    ):
        self.system_prompt_path = Path(system_prompt_path)
        self.user_prompt_template_path = Path(user_prompt_template_path)
        self.few_shot_examples_path = (
            Path(few_shot_examples_path) if few_shot_examples_path else None
        )
        self.num_few_shots = num_few_shots
        self.few_shot_context_key = few_shot_context_key
        self.few_shot_assistant_key = few_shot_assistant_key
        self.few_shot_id_key = few_shot_id_key

        # Initialize Jinja2 environment
        # The search path should be the directory containing the templates.
        # Assuming user_prompt_template_path and system_prompt_path are in the same directory.
        template_dir = self.user_prompt_template_path.parent
        self.jinja_env = Environment(
            loader=FileSystemLoader(searchpath=str(template_dir)),
            autoescape=select_autoescape(
                disabled_extensions=("txt", "j2", "template"),
                default_for_string=False,  # No autoescaping for from_string by default
                default=False,  # No autoescaping for get_template by default
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        logger.info(f"Jinja2 Environment initialized with search path: {template_dir}")

        # Load system prompt (as a string, to be rendered with from_string for flexibility)
        try:
            self.system_prompt_str = self._load_text_file(self.system_prompt_path)
        except FileNotFoundError:
            logger.warning(
                f"System prompt file not found: {self.system_prompt_path}. Using an empty system prompt."
            )
            self.system_prompt_str = ""

        # Load user prompt as a Jinja2 Template object
        try:
            self.user_template = self.jinja_env.get_template(
                self.user_prompt_template_path.name
            )
            logger.info(
                f"User prompt template loaded: {self.user_prompt_template_path.name}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load user prompt template '{self.user_prompt_template_path.name}' from '{template_dir}': {e}"
            )
            raise

        self.few_shot_examples = (
            self._load_few_shot_examples()
            if self.few_shot_examples_path and self.num_few_shots > 0
            else []
        )

    def _load_text_file(self, file_path: Path) -> str:
        """Loads content from a text file."""
        logger.debug(f"Loading text file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return (
                    f.read()
                )  # Keep leading/trailing whitespace if Jinja templates handle it
        except FileNotFoundError:
            # Error already logged by caller if it's critical (e.g. system prompt)
            # This helper can just raise it.
            raise
        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            raise

    def _load_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Loads few-shot examples from a JSONL file."""
        if not self.few_shot_examples_path or not self.few_shot_examples_path.exists():
            logger.warning(
                f"Few-shot examples file not found or not specified: {self.few_shot_examples_path}"
            )
            return []

        logger.info(f"Loading few-shot examples from: {self.few_shot_examples_path}")
        examples = []
        try:
            with open(self.few_shot_examples_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        if line.strip():
                            example = json.loads(line)
                            # Validate that the context key exists and is a dictionary for template rendering
                            if (
                                self.few_shot_context_key not in example
                                or not isinstance(
                                    example[self.few_shot_context_key], dict
                                )
                            ):
                                logger.warning(
                                    f"Skipping few-shot example on line {line_num} in {self.few_shot_examples_path}: "
                                    f"'{self.few_shot_context_key}' is missing or not a dictionary."
                                )
                                continue
                            if self.few_shot_assistant_key not in example:
                                logger.warning(
                                    f"Skipping few-shot example on line {line_num} in {self.few_shot_examples_path}: "
                                    f"'{self.few_shot_assistant_key}' is missing."
                                )
                                continue
                            examples.append(example)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON on line {line_num} in {self.few_shot_examples_path}: {e}"
                        )

            if examples:
                logger.info(
                    f"Successfully loaded {len(examples)} valid few-shot examples."
                )
            else:
                logger.warning(
                    f"No valid few-shot examples loaded from {self.few_shot_examples_path}"
                )
            return examples
        except Exception as e:
            logger.error(
                f"Error reading few-shot examples file {self.few_shot_examples_path}: {e}"
            )
            return []  # Return empty list on error

    def create_messages(self, template_context: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Formats the prompt messages list using Jinja2 templates, including few-shot examples.
        `template_context` is the dictionary of variables from the main data row.
        """
        messages = []

        # 1. System Prompt
        # Render system prompt string as a template using the main template_context
        try:
            rendered_system_prompt = self.jinja_env.from_string(
                self.system_prompt_str
            ).render(template_context)
            messages.append({"role": "system", "content": rendered_system_prompt})
        except Exception as e:
            logger.error(
                f"Error rendering system prompt: {e}. Using raw system prompt string."
            )
            messages.append({"role": "system", "content": self.system_prompt_str})

        # 2. Few-Shot Examples
        if self.num_few_shots > 0 and self.few_shot_examples:
            k = min(self.num_few_shots, len(self.few_shot_examples))
            selected_examples = random.sample(self.few_shot_examples, k)
            logger.debug(f"Including {k} few-shot examples.")

            for example in selected_examples:
                # The context for rendering the user_template for a few-shot example
                # comes from the 'context' field of the example itself.
                few_shot_render_context = example.get(self.few_shot_context_key)
                assistant_response = example.get(self.few_shot_assistant_key, "")

                if (
                    few_shot_render_context is None
                ):  # Should have been caught by loader, but double check
                    logger.warning(
                        f"Skipping few-shot example due to missing context field: {example.get(self.few_shot_id_key, 'Unknown ID')}"
                    )
                    continue

                try:
                    # Render the main user_template using the few-shot example's specific context
                    rendered_fs_user_prompt = self.user_template.render(
                        few_shot_render_context
                    )
                    messages.append(
                        {"role": "user", "content": rendered_fs_user_prompt}
                    )
                    messages.append(
                        {"role": "assistant", "content": assistant_response}
                    )
                except Exception as e:
                    logger.error(
                        f"Error rendering few-shot example user prompt (ID: {example.get(self.few_shot_id_key, 'Unknown ID')}): {e}"
                    )
                    # Optionally skip this example or add a placeholder

        # 3. Main User Prompt
        try:
            rendered_user_prompt = self.user_template.render(template_context)
            messages.append({"role": "user", "content": rendered_user_prompt})
        except Exception as e:
            logger.error(
                f"Error rendering main user prompt: {e}. Appending empty user message."
            )
            raise e

        return messages
