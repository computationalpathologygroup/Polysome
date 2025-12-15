import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Base Engine Interface ---
class Engine(ABC):
    """Abstract base class for language model inference engines."""

    AVAILABLE = True

    @classmethod
    def is_available(cls) -> bool:
        return cls.AVAILABLE

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any):
        """
        Initializes the engine.
        Args:
            model_name: The identifier for the model (e.g., path or Hugging Face repo ID).
            **kwargs: Additional engine-specific configuration options.
        """
        self.model_name = model_name
        # Tokenizer SHOULD be initialized by subclasses if _apply_chat_template is used directly.
        # Some engines might use internal templating (like llama.cpp via create_chat_completion).
        self.tokenizer = None
        logging.info(f"Initializing engine for model: {self.model_name}")

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generates text based on a list of messages (chat history).
        Args:
            messages: A list of dictionaries, where each dictionary has 'role' and 'content'.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation-specific options for the specific engine.
        Returns:
            The generated text as a string.
        """
        pass

    def generate_text_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generates text for a batch of message lists.
        Default implementation processes each item individually.
        Subclasses should override this for true batch processing.
        
        Args:
            messages_batch: A list of message lists, where each message list 
                          contains dictionaries with 'role' and 'content'.
            **kwargs: Additional generation-specific options for the specific engine.
        Returns:
            A list of generated text strings.
        """
        results = []
        for messages in messages_batch:
            result = self.generate_text(messages, **kwargs)
            results.append(result)
        return results

    def supports_native_batching(self) -> bool:
        """
        Returns whether this engine supports native batch processing.
        Default is False - engines should override if they support batching.
        """
        return False

    def unload_model(self) -> None:
        """
        Unloads the model from memory to free up resources.
        This is a default implementation that can be overridden by subclasses.
        """
        logging.info(f"Unloading model: {self.model_name}")
        # Default implementation - subclasses should override this
        pass

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Applies the chat template to the messages list using the loaded tokenizer.
        Requires self.tokenizer to be initialized by the subclass.
        Args:
            messages: The list of message dictionaries.
        Returns:
            A formatted prompt string ready for the model.
        Raises:
            ValueError: If the tokenizer is not initialized.
            Exception: If the chat template application fails for other reasons.
        """
        if not self.tokenizer:
            # Fallback if tokenizer isn't available (e.g., llama.cpp without a loadable HF tokenizer)
            # This fallback is basic and might not match the model's expected format.
            logging.warning(
                f"{self.__class__.__name__} requires a tokenizer for _apply_chat_template, but none loaded. "
                "Using basic concatenation fallback (may not work well)."
            )
            # Simple concatenation (likely suboptimal)
            return (
                "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                + "\nassistant:"
            )

        # Check if the tokenizer actually has a template
        if not getattr(self.tokenizer, "chat_template", None) and not getattr(
            self.tokenizer, "default_chat_template", None
        ):
            logging.warning(
                f"Tokenizer for {self.model_name} does not have a default chat template according to HF Transformers. "
                f"Returning simple concatenation of messages (may not work well)."
            )
            # Fallback: simple concatenation (likely suboptimal)
            return (
                "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                + "\nassistant:"
            )

        try:
            # Use tokenize=False as we only need the formatted string here.
            prompt_string = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # Crucial for prompting the model's response
            )
            logging.debug(
                f"Applied chat template via HF, resulting prompt string (partial): {prompt_string[:500]}..."
            )
            return prompt_string
        except Exception as e:
            logging.error(
                f"Failed to apply chat template via HF for {self.model_name}: {e}"
            )
            logging.error(f"Problematic messages structure: {messages}")
            raise
