from polysome.engines.base import Engine
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
import logging
from typing import List, Dict, Any


class HuggingFaceEngine(Engine):
    """Inference engine using Hugging Face Transformers library."""

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Initializes the Hugging Face Transformers engine.
        Args:
            model_name: The identifier for the model (HF repo ID or path).
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_pretrained
                      (e.g., torch_dtype, device_map).
        """
        super().__init__(model_name)
        self.model = None  # Initialize model attribute
        self.tokenizer = None  # Initialize tokenizer attribute
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                logging.warning(
                    "Tokenizer does not have a pad token. Setting to eos_token."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = (
                    "left"  # Important for decoder-only models
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs,
            )
            logging.info(
                f"HF Model loaded successfully. Model device: {self.model.device}"
            )
        except Exception as e:
            logging.error(f"Failed to load HF model {self.model_name}: {e}")
            raise

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generates text using the loaded Hugging Face model.
        Args:
            messages: Chat history.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling p.
            do_sample: Whether to use sampling. Set to False for greedy decoding.
            **kwargs: Additional arguments passed to model.generate().
        Returns:
            The generated text string.
        """
        logging.debug(f"Generating text with HF engine for {len(messages)} messages.")
        if not self.tokenizer:
            logging.error("HF Tokenizer not initialized.")
            return "Error: HF Tokenizer not initialized."
        if not self.model:
            logging.error("HF Model not initialized.")
            return "Error: HF Model not initialized."

        try:
            # 1. Apply chat template using the centralized method
            prompt_string = self._apply_chat_template(messages)

            # 2. Tokenize the formatted prompt string
            inputs = self.tokenizer(
                prompt_string, return_tensors="pt", return_attention_mask=True
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]

            # 4. Generate
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **kwargs)

            # 5. Decode the newly generated tokens only
            generation = outputs[0][input_len:]
            decoded_output = self.tokenizer.decode(generation, skip_special_tokens=True)
            logging.debug(f"HF Generated text (decoded): {decoded_output}")
            return decoded_output.strip()

        except Exception as e:
            logging.exception(f"Error during HF text generation: {e}")
            return f"Error generating text with HF: {e}"

    def generate_text_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generates text for a batch of message lists using HuggingFace's native batching.
        
        Args:
            messages_batch: A list of message lists, where each message list 
                          contains dictionaries with 'role' and 'content'.
            **kwargs: Generation parameters passed to model.generate().
                
        Returns:
            A list of generated text strings.
        """
        logging.debug(f"Generating text batch with HF engine for {len(messages_batch)} items")
        
        if not self.tokenizer:
            logging.error("HF Tokenizer not initialized.")
            return [f"Error: HF Tokenizer not initialized."] * len(messages_batch)
        if not self.model:
            logging.error("HF Model not initialized.")
            return [f"Error: HF Model not initialized."] * len(messages_batch)

        try:
            # Apply chat template to all messages in the batch
            prompt_strings = []
            for messages in messages_batch:
                prompt_string = self._apply_chat_template(messages)
                prompt_strings.append(prompt_string)
            
            logging.debug(f"Applied chat templates to {len(prompt_strings)} prompts")

            # Tokenize all prompts in a batch
            inputs = self.tokenizer(
                prompt_strings, 
                return_tensors="pt", 
                return_attention_mask=True,
                padding=True,  # Pad to the same length for batching
                truncation=True  # Truncate if too long
            ).to(self.model.device)
            
            # Store original lengths for each item (before padding)
            original_lengths = []
            for prompt in prompt_strings:
                tokenized = self.tokenizer(prompt, return_tensors="pt")
                original_lengths.append(tokenized["input_ids"].shape[-1])
            
            logging.debug(f"Tokenized batch: {inputs['input_ids'].shape}, original lengths: {original_lengths}")

            # Generate for the entire batch
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **kwargs)

            # Decode each output individually
            results = []
            for i, output_ids in enumerate(outputs):
                # Extract only the newly generated tokens (after original input)
                original_length = original_lengths[i]
                generation = output_ids[original_length:]
                decoded_output = self.tokenizer.decode(generation, skip_special_tokens=True)
                results.append(decoded_output.strip())
            
            logging.debug(f"HF batch generation completed successfully for {len(results)} items")
            return results

        except Exception as e:
            logging.exception(f"Error during HF batch text generation: {e}")
            return [f"Error generating text with HF: {e}"] * len(messages_batch)

    def supports_native_batching(self) -> bool:
        """
        HuggingFace transformers supports native batch processing.
        """
        return True

    def unload_model(self) -> None:
        """
        Unloads the HuggingFace model from memory to free up GPU/CPU resources.
        """
        logging.info(f"Unloading HuggingFace model: {self.model_name}")
        try:
            # Move model to CPU before deletion to free GPU memory
            if self.model is not None:
                if hasattr(self.model, 'device') and 'cuda' in str(self.model.device):
                    logging.info("Moving model from GPU to CPU before deletion")
                    self.model = self.model.cpu()
                
                del self.model
                self.model = None

            # Clear tokenizer as well
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Force garbage collection to free memory immediately
            import gc
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache after model unload")
                
            logging.info(f"Successfully unloaded HuggingFace model: {self.model_name}")
        except Exception as e:
            logging.error(f"Error during HuggingFace model unload: {e}")


# --- Example Usage / Simple Test ---
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    from dotenv import load_dotenv

    load_dotenv()
    model_id = "google/gemma-3-4b-it"
    try:
        # You might want to force CPU for a simple test: device_map="cpu"
        engine = HuggingFaceEngine(model_id, device_map="auto")  # Or device_map="cpu"

        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        print(f"\n--- Testing {model_id} ---")
        print(f"Messages: {test_messages}")

        generated_text = engine.generate_text(
            messages=test_messages,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
        )

        print(f"\nGenerated Response:\n{generated_text}")
        print("\n--- Test Complete ---")

    except Exception as e:
        logging.exception(f"Error during simple test: {e}")
        print("\n--- Test Failed ---")
