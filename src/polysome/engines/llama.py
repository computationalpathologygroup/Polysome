import logging
from typing import List, Dict, Any, cast, TYPE_CHECKING
from polysome.engines.base import Engine  # Assuming this is the correct path

if TYPE_CHECKING:
    from llama_cpp import Llama, CreateChatCompletionResponse

try:
    from llama_cpp import Llama, CreateChatCompletionResponse

    LLAMA_CPP_AVAILABLE = True
except ImportError as e:
    LLAMA_CPP_AVAILABLE = False
    logging.debug(f"llama-cpp-python not available: {e}")


class LlamaCppEngine(Engine):
    """
    Inference engine using the llama-cpp-python library for GGUF models.
    Relies on llama-cpp-python's internal chat templating via create_chat_completion.
    This version focuses on non-streaming generation to return a single string.
    """

    AVAILABLE = LLAMA_CPP_AVAILABLE

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python library not installed")
        super().__init__(model_name)
        self.llm = None

        # --- GPU Detection and Fallback Logic ---
        original_gpu_layers = kwargs.get('n_gpu_layers', 0)
        
        # Log system information for debugging
        try:
            import torch
            logging.info(f"PyTorch CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA device count: {torch.cuda.device_count()}")
                logging.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        except ImportError:
            logging.info("PyTorch not available for CUDA detection")
        
        # Try GPU first if requested
        if original_gpu_layers != 0:
            try:
                logging.info(f"Attempting GPU initialization with {original_gpu_layers} layers...")
                logging.info(f"Full GPU kwargs: {kwargs}")
                self.llm = Llama(model_path=model_name, **kwargs)
                logging.info(f"✅ Llama.cpp GPU initialization successful: {model_name}")
                logging.info(f"GPU layers: {original_gpu_layers}")
                
                # Verify GPU is actually being used by checking internal state
                if hasattr(self.llm, '_model') and hasattr(self.llm._model, 'n_gpu_layers'):
                    actual_gpu_layers = self.llm._model.n_gpu_layers
                    logging.info(f"Verified GPU layers in model: {actual_gpu_layers}")
                    if actual_gpu_layers == 0:
                        logging.warning("⚠️ Model reports 0 GPU layers despite GPU initialization!")
                        
                return
            except Exception as gpu_error:
                logging.error(f"❌ GPU initialization failed: {gpu_error}")
                logging.error(f"GPU error type: {type(gpu_error).__name__}")
                import traceback
                logging.error(f"GPU error traceback: {traceback.format_exc()}")
                logging.info("Will NOT fall back to CPU mode - GPU should work in Docker")
                raise  # Don't fall back, let it fail so we can debug
                
        # Only reach here if n_gpu_layers was 0 (explicit CPU mode)
        try:
            logging.info("Initializing in explicit CPU mode (n_gpu_layers=0)...")
            self.llm = Llama(model_path=model_name, **kwargs)
            logging.info(f"✅ Llama.cpp CPU initialization successful: {model_name}")
                
        except Exception as e:
            logging.error(f"❌ Failed to initialize Llama.cpp in CPU mode: {e}")
            raise

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        logging.debug(f"Generating text for {len(messages)} messages.")
        if not self.llm:
            raise RuntimeError("Llama.cpp LLM object not initialized.")

        try:
            typed_messages = cast(List, messages)

            # --- Generate ---
            output = cast(
                CreateChatCompletionResponse,
                self.llm.create_chat_completion(messages=typed_messages, **kwargs),
            )
            # --- Extract the response ---
            if output and output.get("choices"):
                choice = output["choices"][0]
                message = choice.get("message")
                if message:
                    if (content := message.get("content")) is not None:
                        logging.debug(f"Llama.cpp Generated text: {content}")
                        return str(content).strip()
                    else:
                        logging.error("Llama.cpp message content is None.")
                        return "Error: Llama.cpp returned message with None content."
                else:
                    logging.error("Llama.cpp choice missing 'message' field.")
                    return "Error: Llama.cpp returned unexpected choice format."
            else:
                logging.error(f"Llama.cpp returned no/empty choices: {output}")
                return "Error: Llama.cpp generation failed."

        except Exception as e:
            logging.exception(f"Error during Llama.cpp generation: {e}")
            logging.debug(f"Problematic messages structure passed: {messages}")
            return f"Error generating text with Llama.cpp: {e}"

    def generate_text_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generates text for a batch of message lists using simulated batching.
        LlamaCpp does not support native batching, so this processes items sequentially.
        
        Args:
            messages_batch: A list of message lists, where each message list 
                          contains dictionaries with 'role' and 'content'.
            **kwargs: Generation parameters passed to create_chat_completion().
                
        Returns:
            A list of generated text strings.
        """
        logging.warning(
            f"LlamaCpp engine does not support native batching. "
            f"Processing {len(messages_batch)} items sequentially."
        )
        
        results = []
        for i, messages in enumerate(messages_batch):
            logging.debug(f"Processing batch item {i+1}/{len(messages_batch)}")
            result = self.generate_text(messages, **kwargs)
            results.append(result)
        
        logging.debug(f"LlamaCpp simulated batch processing completed for {len(results)} items")
        return results

    def supports_native_batching(self) -> bool:
        """
        LlamaCpp does not support native batch processing.
        """
        return False

    def unload_model(self) -> None:
        """
        Unloads the llama.cpp model from memory to free up GPU/CPU resources.
        """
        if self.llm is not None:
            logging.info(f"Unloading llama.cpp model: {self.model_name}")
            try:
                # llama-cpp-python doesn't have an explicit unload method,
                # but we can delete the object and force garbage collection
                del self.llm
                self.llm = None
                
                # Force garbage collection to free memory immediately
                import gc
                gc.collect()
                
                # If CUDA is available, clear CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logging.info("Cleared CUDA cache after model unload")
                except ImportError:
                    pass  # torch not available, skip CUDA cache clearing
                    
                logging.info(f"Successfully unloaded llama.cpp model: {self.model_name}")
            except Exception as e:
                logging.error(f"Error during model unload: {e}")
        else:
            logging.debug(f"Model {self.model_name} was not loaded, nothing to unload")


if __name__ == "__main__":
    print("\n" + "-" * 10, "Llama.cpp Example", "-" * 10)
    import os
    from pathlib import Path

    relative_model_path = os.path.join(os.getenv("MODEL_PATH", "./models"), "gemma-3-27b-it-q4_0.gguf")

    if not os.path.exists(relative_model_path):
        print(f"Llama.cpp model not found at: {relative_model_path}")
        print(
            "Skipping Llama.cpp example. Please download a GGUF model and update the path."
        )
    else:
        kwargs = {
            "chat_format": "gemma",  # Set to None for auto-detect
            "n_gpu_layers": -1,  # Use -1 for max GPU offload, 0 for CPU
            "n_ctx": 4096,  # Example context size, adjust as needed
            "verbose": False,  # Set to True for detailed llama.cpp logging
        }
        # --- Initialize the Engine ---
        llama_engine = LlamaCppEngine(
            model_name=relative_model_path,
            **kwargs,
        )

        # --- Test Case 1: Standard Chat ---
        llama_messages_pirate = [
            {
                "role": "system",
                "content": "You are a pirate assistant. Respond like a pirate.",
            },
            {
                "role": "user",
                "content": "What be the weather forecast today, matey?",
            },
        ]
        print(f"Prompt (Pirate): {llama_messages_pirate}")
        generation_kwargs = {
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
        }
        standard_response = llama_engine.generate_text(
            messages=llama_messages_pirate,
            **generation_kwargs,
        )
        print(f"\nLlama.cpp Standard Response:\n{standard_response}")
