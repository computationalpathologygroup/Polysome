import logging
from typing import List, Dict, Any, TYPE_CHECKING
from polysome.engines.base import Engine

if TYPE_CHECKING:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    logging.debug(f"vLLM library not available: {e}")

logger = logging.getLogger(__name__)


class VLLMEngine(Engine):
    """
    Inference engine using the vLLM library for efficient text generation.
    This engine provides optimized inference for text-only models.
    """

    AVAILABLE = VLLM_AVAILABLE

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Initializes the vLLM engine.
        
        Args:
            model_name: The identifier for the model (HF repo ID or path).
            **kwargs: Additional arguments passed to vLLM LLM class:
                - trust_remote_code: Whether to trust remote code (default: True)
                - tensor_parallel_size: Number of GPUs to use for tensor parallelism
                - gpu_memory_utilization: Fraction of GPU memory to use (default: 0.9)
                - max_model_len: Maximum model context length
                - dtype: Model data type (e.g., "auto", "float16", "bfloat16")
                - And other vLLM LLM parameters
        """
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM library not installed. Please install with: pip install vllm")
        
        super().__init__(model_name)
        self.llm = None
        
        # Set default values for common vLLM parameters
        vllm_kwargs = {
            "trust_remote_code": True,
            **kwargs
        }
        
        try:
            logger.info(f"Initializing vLLM engine for model: {model_name}")
            logger.info(f"vLLM initialization parameters: {vllm_kwargs}")
            
            self.llm = LLM(model=model_name, **vllm_kwargs)
            
            # Try to get tokenizer for chat template support
            try:
                self.tokenizer = self.llm.get_tokenizer()
                logger.info("Successfully loaded tokenizer for chat template support")
            except Exception as e:
                logger.warning(f"Could not load tokenizer from vLLM engine: {e}")
                self.tokenizer = None
            
            logger.info(f"vLLM engine successfully initialized for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine for model {model_name}: {e}")
            raise

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """
        Generates text using the loaded vLLM model.
        
        Args:
            messages: Chat history as list of message dictionaries.
            **kwargs: Generation parameters that will be converted to SamplingParams:
                - temperature: Sampling temperature (default: 1.0)
                - top_p: Nucleus sampling parameter (default: 1.0)
                - top_k: Top-k sampling parameter (default: -1, disabled)
                - max_tokens: Maximum number of tokens to generate (default: 16)
                - min_tokens: Minimum number of tokens to generate (default: 1)
                - repetition_penalty: Repetition penalty (default: 1.0)
                - presence_penalty: Presence penalty (default: 0.0)
                - frequency_penalty: Frequency penalty (default: 0.0)
                - And other SamplingParams parameters
                
        Returns:
            The generated text string.
        """
        logger.debug(f"Generating text with vLLM engine for {len(messages)} messages")
        
        if not self.llm:
            logger.error("vLLM model not initialized")
            return "Error: vLLM model not initialized"

        try:
            # Apply chat template using the centralized method from base class
            prompt_string = self._apply_chat_template(messages)
            logger.debug(f"Applied chat template, prompt length: {len(prompt_string)}")

            # Convert generation kwargs to SamplingParams
            # Set defaults for common parameters
            sampling_kwargs = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 16,
                **kwargs
            }
            
            # Handle max_new_tokens -> max_tokens conversion for compatibility
            if "max_new_tokens" in sampling_kwargs:
                sampling_kwargs["max_tokens"] = sampling_kwargs.pop("max_new_tokens")
            
            sampling_params = SamplingParams(**sampling_kwargs)
            logger.debug(f"Created SamplingParams: {sampling_params}")

            # Generate text using vLLM
            logger.debug("Starting vLLM generation...")
            outputs = self.llm.generate(prompts=[prompt_string], sampling_params=sampling_params)
            
            if not outputs:
                logger.error("vLLM returned no outputs")
                return "Error: vLLM generation failed - no outputs returned"
            
            output = outputs[0]
            if not output.outputs:
                logger.error("vLLM output has no completions")
                return "Error: vLLM generation failed - no completions in output"
            
            # Extract the generated text from the first completion
            generated_text = output.outputs[0].text
            logger.debug(f"vLLM generated text (length: {len(generated_text)}): {generated_text[:100]}...")
            
            return generated_text.strip()

        except Exception as e:
            logger.exception(f"Error during vLLM text generation: {e}")
            return f"Error generating text with vLLM: {e}"

    def generate_text_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Generates text for a batch of message lists using vLLM's native batching.
        
        Args:
            messages_batch: A list of message lists, where each message list 
                          contains dictionaries with 'role' and 'content'.
            **kwargs: Generation parameters that will be converted to SamplingParams.
                
        Returns:
            A list of generated text strings.
        """
        logger.debug(f"Generating text batch with vLLM engine for {len(messages_batch)} items")
        
        if not self.llm:
            logger.error("vLLM model not initialized")
            return [f"Error: vLLM model not initialized"] * len(messages_batch)

        try:
            # Apply chat template to all messages in the batch
            prompt_strings = []
            for messages in messages_batch:
                prompt_string = self._apply_chat_template(messages)
                prompt_strings.append(prompt_string)
            
            logger.debug(f"Applied chat templates to {len(prompt_strings)} prompts")

            # Convert generation kwargs to SamplingParams
            sampling_kwargs = {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 16,
                **kwargs
            }
            
            # Handle max_new_tokens -> max_tokens conversion for compatibility
            if "max_new_tokens" in sampling_kwargs:
                sampling_kwargs["max_tokens"] = sampling_kwargs.pop("max_new_tokens")
            
            sampling_params = SamplingParams(**sampling_kwargs)
            logger.debug(f"Created SamplingParams for batch: {sampling_params}")

            # Generate text using vLLM batch processing
            logger.debug(f"Starting vLLM batch generation for {len(prompt_strings)} prompts...")
            outputs = self.llm.generate(prompts=prompt_strings, sampling_params=sampling_params)
            
            if not outputs:
                logger.error("vLLM returned no outputs for batch")
                return [f"Error: vLLM batch generation failed - no outputs returned"] * len(messages_batch)
            
            if len(outputs) != len(messages_batch):
                logger.error(f"vLLM output count mismatch: expected {len(messages_batch)}, got {len(outputs)}")
                return [f"Error: vLLM batch generation failed - output count mismatch"] * len(messages_batch)
            
            # Extract generated text from all outputs
            results = []
            for i, output in enumerate(outputs):
                if not output.outputs:
                    logger.error(f"vLLM output {i} has no completions")
                    results.append(f"Error: vLLM generation failed for item {i} - no completions")
                else:
                    generated_text = output.outputs[0].text
                    results.append(generated_text.strip())
            
            logger.debug(f"vLLM batch generation completed successfully for {len(results)} items")
            return results

        except Exception as e:
            logger.exception(f"Error during vLLM batch text generation: {e}")
            return [f"Error generating text with vLLM: {e}"] * len(messages_batch)

    def supports_native_batching(self) -> bool:
        """
        vLLM supports native batch processing.
        """
        return True

    def unload_model(self) -> None:
        """
        Unloads the vLLM model from memory to free up GPU/CPU resources.
        """
        if self.llm is not None:
            logger.info(f"Unloading vLLM model: {self.model_name}")
            try:
                # vLLM doesn't have an explicit unload method,
                # but we can delete the object and force garbage collection
                del self.llm
                self.llm = None
                
                # Clear tokenizer as well
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                
                # Force garbage collection to free memory immediately
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("Cleared CUDA cache after vLLM model unload")
                except ImportError:
                    pass  # torch not available, skip CUDA cache clearing
                    
                logger.info(f"Successfully unloaded vLLM model: {self.model_name}")
            except Exception as e:
                logger.error(f"Error during vLLM model unload: {e}")
        else:
            logger.debug(f"vLLM model {self.model_name} was not loaded, nothing to unload")


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Example model - adjust as needed
    model_name = "microsoft/DialoGPT-medium"
    
    try:
        # Initialize the engine
        engine = VLLMEngine(
            model_name=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=0.8
        )

        # Test with simple messages
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        print(f"\n--- Testing vLLM engine with {model_name} ---")
        print(f"Messages: {test_messages}")

        generated_text = engine.generate_text(
            messages=test_messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        )

        print(f"\nGenerated Response:\n{generated_text}")
        print("\n--- Test Complete ---")
        
        # Clean up
        engine.unload_model()

    except Exception as e:
        logging.exception(f"Error during vLLM engine test: {e}")
        print("\n--- Test Failed ---")