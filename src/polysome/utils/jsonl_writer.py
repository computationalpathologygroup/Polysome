import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class IncrementalJsonlWriter:
    """Manages incremental writing of JSON objects (one per line) to a JSONL file."""

    def __init__(self, output_path: Path, mode: str = "a"):
        self.output_path = Path(output_path)
        self.mode = mode  # "a" for append, "w" for write (overwrite)
        self._file_handle = None
        logger.debug(f"Initialized IncrementalJsonlWriter for path: {self.output_path}, mode: {self.mode}")

    def __enter__(self):
        """Open file in specified mode (append or write)."""
        logger.debug(f"Entering JSONL writer context for {self.output_path}")
        try:
            # Ensure the output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            # Open in specified mode with UTF-8 encoding
            self._file_handle = open(self.output_path, self.mode, encoding="utf-8")
            mode_desc = "Appending to" if self.mode == "a" else "Writing to"
            logger.info(f"{mode_desc} JSONL file: {self.output_path}")

        except IOError as e:
            logger.error(
                f"Failed to open JSONL file {self.output_path} in mode '{self.mode}': {e}",
                exc_info=True,
            )
            raise  # Reraise so the caller knows setup failed
        except Exception as e:
            logger.exception(
                f"Unexpected error during JSONL writer setup for {self.output_path}",
                exc_info=True,
            )
            raise
        return self

    def write_row(self, data_dict: Dict[str, Any]):
        """Writes a single dictionary as a JSON line."""
        if self._file_handle:
            try:
                # Convert dict to JSON string
                json_string = json.dumps(
                    data_dict, ensure_ascii=False
                )  # ensure_ascii=False for broader char support
                # Write the JSON string followed by a newline
                self._file_handle.write(json_string + "\n")
                # Flush to ensure data is written immediately (important for resume)
                self._file_handle.flush()

                # Log based on 'id' if present, otherwise short dict repr
                log_key = data_dict.get("id", str(data_dict)[:50])
                logger.debug(f"Wrote JSONL row with id/start: {log_key}")

            except TypeError as e:
                logger.error(
                    f"Failed to serialize data to JSON for {self.output_path}: {e}. Data snippet: {str(data_dict)[:100]}...",
                    exc_info=True,
                )
                raise  # Serialization errors are critical
            except IOError as e:
                logger.error(
                    f"Failed to write row to {self.output_path}: {e}. Data snippet: {str(data_dict)[:100]}...",
                    exc_info=True,
                )
                raise  # IO errors during write are critical
            except Exception as e:
                logger.exception(
                    f"Unexpected error writing row to {self.output_path}: {e}. Data snippet: {str(data_dict)[:100]}...",
                    exc_info=True,
                )
                raise
        else:
            logger.error("Attempted to write JSONL row, but file is not open.")
            raise IOError("JSONL file is not open or writer not initialized.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file and log any exceptions from the 'with' block."""
        logger.debug(f"Exiting JSONL writer context for {self.output_path}")
        if self._file_handle:
            try:
                self._file_handle.close()
                logger.debug(f"Closed JSONL file: {self.output_path}")
            except IOError as e:
                logger.error(
                    f"Error closing JSONL file {self.output_path}: {e}", exc_info=True
                )
            finally:
                self._file_handle = None  # Ensure state is reset

        if exc_type:
            logger.error(
                f"Exception occurred within JSONL writer context: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )
        # Return False to propagate exceptions that occurred *within* the 'with' block
        return False
