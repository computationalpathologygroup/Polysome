import csv
import logging
from pathlib import Path

# Get a logger instance specific to this module
# Using __name__ is standard practice and helps identify the source of logs
logger = logging.getLogger(__name__)


class IncrementalCsvWriter:
    """Manages incremental writing to a CSV file, with logging."""

    def __init__(self, output_path, fieldnames, delimiter=","):
        self.output_path = Path(output_path)
        self.fieldnames = fieldnames
        self.delimiter = delimiter
        self._csvfile = None
        self._writer = None
        logger.debug(f"Initialized IncrementalCsvWriter for path: {self.output_path}")

    def __enter__(self):
        """Open file, create writer, write header if needed."""
        logger.debug(f"Entering CSV writer context for {self.output_path}")
        try:
            # Ensure the output directory exists
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            file_exists = self.output_path.exists()
            # Check if file exists AND is empty (safer than just !file_exists)
            is_empty = not file_exists or self.output_path.stat().st_size == 0

            # Open in append mode
            self._csvfile = open(self.output_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._csvfile, fieldnames=self.fieldnames, delimiter=self.delimiter
            )

            if is_empty:
                self._writer.writeheader()
                logger.info(f"Opened new file with header: {self.output_path}")
            else:
                logger.info(f"Appending to existing file: {self.output_path}")

        except IOError as e:
            logger.error(
                f"Failed to open or prepare CSV file {self.output_path}: {e}",
                exc_info=True,
            )
            # Reraise the exception so the caller knows the 'with' block failed setup
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error during CSV writer setup for {self.output_path}",
                exc_info=True,
            )
            raise  # Reraise unexpected errors too
        return self  # Return the writer instance itself

    def write_row(self, data_dict):
        """Writes a single dictionary row."""
        if self._writer:
            try:
                self._writer.writerow(data_dict)
                # Flush to ensure data is written immediately
                if self._csvfile:
                    self._csvfile.flush()  # Ensure data is written immediately

                # Log primary key if possible and fieldnames exist, otherwise short dict repr
                log_key = (
                    data_dict.get(self.fieldnames[0], "N/A")
                    if self.fieldnames
                    else str(data_dict)[:50]
                )
                logger.debug(f"Wrote row with key/start: {log_key}")
            except Exception as e:
                # Log error but allow process to potentially continue
                logger.error(
                    f"Failed to write row to {self.output_path}: {e}. Data snippet: {str(data_dict)[:100]}...",
                    exc_info=True,
                )
                raise  # Uncomment if a single row write failure should stop everything
        else:
            # This case should ideally not happen if used correctly within 'with'
            logger.error("Attempted to write row, but CSV writer is not available.")
            raise IOError("CSV file is not open or writer not initialized.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the file and log any exceptions from the 'with' block."""
        logger.debug(f"Exiting CSV writer context for {self.output_path}")
        if self._csvfile:
            try:
                self._csvfile.close()
                logger.debug(f"Closed CSV file: {self.output_path}")
            except IOError as e:
                logger.error(
                    f"Error closing CSV file {self.output_path}: {e}", exc_info=True
                )
            finally:  # Ensure state is reset even if close fails
                self._csvfile = None
                self._writer = None

        if exc_type:
            # An exception occurred *within* the 'with' block (not during __enter__ or __exit__)
            logger.error(
                f"Exception occurred within CSV writer context: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb),
            )

        # Return False (or nothing) to indicate that if an exception occurred within the 'with' block,
        # it should be propagated (not suppressed). Return True to suppress.
        return False
