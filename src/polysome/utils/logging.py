# logging_setup.py (or add this to your main script / utils module)
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level=logging.INFO, log_file=None, log_dir=None, workflow_name=None):
    """
    Configures logging to stdout and optionally to a file.

    Args:
        level (int): The minimum logging level for the stdout handler
                     (e.g., logging.DEBUG, logging.INFO, logging.WARNING).
        log_file (str or Path, optional): Path to the log file. If provided,
                                          logs will also be written to this file.
                                          Defaults to None.
        log_dir (str or Path, optional): Directory to store log files. If provided,
                                         a timestamped log file will be created in this directory.
                                         Takes precedence over log_file if both are provided.
                                         Defaults to None.
        workflow_name (str, optional): Name of the workflow to include in the log filename.
                                      If not provided, defaults to "workflow".
                                      Defaults to None.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # Basic date format, customize as needed
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Get the root logger. Configuring the root logger affects all loggers
    # unless they have specific handlers defined.
    # You could also configure specific named loggers if needed.
    root_logger = logging.getLogger()

    # Set the overall minimum level for the logger itself.
    # Handlers can have higher levels. Setting root to DEBUG allows handlers
    # to decide what they want to capture (e.g., file DEBUG, console INFO)
    root_logger.setLevel(logging.DEBUG)

    # --- Clear existing handlers (important if setup might be called multiple times) ---
    # Be cautious with this in complex library scenarios, but useful for scripts
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # --- StreamHandler (stdout) ---
    stdout_handler = logging.StreamHandler(sys.stdout)
    # Set the level for what appears on console
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # --- FileHandler (optional) ---
    # Determine the log file path
    log_file_path = None
    
    if log_dir:
        # Create timestamped log file in the specified directory
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Use workflow name if provided, otherwise default to "workflow"
            name_part = workflow_name if workflow_name else "workflow"
            # Sanitize workflow name for filename (replace invalid characters)
            name_part = "".join(c if c.isalnum() or c in "._-" else "_" for c in name_part)
            
            log_file_path = log_dir_path / f"{name_part}_{timestamp}.log"
        except Exception as e:
            root_logger.error(
                f"Failed to create log directory {log_dir}: {e}", exc_info=True
            )
    elif log_file:
        # Use the explicitly provided log file path
        log_file_path = Path(log_file)
    
    if log_file_path:
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

            file_handler = logging.FileHandler(log_file_path, mode="a")  # Append mode
            # Example: Log everything (DEBUG level and above) to the file
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            # Log confirmation message using the configured logger
            root_logger.info(
                f"Logging detailed output (DEBUG level) to file: {log_file_path}"
            )
        except Exception as e:
            # Log error using the configured stdout handler if file setup fails
            root_logger.error(
                f"Failed to configure file logging to {log_file_path}: {e}", exc_info=True
            )

    root_logger.info(
        f"Logging configured. Console level: {logging.getLevelName(level)}"
    )
