from pathlib import Path
from typing import Dict, Callable, Any, List
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class DataFileLoader:
    def __init__(self, input_data_path: Path, primary_key: str):
        """
        Initializes the DataFileLoader.

        Args:
            input_data_path: Path to the input data file (.csv, .xls, .xlsx, .jsonl).
            primary_key: The name of the column/key to use as the primary identifier.
        """
        self.input_data_path = input_data_path
        self.primary_key = primary_key

        # Updated Callable signature: no longer takes List[str]
        self._loaders: Dict[str, Callable[[Path, str], Dict[str, Dict[str, Any]]]] = {
            ".csv": self._load_input_data_csv,
            ".xls": self._load_input_data_excel,
            ".xlsx": self._load_input_data_excel,
            ".json": self._load_input_data_json,
            ".jsonl": self._load_input_data_jsonl,
            # Add more loaders here in the future
        }

    def load_input_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load input data from the specified file, using the primary key.

        Returns:
            A dictionary where keys are the primary key values (as strings)
            and values are dictionaries representing the rest of the data
            for that key.
        """
        suffix = self.input_data_path.suffix.lower()
        if suffix in self._loaders:
            # Call loader without data_attributes
            return self._loaders[suffix](self.input_data_path, self.primary_key)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_input_data_csv(
        self, input_data_path: Path, primary_key_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load input data from a CSV file."""
        try:
            data = pd.read_csv(input_data_path)
            # Check only for primary key column
            if primary_key_name not in data.columns:
                raise ValueError(
                    f"Missing primary key column in CSV: {primary_key_name}"
                )

            output_data = {}
            # Get all column names except the primary key
            attribute_columns = [col for col in data.columns if col != primary_key_name]

            primary_key_series = data[primary_key_name].astype(str)
            for index, key in primary_key_series.items():
                # Create a dictionary containing all other columns for this row
                row_data = {attr: data.at[index, attr] for attr in attribute_columns}
                if key in output_data:
                    logger.warning(
                        f"Duplicate primary key '{key}' found in CSV '{input_data_path}'. Overwriting previous value."
                    )
                output_data[key] = row_data
            return output_data
        except FileNotFoundError:
            logger.error(f"Input CSV file not found: {input_data_path}")
            raise FileNotFoundError(f"Input file not found: {input_data_path}")
        except Exception as e:
            logger.exception(f"Error loading CSV data from {input_data_path}")
            raise Exception(f"Error loading CSV data: {e}")

    def _load_input_data_excel(
        self, input_data_path: Path, primary_key_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load input data from an Excel file."""
        try:
            data = pd.read_excel(input_data_path)
            # Check only for primary key column
            if primary_key_name not in data.columns:
                raise ValueError(
                    f"Missing primary key column in Excel: {primary_key_name}"
                )

            output_data = {}
            attribute_columns = [col for col in data.columns]

            primary_key_series = data[primary_key_name].astype(str)
            for index, key in primary_key_series.items():
                # Create a dictionary containing all other columns for this row
                row_data = {attr: data.at[index, attr] for attr in attribute_columns}
                if key in output_data:
                    logger.warning(
                        f"Duplicate primary key '{key}' found in Excel '{input_data_path}'. Overwriting previous value."
                    )
                output_data[key] = row_data
            return output_data
        except FileNotFoundError:
            logger.error(f"Input Excel file not found: {input_data_path}")
            raise FileNotFoundError(f"Input file not found: {input_data_path}")
        except Exception as e:
            logger.exception(f"Error loading Excel data from {input_data_path}")
            raise Exception(f"Error loading Excel data: {e}")

    def _process_json_record(
        self,
        record: Any,
        index: int,
        primary_key_name: str,
        input_data_path: Path,
        loaded_data: Dict[str, Dict[str, Any]],
        context: str = "element",
    ) -> None:
        """
        Process a single JSON record and add it to loaded_data if valid.

        Args:
            record: The JSON record to process
            index: The index/line number for logging
            primary_key_name: The primary key field name
            input_data_path: Path to the input file (for logging)
            loaded_data: Dictionary to add the processed record to
            context: Context string for logging ("element" for JSON arrays, "line" for JSONL)
        """
        if not isinstance(record, dict):
            logger.warning(
                f"Skipping non-dict {context} {index} in {input_data_path}: {str(record)[:100]}..."
            )
            return

        if primary_key_name not in record:
            logger.warning(
                f"Skipping {context} {index} in {input_data_path}: missing primary key '{primary_key_name}'. {context.capitalize()}: {str(record)[:100]}..."
            )
            return

        key = str(record[primary_key_name])  # Ensure key is string

        # The value will be the entire JSON object (record)
        # Alternatively, could remove the primary key:
        # value_record = {k: v for k, v in record.items() if k != primary_key_name}
        value_record = record

        if key in loaded_data:
            logger.warning(
                f"Duplicate primary key '{key}' found at {context} {index} in {input_data_path}. Overwriting previous value."
            )
        loaded_data[key] = value_record

    def _load_input_data_json(
        self, input_data_path: Path, primary_key_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load data from a JSON file containing an array of JSON objects."""
        loaded_data = {}
        try:
            with open(input_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure the loaded data is a list
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected JSON file to contain an array, but got {type(data).__name__}: {input_data_path}"
                )

            for i, record in enumerate(data):
                self._process_json_record(
                    record, i, primary_key_name, input_data_path, loaded_data, "element"
                )

        except FileNotFoundError:
            logger.error(f"Input JSON file not found: {input_data_path}")
            raise FileNotFoundError(f"Input file not found: {input_data_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {input_data_path}: {e}")
            raise Exception(f"Invalid JSON format: {e}")
        except IOError as e:
            logger.exception(f"IOError loading JSON data from {input_data_path}: {e}")
            raise Exception(f"IOError loading JSON data: {e}")
        except Exception as e:
            logger.exception(
                f"Unexpected error loading JSON data from {input_data_path}: {e}"
            )
            raise Exception(f"Error loading JSON data: {e}")

        return loaded_data

    def _load_input_data_jsonl(
        self, input_data_path: Path, primary_key_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """Load data from a JSONL file, using primary_key_name to key the records."""
        loaded_data = {}
        try:
            with open(input_data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):  # Start from 1 for line numbers
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    try:
                        record = json.loads(line)
                        self._process_json_record(
                            record,
                            i,
                            primary_key_name,
                            input_data_path,
                            loaded_data,
                            "line",
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping invalid JSON line {i} in {input_data_path}: {line[:100]}..."
                        )
                    except Exception as inner_e:
                        logger.warning(
                            f"Error processing line {i} in {input_data_path}: {inner_e}. Line: {line[:100]}...",
                            exc_info=True,  # Set to True for full traceback in logs
                        )

        except FileNotFoundError:
            logger.error(f"Input JSONL file not found: {input_data_path}")
            raise FileNotFoundError(f"Input file not found: {input_data_path}")
        except IOError as e:
            logger.exception(f"IOError loading JSONL data from {input_data_path}: {e}")
            raise Exception(f"IOError loading JSONL data: {e}")
        except Exception as e:
            logger.exception(
                f"Unexpected error loading JSONL data from {input_data_path}: {e}"
            )
            raise Exception(f"Error loading JSONL data: {e}")

        return loaded_data
