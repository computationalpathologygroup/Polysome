import json
import re
import logging
from typing import Optional, Dict, Any, Union, List
from .json_parsing_pipeline import create_default_json_parser

# Configure logging for potential issues
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_and_parse_json(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Extracts JSON content from a string using a comprehensive parsing pipeline.

    This function now uses the advanced json_parsing_pipeline which handles:
    1. Plain JSON strings
    2. JSON strings embedded within Markdown code fences (```json ... ``` or ``` ... ```)
    3. Python dict format conversion (single quotes, True/False/None values)
    4. Truncated JSON completion
    5. Various quote escaping scenarios
    6. Control character handling
    7. Leading/trailing whitespace and language identifiers

    Args:
        text: The input string potentially containing JSON.

    Returns:
        A dictionary or list parsed from the JSON content if successful and the root JSON object
        is a dictionary or array. Returns None if no valid JSON can be extracted or parsed.
    """
    if not isinstance(text, str):
        logging.warning("Input is not a string. Type: %s", type(text))
        return None

    # Use the comprehensive JSON parsing pipeline
    parser = create_default_json_parser()
    result = parser.parse(text)
    
    if result.success and result.data is not None:
        # Ensure the parsed result is a dictionary or list, as valid JSON types
        if isinstance(result.data, (dict, list)):
            logging.debug("Successfully parsed JSON into a %s using comprehensive pipeline.", type(result.data).__name__)
            return result.data
        else:
            logging.warning(
                "Parsed JSON is not a dictionary or list (Type: %s). Input: '%.100s...'",
                type(result.data).__name__,
                text[:100],
            )
            return None
    else:
        logging.warning(
            "Comprehensive JSON parsing failed: %s. Input: '%.100s...'", 
            result.error if result.error else "Unknown error", 
            text[:100]
        )
        return None
