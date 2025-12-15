#!/usr/bin/env python3
"""
Unified JSON parsing pipeline with isolated, testable steps.

This module provides a flexible and extensible pipeline for parsing JSON
from various formats including markdown blocks, Python dict format, and
regular JSON strings.
"""

import json
import re
import ast
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of a parsing operation."""

    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    original_input: Optional[str] = None


class ParseStep(ABC):
    """Abstract base class for parsing pipeline steps."""

    @abstractmethod
    def process(self, input_data: str) -> str:
        """
        Process the input data and return modified data.

        Args:
            input_data: Input string to process

        Returns:
            Processed string
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this parsing step."""
        pass


class MarkdownRemovalStep(ParseStep):
    """Remove markdown code blocks and extract JSON content."""

    def process(self, input_data: str) -> str:
        """
        Remove markdown code blocks from text and extract the content.

        Handles patterns like:
        ```json
        {"key": "value"}
        ```

        Args:
            input_data: Input text that may contain markdown blocks

        Returns:
            Text with markdown blocks removed and content extracted
        """
        text = input_data.strip()

        # Pattern 1: ```json ... ``` blocks
        json_block_pattern = r"```json\s*\n?(.*?)\n?```"
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: ``` ... ``` blocks (without language specifier)
        generic_block_pattern = r"```\s*\n?(.*?)\n?```"
        match = re.search(generic_block_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Only return if it looks like JSON (starts with { or [)
            if content.startswith(("{", "[")):
                return content

        # Pattern 3: Multiple consecutive backticks
        multi_backtick_pattern = r"`{3,}json\s*\n?(.*?)\n?`{3,}"
        match = re.search(multi_backtick_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # No markdown blocks found, return original text
        return text

    def get_name(self) -> str:
        return "MarkdownRemoval"


class PythonDictFormatStep(ParseStep):
    """Convert Python dict/list format to valid JSON."""

    def process(self, input_data: str) -> str:
        """
        Convert Python dict/list format to valid JSON.

        Handles cases like:
        - Single quotes to double quotes: {'key': 'value'} -> {"key": "value"}
        - Python boolean/null values: True/False/None -> true/false/null

        Args:
            input_data: Input text in Python format

        Returns:
            Text converted to valid JSON format
        """
        text = input_data.strip()

        # Only apply this fix if the text looks like Python dict/list format
        if not (text.startswith(("{", "[")) and text.endswith(("}", "]"))):
            return text

        try:
            # Try using ast.literal_eval to safely parse Python literals
            parsed = ast.literal_eval(text)
            # If successful, convert back to JSON
            return json.dumps(parsed, ensure_ascii=False)
        except (ValueError, SyntaxError):
            # Fallback to manual replacement if ast fails
            result = text

            # Replace Python boolean and null values
            result = re.sub(r"\bTrue\b", "true", result)
            result = re.sub(r"\bFalse\b", "false", result)
            result = re.sub(r"\bNone\b", "null", result)

            # Simple single quote to double quote replacement
            result = result.replace("'", '"')

            return result

    def get_name(self) -> str:
        return "PythonDictFormat"


class TruncatedJSONFixStep(ParseStep):
    """Fix truncated JSON by completing incomplete structures."""

    def process(self, input_data: str) -> str:
        """
        Fix truncated JSON by analyzing structure and completing missing parts.

        Args:
            input_data: Potentially truncated JSON string

        Returns:
            Completed JSON string with proper closing brackets/braces
        """
        text = input_data.strip()

        # Handle empty input
        if not text:
            return text

        # Check if already complete JSON
        if self._is_complete_json(text):
            return text

        # Remove explicit truncation indicators
        text = self._remove_truncation_indicators(text)

        # Analyze structure and find truncation points
        structure_info = self._analyze_structure(text)

        # Clean up incomplete content
        cleaned_text = self._cleanup_incomplete_content(text, structure_info)

        # Add missing structural closures
        completed_text = self._add_missing_closures(cleaned_text, structure_info)

        return completed_text

    def get_name(self) -> str:
        return "TruncatedJSONFix"

    def _is_complete_json(self, text: str) -> bool:
        """Check if the JSON is already complete and valid."""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _remove_truncation_indicators(self, text: str) -> str:
        """Remove explicit truncation indicators like '...'."""
        # Remove trailing ellipsis
        if text.endswith("..."):
            text = text[:-3].rstrip()

        # Remove ellipsis within values (but be careful not to break valid content)
        # Only remove if it's at the end of a potential string value
        text = re.sub(r'\.\.\."\s*[,}\]]?\s*$', '"', text)
        text = re.sub(r'\.\.\."\s*$', '"', text)

        return text

    def _analyze_structure(self, text: str) -> dict:
        """
        Analyze JSON structure to understand nesting and truncation points.

        Returns:
            Dictionary with structure analysis information
        """
        brace_depth = 0
        bracket_depth = 0
        in_string = False
        escape_next = False
        last_valid_pos = 0
        string_start_pos = None

        structure_info = {
            "final_brace_depth": 0,
            "final_bracket_depth": 0,
            "in_string_at_end": False,
            "last_valid_pos": 0,
            "string_start_pos": None,
            "root_type": None,
            "truncation_type": None,
        }

        # Determine root type
        if text.strip().startswith("{"):
            structure_info["root_type"] = "object"
        elif text.strip().startswith("["):
            structure_info["root_type"] = "array"
        else:
            structure_info["root_type"] = "primitive"
            return structure_info

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                if not in_string:
                    in_string = True
                    string_start_pos = i
                else:
                    in_string = False
                    last_valid_pos = i + 1
                    string_start_pos = None
                continue

            # Skip structure tracking if we're inside a string
            if in_string:
                continue

            if char == "{":
                brace_depth += 1
            elif char == "}":
                brace_depth -= 1
                last_valid_pos = i + 1
            elif char == "[":
                bracket_depth += 1
            elif char == "]":
                bracket_depth -= 1
                last_valid_pos = i + 1
            elif char in ",:\n\t ":
                if char == ",":
                    last_valid_pos = i + 1

        structure_info.update(
            {
                "final_brace_depth": brace_depth,
                "final_bracket_depth": bracket_depth,
                "in_string_at_end": in_string,
                "last_valid_pos": last_valid_pos,
                "string_start_pos": string_start_pos,
            }
        )

        # Determine truncation type
        if in_string:
            structure_info["truncation_type"] = "string_value"
        elif brace_depth > 0 or bracket_depth > 0:
            structure_info["truncation_type"] = "missing_closure"
        else:
            structure_info["truncation_type"] = "content"

        return structure_info

    def _cleanup_incomplete_content(self, text: str, structure_info: dict) -> str:
        """Clean up incomplete content based on structure analysis."""

        # If we're in a string at the end, complete the string intelligently
        if (
            structure_info["in_string_at_end"]
            and structure_info["string_start_pos"] is not None
        ):
            string_start = structure_info["string_start_pos"]

            # Check if this is a value string (preceded by colon) or key string
            prefix = text[:string_start].rstrip()
            incomplete_content = text[string_start:]

            if prefix.endswith(":"):
                # This is an incomplete value string - close the quote to preserve content
                text = text + '"'
            else:
                # This might be an incomplete key - try to complete it intelligently
                # Remove the opening quote from incomplete content
                key_content = (
                    incomplete_content[1:]
                    if incomplete_content.startswith('"')
                    else incomplete_content
                )
                completed_key = self._complete_key(key_content)
                if completed_key:
                    # Complete the key with colon and empty string value
                    text = prefix + '"' + completed_key + '": ""'
                else:
                    # If we can't complete the key, still add colon and value
                    text = text + '": ""'

        # Clean up other incomplete patterns (but preserve content where possible)
        text = self._clean_incomplete_patterns(text, structure_info)

        return text.strip()

    def _complete_key(self, incomplete_key: str) -> Optional[str]:
        """
        Complete incomplete keys based on expected conversational format.

        Args:
            incomplete_key: Partial key string

        Returns:
            Completed key or None if can't be determined
        """
        if not incomplete_key:
            return None

        # Common conversational keys
        key_completions = {
            "u": "user",
            "us": "user",
            "use": "user",
            "user": "user",
            "a": "assistant",
            "as": "assistant",
            "ass": "assistant",
            "assi": "assistant",
            "assis": "assistant",
            "assist": "assistant",
            "assista": "assistant",
            "assistan": "assistant",
            "assistant": "assistant",
        }

        return key_completions.get(incomplete_key.lower())

    def _clean_incomplete_patterns(self, text: str, structure_info: dict) -> str:
        """Clean up incomplete patterns while preserving content where possible."""

        # Only remove truly incomplete patterns that can't be salvaged

        # Remove trailing colons without values (but keys without colons are handled above)
        text = re.sub(r":\s*$", "", text)

        # Remove trailing commas
        text = re.sub(r",\s*$", "", text)

        # Clean up commas before closing brackets
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        # For arrays, remove incomplete unquoted values at the end (but preserve quoted strings)
        if structure_info["root_type"] == "array":
            # Remove incomplete unquoted values at the end (but keep quoted strings)
            text = re.sub(r',\s*[^,\]"]*$', "", text)

        return text.strip()

    def _add_missing_closures(self, text: str, structure_info: dict) -> str:
        """Add missing closing brackets and braces in the correct order."""

        if not text.strip():
            # If we ended up with empty text, return appropriate empty structure
            if structure_info["root_type"] == "object":
                return "{}"
            elif structure_info["root_type"] == "array":
                return "[]"
            else:
                return text

        # For complex nested structures, we need to determine the correct closing order
        # by tracking the nesting stack
        closing_stack = self._determine_closing_sequence(text)

        result = text
        for close_char in closing_stack:
            result += close_char

        return result

    def _determine_closing_sequence(self, text: str) -> List[str]:
        """Determine the correct sequence of closing brackets/braces."""
        stack = []
        in_string = False
        escape_next = False

        for char in text:
            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" and stack and stack[-1] == "}":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "]":
                stack.pop()

        # Return the remaining closing characters in reverse order (LIFO)
        return list(reversed(stack))


class JSONParsingStep(ParseStep):
    """Parse JSON string using standard json.loads."""

    def process(self, input_data: str) -> str:
        """
        Parse JSON string and return it as formatted JSON.

        This step validates that the input is valid JSON by parsing it
        and then returns the formatted JSON string.

        Args:
            input_data: JSON string to parse

        Returns:
            Formatted JSON string

        Raises:
            json.JSONDecodeError: If the input is not valid JSON
        """
        if not input_data or input_data.strip() == "":
            return input_data

        # Parse and re-serialize to ensure valid JSON
        parsed_data = json.loads(input_data.strip())
        return json.dumps(parsed_data, ensure_ascii=False)

    def get_name(self) -> str:
        return "JSONParsing"


class JSONParsingPipeline:
    """Pipeline for parsing JSON with multiple preprocessing steps."""

    def __init__(self, steps: Optional[List[ParseStep]] = None):
        """
        Initialize the parsing pipeline.

        Args:
            steps: List of parsing steps to apply. If None, uses default steps.
        """
        if steps is None:
            self.steps = [
                MarkdownRemovalStep(),
                PythonDictFormatStep(),
                JSONParsingStep(),
            ]
        else:
            self.steps = steps

    def add_step(self, step: ParseStep, position: Optional[int] = None) -> None:
        """
        Add a parsing step to the pipeline.

        Args:
            step: ParseStep instance to add
            position: Position to insert step. If None, appends to end.
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)

    def remove_step(self, step_class: type) -> bool:
        """
        Remove the first step of given class from pipeline.

        Args:
            step_class: Class of step to remove

        Returns:
            True if step was found and removed, False otherwise
        """
        for i, step in enumerate(self.steps):
            if isinstance(step, step_class):
                del self.steps[i]
                return True
        return False

    def parse(self, input_data: str) -> ParseResult:
        """
        Parse input data through the pipeline.

        Args:
            input_data: Input string to parse

        Returns:
            ParseResult with success status and parsed data or error
        """
        if not input_data or (isinstance(input_data, str) and input_data.strip() == ""):
            return ParseResult(success=True, data=None, original_input=input_data)

        current_data = input_data
        original_input = input_data

        try:
            # Apply all steps except the final JSONParsingStep
            for step in self.steps[:-1]:
                current_data = step.process(current_data)

            # Apply final JSON parsing step
            if self.steps:
                final_step = self.steps[-1]
                if isinstance(final_step, JSONParsingStep):
                    # JSONParsingStep returns formatted JSON string, but we want parsed data
                    parsed_json = json.loads(current_data.strip())
                    return ParseResult(
                        success=True, data=parsed_json, original_input=original_input
                    )
                else:
                    # If final step is not JSONParsingStep, just process and try to parse
                    current_data = final_step.process(current_data)
                    parsed_json = json.loads(current_data.strip())
                    return ParseResult(
                        success=True, data=parsed_json, original_input=original_input
                    )
            else:
                # No steps, try direct parsing
                parsed_json = json.loads(current_data.strip())
                return ParseResult(
                    success=True, data=parsed_json, original_input=original_input
                )

        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                error=f"JSON parsing failed: {str(e)}",
                original_input=original_input,
            )
        except Exception as e:
            return ParseResult(
                success=False,
                error=f"Pipeline processing failed: {str(e)}",
                original_input=original_input,
            )

    def get_step_names(self) -> List[str]:
        """Get names of all steps in the pipeline."""
        return [step.get_name() for step in self.steps]


def create_default_pipeline() -> JSONParsingPipeline:
    """Create a pipeline with default parsing steps."""
    return JSONParsingPipeline()


class FallbackStrategy(ABC):
    """Abstract base class for JSON parsing fallback strategies."""

    @abstractmethod
    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """
        Attempt to parse the input data using this strategy.

        Args:
            input_data: Input string to parse

        Returns:
            ParseResult if successful, None if this strategy can't handle the input
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this fallback strategy."""
        pass


class PythonDictFallback(FallbackStrategy):
    """Fallback strategy for Python dict format conversion."""

    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """Try to parse input as Python dict format."""
        try:
            # Only attempt if it looks like Python dict/list format
            text = input_data.strip()
            if not (text.startswith(("{", "[")) and text.endswith(("}", "]"))):
                return None

            # Try ast.literal_eval approach
            import ast

            parsed = ast.literal_eval(text)
            return ParseResult(success=True, data=parsed, original_input=input_data)

        except (ValueError, SyntaxError):
            # Try manual conversion fallback
            try:
                result = text
                result = re.sub(r"\bTrue\b", "true", result)
                result = re.sub(r"\bFalse\b", "false", result)
                result = re.sub(r"\bNone\b", "null", result)
                result = result.replace("'", '"')

                parsed_data = json.loads(result)
                return ParseResult(
                    success=True, data=parsed_data, original_input=input_data
                )
            except json.JSONDecodeError:
                return None
        except Exception:
            return None

    def get_name(self) -> str:
        return "PythonDictFallback"


class TruncatedJSONFallback(FallbackStrategy):
    """Fallback strategy for truncated JSON completion."""

    def __init__(self):
        self.fixer = TruncatedJSONFixStep()

    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """Try to fix truncated JSON and parse."""
        try:
            # Check if input looks like potentially truncated JSON
            text = input_data.strip()
            if not text.startswith(("{", "[")):
                return None

            # Try to fix the truncated JSON
            fixed_json = self.fixer.process(text)
            parsed_data = json.loads(fixed_json)
            return ParseResult(
                success=True, data=parsed_data, original_input=input_data
            )

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def get_name(self) -> str:
        return "TruncatedJSONFallback"


class QuoteEscapingFallback(FallbackStrategy):
    """Fallback strategy for fixing unescaped quotes in JSON strings."""

    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """Try to fix unescaped quote issues and parse."""
        try:
            # Check if it looks like JSON structure
            text = input_data.strip()
            if not (text.startswith(("{", "["))):
                return None

            # Fix unescaped single quotes within string values
            fixed_text = self._fix_unescaped_quotes(text)

            # Try to parse the fixed text
            parsed_data = json.loads(fixed_text)
            return ParseResult(
                success=True, data=parsed_data, original_input=input_data
            )

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _fix_unescaped_quotes(self, text: str) -> str:
        """Fix unescaped single quotes within JSON string values."""
        result = []
        in_string = False
        escape_next = False

        i = 0
        while i < len(text):
            char = text[i]

            if escape_next:
                result.append(char)
                escape_next = False
            elif char == "\\":
                result.append(char)
                escape_next = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif char == "'" and in_string:
                # Escape single quotes within string values
                result.append("\\'")
            else:
                result.append(char)

            i += 1

        return "".join(result)

    def get_name(self) -> str:
        return "QuoteEscapingFallback"


class ControlCharacterFallback(FallbackStrategy):
    """Fallback strategy for fixing unescaped control characters in JSON strings."""

    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """Try to fix unescaped control character issues and parse."""
        try:
            # Check if it looks like JSON structure
            text = input_data.strip()
            if not (text.startswith(("{", "["))):
                return None

            # Fix unescaped control characters within string values
            fixed_text = self._fix_control_characters(text)

            # Try to parse the fixed text
            parsed_data = json.loads(fixed_text)
            return ParseResult(
                success=True, data=parsed_data, original_input=input_data
            )

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _fix_control_characters(self, text: str) -> str:
        """Fix unescaped control characters within JSON string values."""
        result = []
        in_string = False
        escape_next = False

        i = 0
        while i < len(text):
            char = text[i]

            if escape_next:
                result.append(char)
                escape_next = False
            elif char == "\\":
                result.append(char)
                escape_next = True
            elif char == '"':
                result.append(char)
                in_string = not in_string
            elif in_string and ord(char) < 32:
                # Handle control characters within string values
                if char == "\n":
                    result.append("\\n")
                elif char == "\r":
                    result.append("\\r")
                elif char == "\t":
                    result.append("\\t")
                elif char == "\b":
                    result.append("\\b")
                elif char == "\f":
                    result.append("\\f")
                else:
                    # For other control characters, use unicode escape
                    result.append(f"\\u{ord(char):04x}")
            else:
                result.append(char)

            i += 1

        return "".join(result)

    def get_name(self) -> str:
        return "ControlCharacterFallback"


class PropertyQuoteFallback(FallbackStrategy):
    """
    Enhanced fallback strategy for fixing mixed quote scenarios comprehensively.

    This strategy handles:
    1. Single quotes around property names
    2. Mixed quote types in string values
    3. Embedded apostrophes in string content
    4. Unicode characters in content
    5. Markdown-wrapped JSON
    """

    def try_parse(self, input_data: str) -> Optional[ParseResult]:
        """Try to parse mixed quote scenarios using multi-stage approach."""
        try:
            # Stage 1: Remove markdown blocks if present
            text = self._remove_markdown_blocks(input_data.strip())

            # Stage 2: Quick validation - must look like JSON structure
            if not (text.startswith(("{", "["))):
                return None

            # Stage 3: Apply comprehensive quote normalization
            fixed_text = self._normalize_quotes_comprehensively(text)

            # Stage 4: Try to parse the fixed text
            parsed_data = json.loads(fixed_text)
            return ParseResult(
                success=True, data=parsed_data, original_input=input_data
            )

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _normalize_quotes_comprehensively(self, text: str) -> str:
        """
        Comprehensive quote normalization using state machine approach.

        This handles the complex cases where we have mixed quote types
        and need to properly escape content while normalizing delimiters.
        """

        # For the most common pattern in our data: {'user': "...", 'assistant': '...'}
        # We'll use a targeted approach that's more reliable than generic parsing

        # Step 1: Handle the standard conversational format
        if self._is_conversational_format(text):
            return self._normalize_conversational_format(text)

        # Step 2: Fallback to generic normalization
        return self._generic_quote_normalization(text)

    def _is_conversational_format(self, text: str) -> bool:
        """Check if this looks like our standard conversational format."""
        # Look for the pattern: {optional_whitespace}'user'optional_whitespace:
        import re

        pattern = r'^\s*\{\s*[\'"]user[\'"]\s*:'
        return bool(re.match(pattern, text))

    def _normalize_conversational_format(self, text: str) -> str:
        """
        Normalize the specific conversational format we see in our data.

        Pattern: {'user': content, 'assistant': content}
        """
        import re

        # Use regex to extract the user and assistant content
        # This is more reliable than character-by-character parsing for this specific pattern

        # Pattern to match the full conversational structure
        pattern = r'^\s*\{\s*[\'"]user[\'"]\s*:\s*(.*),\s*[\'"]assistant[\'"]\s*:\s*(.*)\}\s*$'
        match = re.match(pattern, text, re.DOTALL)

        if not match:
            # Fallback to generic approach
            return self._generic_quote_normalization(text)

        user_content = match.group(1).strip()
        assistant_content = match.group(2).strip()

        # Normalize each content piece
        user_normalized = self._normalize_string_value(user_content)
        assistant_normalized = self._normalize_string_value(assistant_content)

        # Reconstruct with proper JSON format
        result = f'{{"user": {user_normalized}, "assistant": {assistant_normalized}}}'

        return result

    def _normalize_string_value(self, value: str) -> str:
        """
        Normalize a string value to proper JSON format.

        Handles cases where the value might be:
        - "double quoted content"
        - 'single quoted content'
        - 'content with apostrophe\'s'
        - "content with \"embedded quotes\""
        """

        value = value.strip()

        # Case 1: Already properly double-quoted
        if value.startswith('"') and value.endswith('"'):
            # Verify it's valid JSON
            try:
                json.loads(value)
                return value
            except:
                # Fix any issues within the double-quoted string
                content = value[1:-1]  # Remove outer quotes
                escaped_content = self._escape_for_json(content)
                return f'"{escaped_content}"'

        # Case 2: Single-quoted - need to convert to double-quoted
        elif value.startswith("'") and value.endswith("'"):
            content = value[1:-1]  # Remove outer single quotes
            escaped_content = self._escape_for_json(content)
            return f'"{escaped_content}"'

        # Case 3: Unquoted (shouldn't happen in our data, but handle gracefully)
        else:
            escaped_content = self._escape_for_json(value)
            return f'"{escaped_content}"'

    def _escape_for_json(self, content: str) -> str:
        """Escape content for inclusion in a JSON string."""

        # Escape backslashes first (to avoid double-escaping)
        content = content.replace("\\", "\\\\")

        # Escape double quotes
        content = content.replace('"', '\\"')

        # Handle control characters that are common in our data
        content = content.replace("\n", "\\n")
        content = content.replace("\r", "\\r")
        content = content.replace("\t", "\\t")

        return content

    def _generic_quote_normalization(self, text: str) -> str:
        """
        Generic quote normalization for non-conversational formats.

        This is a simpler approach for cases that don't match our
        standard conversational pattern.
        """
        import re

        # Replace Python boolean/null values first
        result = re.sub(r"\bTrue\b", "true", text)
        result = re.sub(r"\bFalse\b", "false", result)
        result = re.sub(r"\bNone\b", "null", result)

        # Simple single-to-double quote replacement
        # This works for simpler cases but may not handle complex embedded quotes
        result = result.replace("'", '"')

        return result

    def _remove_markdown_blocks(self, text: str) -> str:
        """Remove markdown code blocks from text."""
        import re

        # Patterns for various markdown block formats
        patterns = [
            r"```json\s*\n?(.*?)\n?```",  # Complete json blocks
            r"```JSON\s*\n?(.*?)\n?```",  # Case variations
            r"```Json\s*\n?(.*?)\n?```",
            r"```\s*\n?(.*?)\n?```",  # Generic blocks
            r"```json\s*\n?(.*?)$",  # Incomplete json blocks
            r"```JSON\s*\n?(.*?)$",
            r"```Json\s*\n?(.*?)$",
            r"```\s*\n?(.*?)$",  # Incomplete generic blocks
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                # Only return if it looks like JSON (starts with { or [)
                if content.startswith(("{", "[")):
                    return content

        return text.strip()

    def get_name(self) -> str:
        return "PropertyQuoteFallback"


class JSONParser:
    """
    JSON parser with preprocessing and fallback chain architecture.

    This parser first applies preprocessing (markdown removal), then tries
    standard JSON parsing, and finally applies fallback strategies in order
    until one succeeds.
    """

    def __init__(self, fallback_strategies: Optional[List[FallbackStrategy]] = None):
        """
        Initialize parser with fallback strategies.

        Args:
            fallback_strategies: List of fallback strategies. If None, uses defaults.
        """
        if fallback_strategies is None:
            self.fallback_strategies = [
                PythonDictFallback(),
                TruncatedJSONFallback(),
                QuoteEscapingFallback(),
                ControlCharacterFallback(),
                PropertyQuoteFallback(),
            ]
        else:
            self.fallback_strategies = fallback_strategies

    def parse(self, input_data: str) -> ParseResult:
        """
        Parse JSON string using preprocessing and fallback chain.

        Args:
            input_data: Input string to parse

        Returns:
            ParseResult with success status and parsed data or error
        """
        if not input_data or (isinstance(input_data, str) and input_data.strip() == ""):
            return ParseResult(success=True, data=None, original_input=input_data)

        original_input = input_data

        # Step 1: Always preprocess - remove markdown blocks
        cleaned_data = self._remove_markdown_blocks(input_data)

        # Step 2: Try standard JSON parsing first
        standard_result = self._try_standard_json(cleaned_data)
        if standard_result.success:
            return standard_result

        # Step 3: Try fallback strategies in order
        for strategy in self.fallback_strategies:
            fallback_result = strategy.try_parse(cleaned_data)
            if fallback_result and fallback_result.success:
                # Preserve original input
                fallback_result.original_input = original_input
                return fallback_result

        # All strategies failed
        return ParseResult(
            success=False,
            error=f"All parsing strategies failed. Standard JSON error: {standard_result.error}",
            original_input=original_input,
        )

    def _remove_markdown_blocks(self, text: str) -> str:
        """Remove markdown code blocks and extract JSON content."""
        text = text.strip()

        # Pattern 1: Complete ```json ... ``` blocks
        json_block_pattern = r"```json\s*\n?(.*?)\n?```"
        match = re.search(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 2: Complete ``` ... ``` blocks (without language specifier)
        generic_block_pattern = r"```\s*\n?(.*?)\n?```"
        match = re.search(generic_block_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Only return if it looks like JSON (starts with { or [)
            if content.startswith(("{", "[")):
                return content

        # Pattern 3: Complete multiple consecutive backticks
        multi_backtick_pattern = r"`{3,}json\s*\n?(.*?)\n?`{3,}"
        match = re.search(multi_backtick_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Pattern 4: Incomplete ```json blocks (for truncated content)
        incomplete_json_pattern = r"```json\s*\n?(.*?)$"
        match = re.search(incomplete_json_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Only return if it looks like JSON (starts with { or [)
            if content.startswith(("{", "[")):
                return content

        # Pattern 5: Incomplete ``` blocks (for truncated content)
        incomplete_generic_pattern = r"```\s*\n?(.*?)$"
        match = re.search(incomplete_generic_pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Only return if it looks like JSON (starts with { or [)
            if content.startswith(("{", "[")):
                return content

        # No markdown blocks found, return original text
        return text

    def _try_standard_json(self, input_data: str) -> ParseResult:
        """Try standard JSON parsing."""
        try:
            parsed_data = json.loads(input_data.strip())
            return ParseResult(
                success=True, data=parsed_data, original_input=input_data
            )
        except json.JSONDecodeError as e:
            return ParseResult(
                success=False,
                error=f"Standard JSON parsing failed: {str(e)}",
                original_input=input_data,
            )
        except Exception as e:
            return ParseResult(
                success=False,
                error=f"Unexpected error in standard JSON parsing: {str(e)}",
                original_input=input_data,
            )

    def add_fallback_strategy(
        self, strategy: FallbackStrategy, position: Optional[int] = None
    ) -> None:
        """Add a fallback strategy to the chain."""
        if position is None:
            self.fallback_strategies.append(strategy)
        else:
            self.fallback_strategies.insert(position, strategy)

    def get_strategy_names(self) -> List[str]:
        """Get names of all fallback strategies."""
        return [strategy.get_name() for strategy in self.fallback_strategies]


def create_default_json_parser() -> JSONParser:
    """Create a JSONParser with default fallback strategies."""
    return JSONParser()


def parse_json_string(input_data: str) -> ParseResult:
    """
    Convenience function to parse JSON string with fallback chain.

    Args:
        input_data: JSON string to parse

    Returns:
        ParseResult with parsed data or error
    """
    parser = create_default_json_parser()
    return parser.parse(input_data)


# Legacy pipeline support - keep for backward compatibility
def create_default_pipeline() -> JSONParsingPipeline:
    """Create a pipeline with default parsing steps (legacy)."""
    return JSONParsingPipeline()
