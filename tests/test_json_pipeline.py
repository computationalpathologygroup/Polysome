#!/usr/bin/env python3
"""
Unit tests for the JSON parsing pipeline.

Tests each step individually and the complete pipeline integration.
"""

import unittest
import json
from polysome.utils.json_parsing_pipeline import (
    JSONParsingPipeline,
    MarkdownRemovalStep,
    PythonDictFormatStep,
    JSONParsingStep,
    ParseResult,
    create_default_pipeline,
    parse_json_string,
    JSONParser,
    create_default_json_parser,
    PythonDictFallback,
    TruncatedJSONFallback,
)


class TestMarkdownRemovalStep(unittest.TestCase):
    """Test cases for MarkdownRemovalStep."""

    def setUp(self):
        """Set up test fixtures."""
        self.step = MarkdownRemovalStep()

    def test_json_block_removal(self):
        """Test removal of ```json blocks."""
        test_cases = [
            # Basic json block
            ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
            # Json block without newlines
            ('```json{"key": "value"}```', '{"key": "value"}'),
            # Json block with extra whitespace
            ('```json\n\n  {"key": "value"}  \n\n```', '{"key": "value"}'),
            # Case insensitive
            ('```JSON\n{"key": "value"}\n```', '{"key": "value"}'),
            ('```Json\n{"key": "value"}\n```', '{"key": "value"}'),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_generic_block_removal(self):
        """Test removal of ``` blocks without language specifier."""
        test_cases = [
            # Generic block with JSON content
            ('```\n{"array": [1, 2, 3]}\n```', '{"array": [1, 2, 3]}'),
            ('```\n[{"item": 1}]\n```', '[{"item": 1}]'),
            # Generic block with non-JSON content (should not extract)
            ("```\nsome random text\n```", "```\nsome random text\n```"),
            ("```\nfunction test() {}\n```", "```\nfunction test() {}\n```"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_multi_backtick_removal(self):
        """Test removal of blocks with multiple backticks."""
        test_cases = [
            ('````json\n{"key": "value"}\n````', '{"key": "value"}'),
            ('`````json\n{"key": "value"}\n`````', '{"key": "value"}'),
            ('````JSON\n{"key": "value"}\n````', '{"key": "value"}'),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_no_markdown_blocks(self):
        """Test input without markdown blocks."""
        test_cases = [
            '{"key": "value"}',
            "[1, 2, 3]",
            "plain text without blocks",
            '{"nested": {"data": "test"}}',
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, input_text)

    def test_step_name(self):
        """Test step name."""
        self.assertEqual(self.step.get_name(), "MarkdownRemoval")


class TestPythonDictFormatStep(unittest.TestCase):
    """Test cases for PythonDictFormatStep."""

    def setUp(self):
        """Set up test fixtures."""
        self.step = PythonDictFormatStep()

    def test_python_dict_conversion(self):
        """Test conversion of Python dict format to JSON."""
        test_cases = [
            # Basic dict with single quotes
            ("{'key': 'value'}", '{"key": "value"}'),
            # Nested dict
            ("{'outer': {'inner': 'value'}}", '{"outer": {"inner": "value"}}'),
            # Dict with various data types
            (
                "{'text': 'hello', 'number': 42, 'list': [1, 2, 3]}",
                '{"text": "hello", "number": 42, "list": [1, 2, 3]}',
            ),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_python_list_conversion(self):
        """Test conversion of Python list format to JSON."""
        test_cases = [
            # Basic list
            ("['item1', 'item2']", '["item1", "item2"]'),
            # List with mixed types
            ("['text', 42, True, None]", '["text", 42, true, null]'),
            # List of dicts
            (
                "[{'key1': 'val1'}, {'key2': 'val2'}]",
                '[{"key1": "val1"}, {"key2": "val2"}]',
            ),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_python_boolean_conversion(self):
        """Test conversion of Python boolean values."""
        test_cases = [
            ("{'flag': True}", '{"flag": true}'),
            ("{'flag': False}", '{"flag": false}'),
            (
                "{'active': True, 'inactive': False}",
                '{"active": true, "inactive": false}',
            ),
            ("[True, False, True]", "[true, false, true]"),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_python_none_conversion(self):
        """Test conversion of Python None values."""
        test_cases = [
            ("{'value': None}", '{"value": null}'),
            ("[1, None, 3]", "[1, null, 3]"),
            ("{'a': None, 'b': 'not null'}", '{"a": null, "b": "not null"}'),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, expected)

    def test_already_valid_json(self):
        """Test that already valid JSON remains unchanged."""
        test_cases = [
            '{"key": "value"}',
            '["item1", "item2"]',
            '{"flag": true, "value": null}',
            "42",
            '"string"',
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, input_text)

    def test_non_dict_list_input(self):
        """Test that non-dict/list input remains unchanged."""
        test_cases = [
            "some random text",
            "42",
            "function() { return true; }",
            "partial json {",
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                self.assertEqual(result, input_text)

    def test_malformed_python_fallback(self):
        """Test fallback behavior for malformed Python literals."""
        # This tests the fallback when ast.literal_eval fails
        input_text = "{'key': undefined_variable}"  # This will fail ast.literal_eval
        result = self.step.process(input_text)

        # Should fall back to manual replacement
        expected = '{"key": undefined_variable}'  # Single quotes replaced
        self.assertEqual(result, expected)

    def test_step_name(self):
        """Test step name."""
        self.assertEqual(self.step.get_name(), "PythonDictFormat")


class TestJSONParsingStep(unittest.TestCase):
    """Test cases for JSONParsingStep."""

    def setUp(self):
        """Set up test fixtures."""
        self.step = JSONParsingStep()

    def test_valid_json_parsing(self):
        """Test parsing of valid JSON strings."""
        test_cases = [
            '{"key": "value"}',
            '["item1", "item2"]',
            "42",
            '"string"',
            "true",
            "false",
            "null",
            '{"nested": {"data": [1, 2, 3]}}',
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.step.process(input_text)
                # Should return formatted JSON
                self.assertEqual(json.loads(result), json.loads(input_text))

    def test_empty_input(self):
        """Test handling of empty input."""
        test_cases = ["", "   ", "\n\t  "]

        for input_text in test_cases:
            with self.subTest(input_text=repr(input_text)):
                result = self.step.process(input_text)
                self.assertEqual(result, input_text)

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        test_cases = [
            '{"key": value}',  # Unquoted value
            '{key: "value"}',  # Unquoted key
            '{"key": "value",}',  # Trailing comma
            '{"incomplete": "json"',  # Missing closing brace
            "not json at all",
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                with self.assertRaises(json.JSONDecodeError):
                    self.step.process(input_text)

    def test_step_name(self):
        """Test step name."""
        self.assertEqual(self.step.get_name(), "JSONParsing")


class TestJSONParsingPipeline(unittest.TestCase):
    """Test cases for JSONParsingPipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.pipeline = JSONParsingPipeline()

    def test_default_pipeline_steps(self):
        """Test that default pipeline has expected steps."""
        step_names = self.pipeline.get_step_names()
        expected_names = ["MarkdownRemoval", "PythonDictFormat", "JSONParsing"]
        self.assertEqual(step_names, expected_names)

    def test_custom_pipeline_steps(self):
        """Test pipeline with custom steps."""
        custom_steps = [MarkdownRemovalStep(), JSONParsingStep()]
        pipeline = JSONParsingPipeline(custom_steps)

        step_names = pipeline.get_step_names()
        expected_names = ["MarkdownRemoval", "JSONParsing"]
        self.assertEqual(step_names, expected_names)

    def test_add_step(self):
        """Test adding steps to pipeline."""
        initial_count = len(self.pipeline.steps)

        # Add step at end
        new_step = MarkdownRemovalStep()
        self.pipeline.add_step(new_step)
        self.assertEqual(len(self.pipeline.steps), initial_count + 1)
        self.assertIs(self.pipeline.steps[-1], new_step)

        # Add step at specific position
        another_step = PythonDictFormatStep()
        self.pipeline.add_step(another_step, position=0)
        self.assertEqual(len(self.pipeline.steps), initial_count + 2)
        self.assertIs(self.pipeline.steps[0], another_step)

    def test_remove_step(self):
        """Test removing steps from pipeline."""
        initial_count = len(self.pipeline.steps)

        # Remove existing step
        result = self.pipeline.remove_step(MarkdownRemovalStep)
        self.assertTrue(result)
        self.assertEqual(len(self.pipeline.steps), initial_count - 1)

        # Try to remove non-existing step
        result = self.pipeline.remove_step(MarkdownRemovalStep)
        self.assertFalse(result)
        self.assertEqual(len(self.pipeline.steps), initial_count - 1)

    def test_parse_empty_input(self):
        """Test parsing empty or None input."""
        test_cases = ["", "   ", "\n\t  "]

        for input_text in test_cases:
            with self.subTest(input_text=repr(input_text)):
                result = self.pipeline.parse(input_text)
                self.assertTrue(result.success)
                self.assertIsNone(result.data)
                self.assertEqual(result.original_input, input_text)

    def test_parse_valid_json(self):
        """Test parsing valid JSON through pipeline."""
        test_cases = [
            '{"key": "value"}',
            '["item1", "item2"]',
            "42",
            '"string"',
            "true",
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pipeline.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, json.loads(input_text))
                self.assertEqual(result.original_input, input_text)

    def test_parse_markdown_json(self):
        """Test parsing JSON within markdown blocks."""
        test_cases = [
            ('```json\n{"key": "value"}\n```', {"key": "value"}),
            ('```json\n["item1", "item2"]\n```', ["item1", "item2"]),
            ('````json\n{"nested": {"data": 42}}\n````', {"nested": {"data": 42}}),
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pipeline.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, expected_data)

    def test_parse_python_dict_format(self):
        """Test parsing Python dict format."""
        test_cases = [
            ("{'key': 'value'}", {"key": "value"}),
            ("['item1', 'item2']", ["item1", "item2"]),
            ("{'flag': True, 'value': None}", {"flag": True, "value": None}),
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pipeline.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, expected_data)

    def test_parse_combined_formats(self):
        """Test parsing combined format issues."""
        test_cases = [
            # Python dict in markdown block
            (
                "```json\n{'key': 'value', 'flag': True}\n```",
                {"key": "value", "flag": True},
            ),
            # Python list in markdown block
            ("```\n['item1', 'item2', None]\n```", ["item1", "item2", None]),
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pipeline.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, expected_data)

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        test_cases = [
            '{"key": value}',  # Unquoted value
            '{key: "value"}',  # Unquoted key
            "not json at all",
            '{"incomplete": "json"',  # Missing closing brace
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.pipeline.parse(input_text)
                self.assertFalse(result.success)
                self.assertIsNotNone(result.error)
                self.assertEqual(result.original_input, input_text)

    def test_parse_result_error_handling(self):
        """Test proper error information in ParseResult."""
        input_text = '{"invalid": json}'
        result = self.pipeline.parse(input_text)

        self.assertFalse(result.success)
        self.assertIn("JSON parsing failed", result.error)
        self.assertEqual(result.original_input, input_text)
        self.assertIsNone(result.data)


class TestParseResult(unittest.TestCase):
    """Test cases for ParseResult dataclass."""

    def test_successful_result(self):
        """Test successful ParseResult creation."""
        data = {"key": "value"}
        result = ParseResult(success=True, data=data, original_input='{"key": "value"}')

        self.assertTrue(result.success)
        self.assertEqual(result.data, data)
        self.assertIsNone(result.error)
        self.assertEqual(result.original_input, '{"key": "value"}')

    def test_failed_result(self):
        """Test failed ParseResult creation."""
        error_msg = "Parsing failed"
        original = '{"invalid": json}'
        result = ParseResult(success=False, error=error_msg, original_input=original)

        self.assertFalse(result.success)
        self.assertEqual(result.error, error_msg)
        self.assertEqual(result.original_input, original)
        self.assertIsNone(result.data)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""

    def test_create_default_pipeline(self):
        """Test create_default_pipeline function."""
        pipeline = create_default_pipeline()

        self.assertIsInstance(pipeline, JSONParsingPipeline)
        step_names = pipeline.get_step_names()
        expected_names = ["MarkdownRemoval", "PythonDictFormat", "JSONParsing"]
        self.assertEqual(step_names, expected_names)

    def test_parse_json_string_convenience(self):
        """Test parse_json_string convenience function."""
        test_cases = [
            ('{"key": "value"}', {"key": "value"}),
            ("```json\n{'key': 'value'}\n```", {"key": "value"}),
            ("invalid json", None),  # Should fail
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = parse_json_string(input_text)
                self.assertIsInstance(result, ParseResult)

                if expected_data is not None:
                    self.assertTrue(result.success)
                    self.assertEqual(result.data, expected_data)
                else:
                    self.assertFalse(result.success)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_real_world_examples(self):
        """Test with realistic examples from the original codebase."""
        pipeline = create_default_pipeline()

        test_cases = [
            # Markdown with Python dict format
            {
                "input": "```json\n{'user': 'What is this?', 'assistant': 'A medical scan', 'complete': True}\n```",
                "expected": {
                    "user": "What is this?",
                    "assistant": "A medical scan",
                    "complete": True,
                },
            },
            # Python list format
            {
                "input": "['question1', 'question2', 'question3']",
                "expected": ["question1", "question2", "question3"],
            },
            # Mixed boolean and null values
            {
                "input": "{'has_findings': True, 'urgent': False, 'notes': None}",
                "expected": {"has_findings": True, "urgent": False, "notes": None},
            },
            # Already valid JSON
            {
                "input": '{"diagnosis": "pneumonia", "confidence": 0.95}',
                "expected": {"diagnosis": "pneumonia", "confidence": 0.95},
            },
            # Complex nested structure
            {
                "input": """```json
{
    'conversation': [
        {'role': 'user', 'content': 'What do you see?'},
        {'role': 'assistant', 'content': 'I see a chest X-ray showing...'}
    ],
    'metadata': {'processed': True, 'confidence': 0.92}
}
```""",
                "expected": {
                    "conversation": [
                        {"role": "user", "content": "What do you see?"},
                        {
                            "role": "assistant",
                            "content": "I see a chest X-ray showing...",
                        },
                    ],
                    "metadata": {"processed": True, "confidence": 0.92},
                },
            },
        ]

        for i, test_case in enumerate(test_cases):
            with self.subTest(case=i):
                result = pipeline.parse(test_case["input"])
                self.assertTrue(
                    result.success, f"Failed to parse case {i}: {result.error}"
                )
                self.assertEqual(result.data, test_case["expected"])

    def test_error_recovery(self):
        """Test pipeline behavior with various error conditions."""
        pipeline = create_default_pipeline()

        error_cases = [
            "completely invalid input",
            '{"missing_closing_brace": "value"',
            '{unquoted_key: "value"}',
            '{"trailing_comma": "value",}',
            "",  # Empty string should succeed with None
        ]

        for input_text in error_cases:
            with self.subTest(input_text=input_text):
                result = pipeline.parse(input_text)
                self.assertIsInstance(result, ParseResult)

                # Empty string should succeed
                if input_text == "":
                    self.assertTrue(result.success)
                    self.assertIsNone(result.data)
                else:
                    # Other cases should fail gracefully
                    if not result.success:
                        self.assertIsNotNone(result.error)
                        self.assertEqual(result.original_input, input_text)


class TestFallbackChainArchitecture(unittest.TestCase):
    """Test cases for the new fallback chain architecture."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = create_default_json_parser()

    def test_standard_json_primary_strategy(self):
        """Test that standard JSON uses the primary strategy."""
        test_cases = [
            '{"key": "value"}',
            '["item1", "item2"]',
            '{"nested": {"data": [1, 2, 3]}}',
            "42",
            '"string"',
            "true",
            "null",
        ]

        for input_json in test_cases:
            with self.subTest(input_json=input_json):
                result = self.parser.parse(input_json)
                self.assertTrue(result.success)
                self.assertEqual(result.data, json.loads(input_json))

    def test_markdown_preprocessing(self):
        """Test that markdown preprocessing works correctly."""
        test_cases = [
            ('```json\n{"key": "value"}\n```', {"key": "value"}),
            ('```json\n["item1", "item2"]\n```', ["item1", "item2"]),
            ('```\n{"array": [1, 2, 3]}\n```', {"array": [1, 2, 3]}),
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, expected_data)

    def test_python_dict_fallback(self):
        """Test Python dict format fallback strategy."""
        test_cases = [
            ("{'key': 'value'}", {"key": "value"}),
            ("['item1', 'item2']", ["item1", "item2"]),
            ("{'flag': True, 'value': None}", {"flag": True, "value": None}),
            ("{'nested': {'data': [1, 2, 3]}}", {"nested": {"data": [1, 2, 3]}}),
        ]

        for input_text, expected_data in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                self.assertTrue(result.success)
                self.assertEqual(result.data, expected_data)

    def test_truncated_json_fallback(self):
        """Test truncated JSON fallback strategy."""
        test_cases = [
            '{"key": "incomplete_value',
            '{"user": "Hello", "assistant": "I can see signs of pneumonia in the',
            '{"u',
            '{"ass',
            '{"conversation": [{"role": "user", "content": "Test',
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                self.assertTrue(result.success, f"Failed to parse: {input_text}")
                self.assertIsInstance(result.data, dict)

    def test_combined_preprocessing_and_fallbacks(self):
        """Test combination of preprocessing and fallback strategies."""
        test_cases = [
            # Python dict in markdown
            (
                "```json\n{'user': 'Hello', 'flag': True}\n```",
                {"user": "Hello", "flag": True},
            ),
            # Truncated JSON in markdown
            (
                '```json\n{"user": "What do you see?", "assistant": "I see signs of',
                dict,
            ),
        ]

        for input_text, expected in test_cases:
            with self.subTest(input_text=input_text):
                result = self.parser.parse(input_text)
                self.assertTrue(result.success)
                if isinstance(expected, type):
                    self.assertIsInstance(result.data, expected)
                else:
                    self.assertEqual(result.data, expected)

    def test_fallback_strategy_order(self):
        """Test that fallback strategies are tried in correct order."""
        # This tests that TruncatedJSONFallback comes after PythonDictFallback
        # so that complete Python dicts are handled before truncation fixing

        complete_python_dict = "{'complete': True, 'test': 'value'}"
        result = self.parser.parse(complete_python_dict)

        self.assertTrue(result.success)
        self.assertEqual(result.data, {"complete": True, "test": "value"})

    def test_strategy_names(self):
        """Test that strategy names are returned correctly."""
        strategy_names = self.parser.get_strategy_names()
        expected_names = [
            "PythonDictFallback",
            "TruncatedJSONFallback",
            "QuoteEscapingFallback",
            "ControlCharacterFallback",
            "PropertyQuoteFallback",
        ]
        self.assertEqual(strategy_names, expected_names)

    def test_add_custom_fallback_strategy(self):
        """Test adding custom fallback strategies."""

        # Create a simple custom strategy for testing
        class TestFallback:
            def try_parse(self, input_data):
                if input_data.strip() == "TEST":
                    return ParseResult(
                        success=True, data="test_result", original_input=input_data
                    )
                return None

            def get_name(self):
                return "TestFallback"

        # Add custom strategy
        custom_strategy = TestFallback()
        self.parser.add_fallback_strategy(custom_strategy)

        # Test that it's added
        strategy_names = self.parser.get_strategy_names()
        self.assertIn("TestFallback", strategy_names)

        # Test that it works
        result = self.parser.parse("TEST")
        self.assertTrue(result.success)
        self.assertEqual(result.data, "test_result")

    def test_all_strategies_fail(self):
        """Test behavior when all strategies fail."""
        invalid_input = "completely invalid input that no strategy can handle"
        result = self.parser.parse(invalid_input)

        self.assertFalse(result.success)
        self.assertIn("All parsing strategies failed", result.error)

    def test_empty_input_handling(self):
        """Test handling of empty and whitespace inputs."""
        test_cases = ["", "   ", "\n\t  \n"]

        for input_text in test_cases:
            with self.subTest(input_text=repr(input_text)):
                result = self.parser.parse(input_text)
                self.assertTrue(result.success)
                self.assertIsNone(result.data)


class TestFallbackStrategyIsolation(unittest.TestCase):
    """Test fallback strategies in isolation."""

    def setUp(self):
        """Set up test fixtures."""
        self.python_fallback = PythonDictFallback()
        self.truncated_fallback = TruncatedJSONFallback()

    def test_python_dict_fallback_success(self):
        """Test PythonDictFallback with valid inputs."""
        test_cases = [
            "{'key': 'value'}",
            "['item1', 'item2']",
            "{'flag': True, 'value': None}",
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.python_fallback.try_parse(input_text)
                self.assertIsNotNone(result)
                self.assertTrue(result.success)

    def test_python_dict_fallback_rejection(self):
        """Test PythonDictFallback rejects invalid inputs."""
        test_cases = [
            "not a dict at all",
            "just plain text",
            '{"already": "valid json"}',  # Valid JSON, not Python dict format
            '{"incomplete": "dict',  # Incomplete, should be handled by truncation
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.python_fallback.try_parse(input_text)
                # Should either return None or a failed ParseResult
                if result is not None:
                    # If it tries to parse, it should fail gracefully
                    self.assertIsInstance(result, ParseResult)

    def test_truncated_json_fallback_success(self):
        """Test TruncatedJSONFallback with valid inputs."""
        test_cases = [
            '{"key": "incomplete_value',
            '{"user": "Hello", "assistant": "Response was cut off',
            '{"u',
            '[{"item": "one"}, {"item": "incomplete',
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.truncated_fallback.try_parse(input_text)
                self.assertIsNotNone(result)
                self.assertTrue(result.success)

    def test_truncated_json_fallback_rejection(self):
        """Test TruncatedJSONFallback rejects invalid inputs."""
        test_cases = [
            "not json at all",
            "just plain text",
            "no brackets or braces here",
        ]

        for input_text in test_cases:
            with self.subTest(input_text=input_text):
                result = self.truncated_fallback.try_parse(input_text)
                self.assertIsNone(result)


class TestConvenienceFunctionWithFallbacks(unittest.TestCase):
    """Test convenience functions with fallback chain."""

    def test_parse_json_string_with_fallbacks(self):
        """Test parse_json_string convenience function with various inputs."""
        test_cases = [
            ('{"standard": "json"}', True),
            ("{'python': 'dict'}", True),
            ('{"truncated": "value', True),
            ('```json\n{"markdown": "block"}\n```', True),
            ("invalid input", False),
        ]

        for input_text, should_succeed in test_cases:
            with self.subTest(input_text=input_text):
                result = parse_json_string(input_text)
                self.assertEqual(result.success, should_succeed)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
