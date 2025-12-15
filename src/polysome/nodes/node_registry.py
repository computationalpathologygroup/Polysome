from polysome.nodes.text_prompt_node import TextPromptNode
from polysome.nodes.load_node import LoadNode
from polysome.nodes.util_nodes import (
    RegexSplitNode,
    SentenceSplitNode,
    DeduplicationNode,
    RowConcatenationNode,
    ColumnConcatenationNode,
)
from polysome.nodes.combine_outputs_node import CombineIntermediateOutputsNode

# --- Registry ---
# Maps node types from JSON config to Python classes
NODE_TYPE_MAP = {
    "text_prompt": TextPromptNode,
    "load": LoadNode,
    "regex_split": RegexSplitNode,
    "sentence_split": SentenceSplitNode,
    "deduplication": DeduplicationNode,
    "row_concatenation": RowConcatenationNode,
    "column_concatenation": ColumnConcatenationNode,
    "combine_intermediate_outputs": CombineIntermediateOutputsNode,
    # Add other node types here
}
