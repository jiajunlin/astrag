"""astrag — AST-based two-stage code RAG with semantic-anchor compression.

Layer 1  parsing.py / graph.py    structural chunks + code graph
Layer 2  retrieval.py / tools.py  two-stage retrieval + lazy body fetch
Layer 3  compression.py           budgeted semantic-anchor compression
         pipeline.py              CodebaseMemory orchestrator
"""
from .compression import (CompressionResult, SemanticAnchorCompressor,
                          knapsack_pack, make_surprisal_score_fn,
                          sliced_text)
from .slicing import slice_lines
from .graph import CodeGraph
from .parsing import (CodeChunk, PythonStdlibParser, TreeSitterParser,
                      approx_tokens, code_tokens)
from .langs import SUPPORTED_EXTENSIONS, heuristic_parser_for
from .universal import (BINARY_EXTENSIONS, KNOWN_EXTENSIONS,
                        GenericCodeChunker, parser_for)
from .pipeline import BuiltContext, CodebaseMemory, REPLICATION_CHECK_INSTRUCTION
from .retrieval import (BM25Index, DenseIndex, RetrievalResult, RetrievedCard,
                        TwoStageRetriever)
from .tools import CodebaseTools

__version__ = "0.3.1"
__all__ = [
    "BM25Index", "BuiltContext", "CodeChunk", "CodeGraph", "CodebaseMemory",
    "CodebaseTools", "CompressionResult", "DenseIndex", "PythonStdlibParser",
    "REPLICATION_CHECK_INSTRUCTION", "RetrievalResult", "RetrievedCard",
    "SUPPORTED_EXTENSIONS", "SemanticAnchorCompressor", "TreeSitterParser",
    "TwoStageRetriever", "approx_tokens", "code_tokens",
    "heuristic_parser_for", "knapsack_pack", "make_surprisal_score_fn",
    "slice_lines", "sliced_text", "BINARY_EXTENSIONS",
    "KNOWN_EXTENSIONS", "GenericCodeChunker", "parser_for",
]
