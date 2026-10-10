"""
State management for TAF agent graphs.

This package provides schemas and context managers for agent state, execution
tracking, and message context management. All core state classes are exported
for use in agent workflows and custom state extensions.
"""

from .agent_state import AgentState
from .base_context import BaseContextManager
from .execution_state import ExecutionState, ExecutionStatus
from .message import (
    Message,
    TokenUsages,
)
from .message_block import (
    AnnotationBlock,
    AnnotationRef,
    AudioBlock,
    ContentBlock,
    DataBlock,
    DocumentBlock,
    ErrorBlock,
    ImageBlock,
    MediaRef,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    VideoBlock,
)
from .message_context_manager import MessageContextManager
from .reducers import (
    add_messages,
    append_items,
    remove_tool_messages,
    replace_messages,
    replace_value,
)
from .stream_chunks import StreamChunk, StreamEvent
from .stream_emitter import StreamEmitter
from .summary_context_manager import SummaryContextManager
from .tool_result import ToolResult


__all__ = [
    "AgentState",
    "AnnotationBlock",
    "AnnotationRef",
    "AudioBlock",
    "BaseContextManager",
    "ContentBlock",
    "DataBlock",
    "DocumentBlock",
    "ErrorBlock",
    "ExecutionState",
    "ExecutionStatus",
    "ImageBlock",
    "MediaRef",
    "Message",
    "MessageContextManager",
    "ReasoningBlock",
    "StreamChunk",
    "StreamEmitter",
    "StreamEvent",
    "SummaryContextManager",
    "TextBlock",
    "TextBlock",
    "TokenUsages",
    "ToolCallBlock",
    "ToolResult",
    "ToolResultBlock",
    "VideoBlock",
    "add_messages",
    "append_items",
    "remove_tool_messages",
    "replace_messages",
    "replace_value",
]
