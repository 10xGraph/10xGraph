# """
# Agent Communication Protocol (ACP)

# A standardized protocol for agent-to-agent communication in the Agentflow system.
# Provides message format, validation, and serialization for reliable agent interactions.
# """

# import uuid
# from datetime import UTC, datetime
# from enum import Enum
# from typing import Any

# from pydantic import BaseModel, Field, field_validator


# class ACPMessageType(str, Enum):
#     """Types of messages supported by ACP."""

#     REQUEST = "REQUEST"  # Agent requests action from another agent
#     RESPONSE = "RESPONSE"  # Response to a previous request
#     BROADCAST = "BROADCAST"  # Message to all agents
#     NOTIFICATION = "NOTIFICATION"  # One-way notification
#     ERROR = "ERROR"  # Error message
#     HEARTBEAT = "HEARTBEAT"  # Keep-alive message


# class MessageContent(BaseModel):
#     """Content of an ACP message."""

#     action: str = Field(..., description="Action to perform or type of content")
#     data: dict[str, Any] = Field(default_factory=dict, description="Message payload data")
#     metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# class MessageContext(BaseModel):
#     """Context information for message routing and tracking."""

#     thread_id: str | None = Field(None, description="Conversation thread ID")
#     conversation_id: str | None = Field(None, description="Conversation ID")
#     correlation_id: str | None = Field(None, description="ID to correlate request-response pairs")
#     parent_message_id: str | None = Field(
#         None, description="ID of the parent message if this is a reply"
#     )


# class ACPMessage(BaseModel):
#     """
#     Agent Communication Protocol Message.

#     Standard message format for all agent-to-agent communications.
#     """

#     protocol_version: str = Field(default="1.0", description="ACP protocol version")
#     message_id: str = Field(
#         default_factory=lambda: str(uuid.uuid4()),
#         description="Unique message identifier",
#     )
#     message_type: ACPMessageType = Field(..., description="Type of message")
#     sender_id: str = Field(..., description="ID of the sending agent")
#     recipient_id: str = Field(
#         ...,
#         description="ID of recipient agent, or '*' for broadcast",
#     )
#     timestamp: datetime = Field(
#         default_factory=lambda: datetime.now(UTC),
#         description="Message creation timestamp",
#     )
#     content: MessageContent = Field(..., description="Message content")
#     context: MessageContext = Field(
#         default_factory=MessageContext,
#         description="Message context for routing",
#     )
#     priority: int = Field(
#         default=5, ge=1, le=10, description="Message priority (1=highest, 10=lowest)"
#     )
#     ttl: int | None = Field(None, gt=0, description="Time-to-live in seconds (optional)")

#     @field_validator("recipient_id")
#     @classmethod
#     def validate_recipient(cls, v: str) -> str:
#         """Validate recipient ID format."""
#         if not v:
#             raise ValueError("recipient_id cannot be empty")
#         # '*' is valid for broadcast
#         if v == "*":
#             return v
#         # Add more validation as needed
#         return v

#     def is_broadcast(self) -> bool:
#         """Check if this is a broadcast message."""
#         return self.recipient_id == "*"

#     def is_expired(self) -> bool:
#         """Check if message has exceeded its TTL."""
#         if self.ttl is None:
#             return False
#         elapsed = (datetime.now(UTC) - self.timestamp).total_seconds()
#         return elapsed > self.ttl

#     def to_dict(self) -> dict[str, Any]:
#         """Convert message to dictionary."""
#         return self.model_dump(mode="json")

#     def to_json(self) -> str:
#         """Serialize message to JSON string."""
#         return self.model_dump_json()

#     @classmethod
#     def from_dict(cls, data: dict[str, Any]) -> "ACPMessage":
#         """Create message from dictionary."""
#         return cls(**data)

#     @classmethod
#     def from_json(cls, json_str: str) -> "ACPMessage":
#         """Deserialize message from JSON string."""
#         return cls.model_validate_json(json_str)


# class ACPProtocol:
#     """
#     Agent Communication Protocol handler.

#     Provides utilities for creating, validating, and processing ACP messages.
#     """

#     PROTOCOL_VERSION = "1.0"

#     @staticmethod
#     def create_request(
#         sender_id: str,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage:
#         """
#         Create a REQUEST message.

#         Args:
#             sender_id: ID of the sending agent
#             recipient_id: ID of the recipient agent
#             action: Action to request
#             data: Request data
#             **kwargs: Additional message fields
#         """
#         return ACPMessage(
#             message_type=ACPMessageType.REQUEST,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             content=MessageContent(
#                 action=action,
#                 data=data or {},
#             ),
#             **kwargs,
#         )

#     @staticmethod
#     def create_response(
#         request_message: ACPMessage,
#         sender_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage:
#         """
#         Create a RESPONSE message to a previous request.

#         Args:
#             request_message: Original request message
#             sender_id: ID of the responding agent
#             action: Response action/result
#             data: Response data
#             **kwargs: Additional message fields
#         """
#         context = MessageContext(
#             correlation_id=request_message.message_id,
#             parent_message_id=request_message.message_id,
#             thread_id=request_message.context.thread_id,
#             conversation_id=request_message.context.conversation_id,
#         )

#         return ACPMessage(
#             message_type=ACPMessageType.RESPONSE,
#             sender_id=sender_id,
#             recipient_id=request_message.sender_id,
#             content=MessageContent(
#                 action=action,
#                 data=data or {},
#             ),
#             context=context,
#             **kwargs,
#         )

#     @staticmethod
#     def create_broadcast(
#         sender_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage:
#         """
#         Create a BROADCAST message to all agents.

#         Args:
#             sender_id: ID of the broadcasting agent
#             action: Broadcast action
#             data: Broadcast data
#             **kwargs: Additional message fields
#         """
#         return ACPMessage(
#             message_type=ACPMessageType.BROADCAST,
#             sender_id=sender_id,
#             recipient_id="*",
#             content=MessageContent(
#                 action=action,
#                 data=data or {},
#             ),
#             **kwargs,
#         )

#     @staticmethod
#     def create_notification(
#         sender_id: str,
#         recipient_id: str,
#         action: str,
#         data: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage:
#         """
#         Create a NOTIFICATION message (one-way, no response expected).

#         Args:
#             sender_id: ID of the sending agent
#             recipient_id: ID of the recipient agent
#             action: Notification action
#             data: Notification data
#             **kwargs: Additional message fields
#         """
#         return ACPMessage(
#             message_type=ACPMessageType.NOTIFICATION,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             content=MessageContent(
#                 action=action,
#                 data=data or {},
#             ),
#             **kwargs,
#         )

#     @staticmethod
#     def create_error(
#         sender_id: str,
#         recipient_id: str,
#         error_message: str,
#         error_code: str | None = None,
#         original_message_id: str | None = None,
#         **kwargs: Any,
#     ) -> ACPMessage:
#         """
#         Create an ERROR message.

#         Args:
#             sender_id: ID of the sending agent
#             recipient_id: ID of the recipient agent
#             error_message: Error description
#             error_code: Optional error code
#             original_message_id: ID of message that caused the error
#             **kwargs: Additional message fields
#         """
#         data = {"error_message": error_message}
#         if error_code:
#             data["error_code"] = error_code

#         context = MessageContext()
#         if original_message_id:
#             context.parent_message_id = original_message_id

#         return ACPMessage(
#             message_type=ACPMessageType.ERROR,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             content=MessageContent(
#                 action="error",
#                 data=data,
#             ),
#             context=context,
#             **kwargs,
#         )

#     @staticmethod
#     def create_heartbeat(sender_id: str, recipient_id: str = "*") -> ACPMessage:
#         """
#         Create a HEARTBEAT message.

#         Args:
#             sender_id: ID of the sending agent
#             recipient_id: ID of recipient (default: broadcast)
#         """
#         return ACPMessage(
#             message_type=ACPMessageType.HEARTBEAT,
#             sender_id=sender_id,
#             recipient_id=recipient_id,
#             content=MessageContent(action="heartbeat"),
#             priority=10,  # Lowest priority
#         )

#     @staticmethod
#     def validate_message(message: ACPMessage) -> tuple[bool, str | None]:
#         """
#         Validate an ACP message.

#         Returns:
#             Tuple of (is_valid, error_message)
#         """
#         try:
#             # Check if expired
#             if message.is_expired():
#                 return False, "Message has expired (TTL exceeded)"

#             # Validate protocol version
#             if message.protocol_version != ACPProtocol.PROTOCOL_VERSION:
#                 return (
#                     False,
#                     f"Unsupported protocol version: {message.protocol_version}",
#                 )

#             # Validate required fields
#             if not message.sender_id:
#                 return False, "sender_id is required"

#             if not message.recipient_id:
#                 return False, "recipient_id is required"

#             if not message.content.action:
#                 return False, "content.action is required"

#             return True, None

#         except Exception as e:
#             return False, f"Validation error: {e!s}"
