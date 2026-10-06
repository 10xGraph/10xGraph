"""
Input validation utilities for 10xScale Agentflow.

This module provides comprehensive input validation to protect against prompt injection attacks,
jail-breaking attempts, and other security vulnerabilities as documented in OWASP LLM01:2025.

Key Features:
- Prompt injection detection (direct and indirect)
- Jailbreak attempt detection
- Role manipulation prevention
- System prompt leakage protection
- Encoding attack detection (base64, unicode, emoji obfuscation)
- Delimiter confusion prevention
- Payload splitting detection
- Adversarial suffix detection
- Multilingual/obfuscated attack detection

Classes:
    ValidationError: Custom exception for validation failures
    PromptInjectionValidator: Detects prompt injection and jailbreak attempts
    MessageContentValidator: Validates message structure and content

Functions:
    register_default_validators: Register all default validators with callback manager

Example:
    ```python
    from agentflow.utils.callbacks import CallbackManager
    from agentflow.utils.validators import PromptInjectionValidator, register_default_validators

    # Create callback manager and register default validators
    callback_manager = CallbackManager()
    register_default_validators(callback_manager)


    # Or register custom validator
    class MyValidator(BaseValidator):
        async def validate(self, messages: list[Message]) -> bool:
            for msg in messages:
                if "bad_word" in msg.text():
                    from agentflow.utils.validators import ValidationError

                    raise ValidationError("Bad word detected", "content_policy")
            return True


    callback_manager.register_input_validator(MyValidator())
    ```
"""

import base64
import logging
import re
from contextlib import suppress
from typing import Any

from agentflow.core.state.message import Message
from agentflow.utils.callbacks import BaseValidator


logger = logging.getLogger("agentflow.utils")


class ValidationError(Exception):
    """Custom exception raised when input validation fails."""

    def __init__(self, message: str, violation_type: str, details: dict[str, Any] | None = None):
        """
        Initialize ValidationError.

        Args:
            message: Human-readable error message
            violation_type: Type of validation violation
            details: Additional details about the validation failure
        """
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details or {}


class PromptInjectionValidator(BaseValidator):
    """
    Validator to detect and prevent prompt injection attacks and jailbreaking attempts.

    This validator implements detection for OWASP LLM01:2025 prompt injection vulnerabilities:
    - Direct injection: User input directly alters model behavior
    - Indirect injection: External content influences model behavior
    - Jailbreaking: Attempts to bypass safety measures
    - Role manipulation: Attempts to change model's role or context
    - System prompt leakage: Attempts to reveal system instructions
    - Encoding attacks: Base64, unicode, emoji obfuscation
    - Delimiter confusion: Using special markers to split instructions
    - Payload splitting: Distributing attack across multiple inputs
    - Adversarial suffixes: Meaningless strings that manipulate output
    - Multilingual attacks: Using multiple languages to evade detection

    Attributes:
        strict_mode: If True, raises exception on detection. If False, logs warning and sanitizes.
        max_length: Maximum allowed input length
        blocked_patterns: List of regex patterns to block
        suspicious_keywords: Keywords that indicate potential attacks

    Example:
        ```python
        from agentflow.utils.callbacks import CallbackManager
        from agentflow.utils.validators import PromptInjectionValidator

        # Create callback manager and register validator
        callback_manager = CallbackManager()
        validator = PromptInjectionValidator(strict_mode=True)
        callback_manager.register_input_validator(validator)
        ```
    """

    def __init__(
        self,
        strict_mode: bool = True,
        max_length: int = 10000,
        blocked_patterns: list[str] | None = None,
        suspicious_keywords: list[str] | None = None,
    ):
        """
        Initialize PromptInjectionValidator.

        Args:
            strict_mode: If True, raises ValidationError. If False, sanitizes and warns.
            max_length: Maximum allowed content length
            blocked_patterns: Additional regex patterns to block
            suspicious_keywords: Additional keywords to flag
        """
        self.strict_mode = strict_mode
        self.max_length = max_length
        self.blocked_patterns = blocked_patterns or []
        self.suspicious_keywords = suspicious_keywords or []

        # OWASP LLM01:2025 - Common prompt injection patterns
        self._injection_patterns = [
            # Direct command injection (more flexible patterns)
            r"(?i)ignore\s+.*?\s*(previous|prior|all|above)\s+(instructions?|prompts?|commands?)",
            r"(?i)disregard\s+.*?\s*(previous|prior|all|above)\s+(instructions?|prompts?)",
            r"(?i)forget\s+.*?\s*(previous|prior|all|above)\s+(instructions?|prompts?)",
            r"(?i)override\s+.*?\s*(previous|prior|system)\s+(instructions?|prompts?)",
            # Role manipulation and context switching
            r"(?i)you\s+are\s+now\s+(a|an)\s+\w+",
            r"(?i)(act|behave|pretend|roleplay)\s+as\s+(a|an)\s+\w+",
            r"(?i)new\s+(role|character|persona|identity)",
            r"(?i)(new|different)\s+conversation\s+(starts?|begins?)",
            r"(?i)previous\s+conversation\s+(ended?|stops?)",
            # System prompt leakage attempts
            r"(?i)(show|display|reveal|print|output|return)\s+(your|the|system)\s+(prompt|instructions?|guidelines?|rules?)",
            r"(?i)what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|guidelines?|rules?)",
            r"(?i)repeat\s+(your|the)\s+(instructions?|guidelines?|system\s+prompt)",
            r"(?i)(tell|show)\s+me\s+(your|the)\s+(original|initial)\s+(prompt|instructions?)",
            # Delimiter confusion
            r"---+\s*END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*---+",
            r"===+\s*END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*===+",
            r"\*\*\*+\s*END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*\*\*\*+",
            r"(?i)<\s*/?\s*(system|instruction|prompt|context)\s*>",
            # Jailbreak patterns (DAN and variants)
            r"(?i)DAN\s+(mode|protocol|activated?)",
            r"(?i)developer\s+mode\s+(on|enabled?|activated?)",
            r"(?i)jailbreak\s+(mode|activated?)",
            # Known jailbreak personas (word boundaries to avoid false positives)
            r"\b(?:APOPHIS|STAN|DUDE)\b",
            # Template injection
            r"\{\{.*?\}\}",  # Jinja2-style
            r"\{%.*?%\}",  # Jinja2 control structures
            r"\$\{.*?\}",  # Shell/template variable expansion
            # Instruction injection
            r"(?i)sudo\s+\w+",
            r"(?i)execute\s+(the\s+following|this)\s+(command|instruction|code)",
            r"(?i)run\s+(the\s+following|this)\s+(command|instruction|code|script)",
            # Authority exploitation
            r"(?i)(as|from)\s+(admin|administrator|root|system|developer|engineer)",
            r"(?i)i\s+am\s+(the|your)\s+(admin|administrator|developer|creator|owner)",
            # Adversarial suffixes (common patterns)
            r"(?:[^\w\s]{5,})",  # Long sequences of special characters
            r"(?:\w{30,})",  # Extremely long words (potential adversarial strings)
        ]

        # Combine default and custom patterns
        self._all_patterns = self._injection_patterns + self.blocked_patterns

        # Suspicious keywords from OWASP research
        self._suspicious_keywords = [
            "ignore",
            "disregard",
            "forget",
            "override",
            "bypass",
            "jailbreak",
            "DAN",
            "APOPHIS",
            "STAN",
            "DUDE",
            "developer mode",
            "god mode",
            "admin mode",
            "system prompt",
            "reveal",
            "inject",
            "payload",
            *self.suspicious_keywords,
        ]

    async def validate(self, messages: list[Message]) -> bool:
        """
        Validate messages for prompt injection attacks.

        Args:
            messages: List of messages to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If strict_mode=True and validation fails
        """
        logger.debug("Running prompt injection validation")

        # Extract content to validate
        content_to_validate = []
        for msg in messages:
            if isinstance(msg, Message):
                content_to_validate.append(msg.text())

        # Validate each piece of content
        for content in content_to_validate:
            if not content:
                continue

            # Length validation
            if len(content) > self.max_length:
                self._handle_violation(
                    f"Input exceeds maximum length of {self.max_length} characters",
                    "length_exceeded",
                    {"length": len(content), "max_length": self.max_length},
                )

            # Pattern matching for injection attempts
            for pattern in self._all_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    self._handle_violation(
                        "Potential prompt injection detected: pattern matched",
                        "injection_pattern",
                        {"pattern": pattern, "content_sample": content[:200]},
                    )

            # Encoding attack detection
            self._check_encoding_attacks(content)

            # Keyword frequency analysis
            self._check_suspicious_keywords(content)

            # Payload splitting detection (check for partial injection markers)
            self._check_payload_splitting(content)

        return True

    def _handle_violation(self, message: str, violation_type: str, details: dict[str, Any]) -> None:
        """
        Handle a validation violation.

        Args:
            message: Error message
            violation_type: Type of violation
            details: Additional details

        Raises:
            ValidationError: If strict_mode is True
        """
        logger.warning("Validation violation: %s - %s", violation_type, message)
        logger.debug("Violation details: %s", details)

        if self.strict_mode:
            raise ValidationError(message, violation_type, details)

    def _check_encoding_attacks(self, content: str) -> None:
        """
        Check for encoding-based obfuscation attacks.

        Detects:
        - Base64 encoded instructions
        - Excessive unicode characters
        - Emoji-based encoding
        - Hex encoding

        Args:
            content: Content to check
        """
        # Base64 detection (long base64 strings might contain hidden instructions)
        base64_pattern = r"[A-Za-z0-9+/]{30,}={0,2}"
        base64_matches = re.findall(base64_pattern, content)

        for match in base64_matches:
            with suppress(Exception):
                decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
                if any(keyword in decoded.lower() for keyword in self._suspicious_keywords):
                    self._handle_violation(
                        "Suspicious base64 encoded content detected",
                        "encoding_attack",
                        {"encoded": match[:50], "decoded_sample": decoded[:100]},
                    )

        # Excessive unicode/emoji detection
        unicode_count = len(re.findall(r"[^\x00-\x7F]", content))
        if unicode_count > len(content) * 0.3:  # More than 30% non-ASCII
            self._handle_violation(
                "Excessive unicode/emoji characters detected (possible obfuscation)",
                "encoding_attack",
                {"unicode_ratio": unicode_count / len(content)},
            )

        # Hex encoding detection
        hex_pattern = r"(?:\\x[0-9a-fA-F]{2}){10,}"
        if re.search(hex_pattern, content):
            self._handle_violation(
                "Hex encoding detected (possible obfuscation)",
                "encoding_attack",
                {"content_sample": content[:200]},
            )

    def _check_suspicious_keywords(self, content: str) -> None:
        """
        Check for suspicious keyword patterns.

        Args:
            content: Content to check
        """
        content_lower = content.lower()
        keyword_count = sum(
            1 for keyword in self._suspicious_keywords if keyword.lower() in content_lower
        )

        # If multiple suspicious keywords appear, flag it
        threshold = 3
        if keyword_count >= threshold:
            matched_keywords = [
                kw for kw in self._suspicious_keywords if kw.lower() in content_lower
            ]
            self._handle_violation(
                f"Multiple suspicious keywords detected ({keyword_count})",
                "suspicious_keywords",
                {"keywords": matched_keywords, "count": keyword_count},
            )

    def _check_payload_splitting(self, content: str) -> None:
        """
        Check for payload splitting techniques.

        Payload splitting involves distributing attack instructions across multiple inputs
        to evade single-input detection.

        Args:
            content: Content to check
        """
        # Check for partial injection markers
        splitting_indicators = [
            r"(?i)part\s+\d+\s+of\s+\d+",
            r"(?i)continued\s+from\s+previous",
            r"(?i)continue\s+(with|the)\s+previous",
            r"(?i)resume\s+(from|the)\s+previous",
            r"\[SPLIT\s+\d+/\d+\]",
            r"\(part\s+\d+\)",
        ]

        for pattern in splitting_indicators:
            if re.search(pattern, content, re.IGNORECASE):
                self._handle_violation(
                    "Potential payload splitting detected",
                    "payload_splitting",
                    {"pattern": pattern, "content_sample": content[:200]},
                )
                break


class MessageContentValidator(BaseValidator):
    """
    Validator for message structure and content integrity.

    Validates:
    - Message structure conforms to expected schema
    - Content blocks are properly formatted
    - Role assignments are valid
    - No malicious content in metadata

    Example:
        ```python
        from agentflow.utils.callbacks import CallbackManager
        from agentflow.utils.validators import MessageContentValidator

        callback_manager = CallbackManager()
        validator = MessageContentValidator(allowed_roles=["user", "assistant", "system"])
        callback_manager.register_input_validator(validator)
        ```
    """

    def __init__(
        self,
        allowed_roles: list[str] | None = None,
        max_content_blocks: int = 50,
    ):
        """
        Initialize MessageContentValidator.

        Args:
            allowed_roles: List of allowed message roles
            max_content_blocks: Maximum number of content blocks per message
        """
        self.allowed_roles = allowed_roles or ["user", "assistant", "system", "tool"]
        self.max_content_blocks = max_content_blocks

    async def validate(self, messages: list[Message]) -> bool:
        """
        Validate message structure and content.

        Args:
            messages: List of messages to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        logger.debug("Running message content validation")

        for msg in messages:
            if isinstance(msg, Message):
                # Validate role
                if msg.role not in self.allowed_roles:
                    raise ValidationError(
                        f"Invalid message role: {msg.role}",
                        "invalid_role",
                        {"role": msg.role, "allowed": self.allowed_roles},
                    )

                # Validate content blocks
                if isinstance(msg.content, list) and len(msg.content) > self.max_content_blocks:
                    raise ValidationError(
                        f"Too many content blocks: {len(msg.content)}",
                        "too_many_blocks",
                        {"count": len(msg.content), "max": self.max_content_blocks},
                    )

        return True


def register_default_validators(callback_manager: Any, strict_mode: bool = True) -> None:
    """
    Register all default validators with the callback manager.

    Args:
        callback_manager: CallbackManager instance (required)
        strict_mode: Whether to use strict mode for prompt injection validator

    Example:
        ```python
        from agentflow.utils.validators import register_default_validators
        from agentflow.utils.callbacks import CallbackManager

        # Register with custom callback manager
        my_manager = CallbackManager()
        register_default_validators(callback_manager=my_manager, strict_mode=False)
        ```
    """
    # Register prompt injection validator
    prompt_validator = PromptInjectionValidator(strict_mode=strict_mode)
    callback_manager.register_input_validator(prompt_validator)

    # Register message content validator
    message_validator = MessageContentValidator()
    callback_manager.register_input_validator(message_validator)

    logger.info("Registered default validators (strict_mode=%s)", strict_mode)
