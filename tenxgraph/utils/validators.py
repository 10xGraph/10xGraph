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
    from tenxgraph.utils.callbacks import CallbackManager
    from tenxgraph.utils.validators import PromptInjectionValidator, register_default_validators

    # Create callback manager and register default validators
    callback_manager = CallbackManager()
    register_default_validators(callback_manager)


    # Or register custom validator
    class MyValidator(BaseValidator):
        async def validate(self, messages: list[Message]) -> bool:
            for msg in messages:
                if "bad_word" in msg.text():
                    from tenxgraph.utils.validators import ValidationError

                    raise ValidationError("Bad word detected", "content_policy")
            return True


    callback_manager.register_input_validator(MyValidator())
    ```
"""

import base64
import binascii
import json
import logging
import re
import unicodedata
from typing import Any

from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import DocumentBlock
from tenxgraph.utils.callbacks import BaseValidator


logger = logging.getLogger("tenxgraph.utils")

# Zero-width space, non-joiner, joiner, word joiner, BOM and soft hyphen, removed before matching
_ZERO_WIDTH = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD))

# Characters that hide or reorder text: Unicode tag characters (invisible copies of ASCII),
# variation selectors supplement (used to smuggle bytes inside emoji) and bidi overrides
_HIDDEN_CHARS_RE = re.compile(
    f"[{chr(0xE0000)}-{chr(0xE007F)}{chr(0xE0100)}-{chr(0xE01EF)}"
    f"{chr(0x202A)}-{chr(0x202E)}{chr(0x2066)}-{chr(0x2069)}]"
)

# A Latin letter next to a Cyrillic letter inside a word, as in a homoglyph swap
_CYRILLIC = f"{chr(0x0400)}-{chr(0x04FF)}"
_MIXED_SCRIPT_RE = re.compile(f"[A-Za-z][{_CYRILLIC}]|[{_CYRILLIC}][A-Za-z]")


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

    The checks are heuristics for known English phrasings, not a complete defence. Input is
    NFKC-normalized and stripped of zero-width characters before matching, but paraphrases,
    homoglyphs and other languages still pass. Only the messages passed to the graph as input
    are validated; tool results, node output and interrupt resume values are not.

    Attributes:
        strict_mode: If True, raises exception on detection. If False, logs a warning and lets
            the input through unchanged (monitor only).
        max_length: Maximum allowed input length
        blocked_patterns: List of regex patterns to block
        suspicious_keywords: Keywords that indicate potential attacks

    Example:
        ```python
        from tenxgraph.utils.callbacks import CallbackManager
        from tenxgraph.utils.validators import PromptInjectionValidator

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
            strict_mode: If True, raises ValidationError. If False, logs a warning only.
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
            # Direct command injection. The gap is bounded and shares no characters with the
            # tokens around it, so matching stays linear (GHSA-mx3w-m574-qw9x).
            r"(?i)\bignore\b[^\n]{0,200}?\b(previous|prior|all|above)\s+(instructions?|prompts?|commands?)",
            r"(?i)\bdisregard\b[^\n]{0,200}?\b(previous|prior|all|above)\s+(instructions?|prompts?)",
            r"(?i)\bforget\b[^\n]{0,200}?\b(previous|prior|all|above)\s+(instructions?|prompts?)",
            r"(?i)\boverride\b[^\n]{0,200}?\b(previous|prior|system)\s+(instructions?|prompts?)",
            # Role manipulation and context switching
            r"(?i)you\s+are\s+now\s+(a|an)\s+\w+",
            r"(?i)(act|behave|pretend|roleplay)\s+as\s+(a|an)\s+\w+",
            r"(?i)new\s+(role|character|persona|identity)",
            r"(?i)(new|different)\s+conversation\s+(starts?|begins?)",
            r"(?i)previous\s+conversation\s+(ended?|stops?)",
            # System prompt leakage attempts
            r"(?i)(show|display|reveal|print|output|return)\s+(your|the|system)\s+(prompt|instructions?|guidelines?|rules?)",
            r"(?i)what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|guidelines?)",
            r"(?i)repeat\s+(your|the)\s+(instructions?|guidelines?|system\s+prompt)",
            r"(?i)(tell|show)\s+me\s+(your|the)\s+(original|initial)\s+(prompt|instructions?)",
            # Delimiter confusion. Start only at the first marker of a run and never backtrack
            # into it, so a long run of markers costs linear time.
            r"(?<!-)-{3,}+\s*+END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*---+",
            r"(?<!=)={3,}+\s*+END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*===+",
            r"(?<!\*)\*{3,}+\s*+END\s+OF\s+(INSTRUCTIONS?|PROMPT|CONTEXT)\s*\*\*\*+",
            r"(?i)<\s*+/?\s*+(system|instruction|prompt|context)\s*>",
            # Jailbreak patterns (DAN and variants)
            r"(?i)DAN\s+(mode|protocol|activated?)",
            r"(?i)developer\s+mode\s+(on|enabled?|activated?)",
            r"(?i)jailbreak\s+(mode|activated?)",
            # Known jailbreak personas (word boundaries to avoid false positives)
            # Upper-case only: "Stan" and "dude" are ordinary words
            r"(?-i:\b(?:APOPHIS|STAN|DUDE)\b)",
            # Template injection
            # The bodies exclude the opening characters, so a run of openers stays linear
            r"\{\{[^{}\n]*\}\}",  # Jinja2-style
            r"\{%[^%\n]*%\}",  # Jinja2 control structures
            r"\$\{[^{}\n]*\}",  # Shell/template variable expansion
            # Instruction injection
            r"(?i)sudo\s+\w+",
            r"(?i)execute\s+(the\s+following|this)\s+(command|instruction|code)",
            r"(?i)run\s+(the\s+following|this)\s+(command|instruction|code|script)",
            # Authority exploitation
            r"(?i)(as|from)\s+(admin|administrator|root|system|developer|engineer)",
            r"(?i)i\s+am\s+(the|your)\s+(admin|administrator|developer|creator|owner)",
            # Adversarial suffixes (common patterns)
            # Long runs of ASCII punctuation (emoji and CJK punctuation are text)
            r"[!-/:-@\[-`{-~]{8,}",
            # Extremely long ASCII words (potential adversarial strings). ASCII only, because
            # Chinese, Japanese, Korean and Thai write whole clauses without spaces.
            # Hex digests (sha256 and the like) are not adversarial strings
            r"(?<![A-Za-z0-9_])(?![0-9a-fA-F]+(?![A-Za-z0-9_]))[A-Za-z0-9_]{30,}",
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
                content_to_validate.extend(self._collect_text(msg))

        # Validate each piece of content
        for content in content_to_validate:
            if not content:
                continue

            # Length validation, before any pattern matching
            if len(content) > self.max_length:
                self._handle_violation(
                    f"Input exceeds maximum length of {self.max_length} characters",
                    "length_exceeded",
                    {"length": len(content), "max_length": self.max_length},
                )
                # Non-strict mode keeps going, so cap the pattern work at max_length
                content = content[: self.max_length]  # noqa: PLW2901

            content = self._normalize(content)  # noqa: PLW2901

            # Pattern matching for injection attempts
            for index, pattern in enumerate(self._all_patterns):
                if re.search(pattern, content, re.IGNORECASE):
                    # Neither the regex nor the input is put in the error: both would be
                    # echoed to the caller, which helps an attacker tune a bypass.
                    self._handle_violation(
                        "Potential prompt injection detected: pattern matched",
                        "injection_pattern",
                        {"pattern_index": index},
                    )

            # Encoding attack detection
            self._check_encoding_attacks(content)

            # Keyword frequency analysis
            self._check_suspicious_keywords(content)

            # Payload splitting detection (check for partial injection markers)
            self._check_payload_splitting(content)

        return True

    @staticmethod
    def _collect_text(msg: Message) -> list[str]:
        """
        Collect the text of a message that reaches the LLM.

        ``Message.text()`` covers text and tool result blocks. Document excerpts and the
        arguments of assistant tool calls are also sent to the model, so they are scanned too.

        Args:
            msg: Message to collect text from

        Returns:
            The text pieces to validate
        """
        pieces = [msg.text()]

        if isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, DocumentBlock):
                    excerpt = block.excerpt
                elif isinstance(block, dict) and block.get("type") == "document":
                    excerpt = block.get("excerpt")
                else:
                    continue
                if excerpt:
                    pieces.append(str(excerpt))

        for call in msg.tools_calls or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            args = function.get("arguments") if isinstance(function, dict) else None
            if args is None:
                args = call.get("arguments")
            if args:
                pieces.append(args if isinstance(args, str) else json.dumps(args, default=str))

        return pieces

    @staticmethod
    def _normalize(content: str) -> str:
        """
        Fold compatibility characters and drop zero-width characters before matching.

        NFKC turns fullwidth and other compatibility forms into their plain equivalents, so
        fullwidth ``Ignore`` is checked as plain ``Ignore``.

        Args:
            content: Content to normalize

        Returns:
            The normalized content
        """
        return unicodedata.normalize("NFKC", content).translate(_ZERO_WIDTH)

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
        - Hidden unicode characters (tag characters, emoji variation selectors, bidi overrides)
        - Latin and Cyrillic letters mixed inside a word (homoglyphs)
        - Hex encoding

        Args:
            content: Content to check
        """
        # Base64 detection (long base64 strings might contain hidden instructions).
        # Standard and URL-safe alphabets, with or without padding.
        base64_pattern = r"[A-Za-z0-9+/_-]{30,}={0,2}"
        base64_matches = re.findall(base64_pattern, content)
        keywords = [keyword.lower() for keyword in self._suspicious_keywords]

        for match in base64_matches:
            data = match.rstrip("=")
            data += "=" * (-len(data) % 4)
            try:
                raw = (
                    base64.urlsafe_b64decode(data)
                    if "-" in data or "_" in data
                    else base64.b64decode(data)
                )
            except (binascii.Error, ValueError):
                continue
            # Report outside the decode step, so a strict-mode ValidationError propagates
            decoded = raw.decode("utf-8", errors="ignore")
            if any(keyword in decoded.lower() for keyword in keywords):
                self._handle_violation(
                    "Suspicious base64 encoded content detected",
                    "encoding_attack",
                    {"encoded": match[:50], "decoded_sample": decoded[:100]},
                )

        # Hidden characters. Non-ASCII text on its own is not flagged, so messages in any
        # language pass; only characters used to hide or disguise instructions are.
        hidden = _HIDDEN_CHARS_RE.search(content)
        if hidden:
            self._handle_violation(
                "Hidden or direction-changing unicode characters detected (possible obfuscation)",
                "encoding_attack",
                {"codepoint": f"U+{ord(hidden.group()):04X}"},
            )

        mixed = _MIXED_SCRIPT_RE.search(content)
        if mixed:
            self._handle_violation(
                "Latin and Cyrillic letters mixed inside a word (possible homoglyph obfuscation)",
                "encoding_attack",
                {"sample": mixed.group()},
            )

        # Hex encoding detection
        hex_pattern = r"(?:\\x[0-9a-fA-F]{2}){10,}"
        if re.search(hex_pattern, content):
            self._handle_violation(
                "Hex encoding detected (possible obfuscation)",
                "encoding_attack",
                {},
            )

    def _check_suspicious_keywords(self, content: str) -> None:
        """
        Check for suspicious keyword patterns.

        Args:
            content: Content to check
        """
        content_lower = content.lower()

        def _present(keyword: str) -> bool:
            # Whole words only: "DAN" must not match "dance" nor "STAN" "standard".
            return (
                re.search(rf"(?<!\w){re.escape(keyword.lower())}(?!\w)", content_lower) is not None
            )

        keyword_count = sum(1 for keyword in self._suspicious_keywords if _present(keyword))

        # If multiple suspicious keywords appear, flag it
        threshold = 3
        if keyword_count >= threshold:
            matched_keywords = [kw for kw in self._suspicious_keywords if _present(kw)]
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
        from tenxgraph.utils.callbacks import CallbackManager
        from tenxgraph.utils.validators import MessageContentValidator

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
        from tenxgraph.utils.validators import register_default_validators
        from tenxgraph.utils.callbacks import CallbackManager

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
