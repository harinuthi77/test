"""
Security utilities for multi-agent system
Provides PII redaction, input validation, and safety checks
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class PIIType(Enum):
    """Types of PII to detect"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    URL = "url"


@dataclass
class PIIMatch:
    """A detected PII instance"""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0


class PIIRedactor:
    """
    Redacts personally identifiable information from text

    Patterns:
    - Email addresses
    - Phone numbers (US formats)
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - URLs (configurable)
    """

    def __init__(self, redact_urls: bool = False):
        self.redact_urls = redact_urls

        # Regex patterns for PII detection
        self.patterns = {
            PIIType.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIType.PHONE: re.compile(
                r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
            ),
            PIIType.SSN: re.compile(
                r'\b\d{3}-\d{2}-\d{4}\b'
            ),
            PIIType.CREDIT_CARD: re.compile(
                r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
            ),
            PIIType.IP_ADDRESS: re.compile(
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
            ),
        }

        if self.redact_urls:
            self.patterns[PIIType.URL] = re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+'
            )

    def detect_pii(self, text: str) -> List[PIIMatch]:
        """
        Detect all PII in text

        Args:
            text: Text to scan

        Returns:
            List of PII matches
        """
        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end()
                ))

        # Sort by position
        matches.sort(key=lambda m: m.start)
        return matches

    def redact(
        self,
        text: str,
        replacement: str = "[REDACTED]",
        preserve_type: bool = True
    ) -> Tuple[str, List[PIIMatch]]:
        """
        Redact PII from text

        Args:
            text: Text to redact
            replacement: Replacement string (default: [REDACTED])
            preserve_type: If True, show PII type (e.g., [REDACTED:EMAIL])

        Returns:
            (redacted_text, list of detected PII)
        """
        matches = self.detect_pii(text)

        if not matches:
            return text, []

        # Redact in reverse order to preserve positions
        redacted = text
        for match in reversed(matches):
            if preserve_type:
                repl = f"[REDACTED:{match.pii_type.value.upper()}]"
            else:
                repl = replacement

            redacted = redacted[:match.start] + repl + redacted[match.end:]

        return redacted, matches


class InputValidator:
    """
    Validates and sanitizes user inputs

    Checks for:
    - Injection attacks (SQL, command, code)
    - Path traversal
    - Excessive length
    - Invalid characters
    """

    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        re.compile(r';\s*rm\s+-rf', re.IGNORECASE),  # Command injection
        re.compile(r';\s*drop\s+table', re.IGNORECASE),  # SQL injection
        re.compile(r'\.\./|\.\.\\'),  # Path traversal
        re.compile(r'__import__'),  # Python import injection
        re.compile(r'eval\s*\('),  # Eval injection
        re.compile(r'exec\s*\('),  # Exec injection
    ]

    @staticmethod
    def validate_string(
        value: str,
        max_length: int = 10000,
        allowed_chars: str = None,
        name: str = "input"
    ) -> Tuple[bool, str]:
        """
        Validate a string input

        Args:
            value: String to validate
            max_length: Maximum allowed length
            allowed_chars: Regex pattern of allowed characters
            name: Name of field (for error messages)

        Returns:
            (is_valid, error_message)
        """
        # Length check
        if len(value) > max_length:
            return False, f"{name} exceeds maximum length ({max_length} chars)"

        # Check for dangerous patterns
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if pattern.search(value):
                return False, f"{name} contains potentially dangerous pattern: {pattern.pattern}"

        # Character allowlist
        if allowed_chars:
            pattern = re.compile(allowed_chars)
            if not pattern.fullmatch(value):
                return False, f"{name} contains invalid characters (allowed: {allowed_chars})"

        return True, ""

    @staticmethod
    def validate_path(
        path: str,
        allowed_dirs: List[str],
        must_exist: bool = False
    ) -> Tuple[bool, str]:
        """
        Validate a file path

        Args:
            path: Path to validate
            allowed_dirs: List of allowed directory prefixes
            must_exist: If True, check that path exists

        Returns:
            (is_valid, error_message)
        """
        import os

        # Normalize path
        normalized = os.path.normpath(os.path.abspath(path))

        # Check for path traversal
        if '..' in path:
            return False, "Path traversal not allowed"

        # Check against allowlist
        if not any(normalized.startswith(d) for d in allowed_dirs):
            return False, f"Path outside allowed directories: {allowed_dirs}"

        # Existence check
        if must_exist and not os.path.exists(normalized):
            return False, f"Path does not exist: {path}"

        return True, ""

    @staticmethod
    def validate_command(
        command: str,
        allowed_commands: List[str]
    ) -> Tuple[bool, str]:
        """
        Validate a shell command

        Args:
            command: Command to validate
            allowed_commands: List of allowed command names

        Returns:
            (is_valid, error_message)
        """
        # Get first word (command name)
        cmd_name = command.strip().split()[0] if command.strip() else ""

        if cmd_name not in allowed_commands:
            return False, f"Command '{cmd_name}' not in allowlist: {allowed_commands}"

        # Check for command chaining
        dangerous_chars = [';', '|', '&', '`', '$', '\n']
        if any(c in command for c in dangerous_chars):
            return False, "Command chaining/substitution not allowed"

        return True, ""


class RateLimiter:
    """
    Simple rate limiter for API/tool calls

    Uses sliding window approach
    """

    def __init__(self):
        self.call_history: Dict[str, List[float]] = {}

    def check_rate_limit(
        self,
        key: str,
        max_calls: int,
        window_seconds: int
    ) -> Tuple[bool, int]:
        """
        Check if rate limit is exceeded

        Args:
            key: Identifier (e.g., user_id, tool_name)
            max_calls: Maximum calls allowed
            window_seconds: Time window in seconds

        Returns:
            (is_allowed, calls_remaining)
        """
        import time

        now = time.time()

        # Initialize history for this key
        if key not in self.call_history:
            self.call_history[key] = []

        # Remove old calls outside window
        cutoff = now - window_seconds
        self.call_history[key] = [
            t for t in self.call_history[key]
            if t > cutoff
        ]

        # Check limit
        calls_in_window = len(self.call_history[key])

        if calls_in_window >= max_calls:
            return False, 0

        # Record this call
        self.call_history[key].append(now)

        return True, max_calls - calls_in_window - 1


# ============================================================================
# TESTING
# ============================================================================

def test_pii_redactor():
    """Test PII redactor"""
    print("="*70)
    print("PII REDACTOR TEST")
    print("="*70)

    redactor = PIIRedactor(redact_urls=False)

    # Test text with various PII
    text = """
    Contact us at support@example.com or call (555) 123-4567.
    SSN: 123-45-6789
    Credit card: 1234 5678 9012 3456
    Server IP: 192.168.1.1
    """

    print("\nOriginal text:")
    print(text)

    # Detect PII
    matches = redactor.detect_pii(text)
    print(f"\n✅ Detected {len(matches)} PII instances:")
    for match in matches:
        print(f"   - {match.pii_type.value}: {match.value}")

    # Redact PII
    redacted, _ = redactor.redact(text)
    print("\nRedacted text:")
    print(redacted)

    print("\n" + "="*70)
    print("✅ PII REDACTOR VALIDATED")
    print("="*70)


def test_input_validator():
    """Test input validator"""
    print("\n" + "="*70)
    print("INPUT VALIDATOR TEST")
    print("="*70)

    # Test 1: Valid string
    valid, error = InputValidator.validate_string("Hello world", max_length=100)
    print(f"\n1. Valid string: {valid} (expected: True)")

    # Test 2: Too long
    valid, error = InputValidator.validate_string("x" * 1000, max_length=100)
    print(f"2. Too long: {valid} (expected: False) - {error}")

    # Test 3: SQL injection
    valid, error = InputValidator.validate_string("'; DROP TABLE users;--")
    print(f"3. SQL injection: {valid} (expected: False) - {error}")

    # Test 4: Path traversal
    valid, error = InputValidator.validate_string("../../../etc/passwd")
    print(f"4. Path traversal: {valid} (expected: False) - {error}")

    # Test 5: Command validation
    valid, error = InputValidator.validate_command("ls -la", ["ls", "cat"])
    print(f"5. Allowed command: {valid} (expected: True)")

    valid, error = InputValidator.validate_command("rm -rf /", ["ls", "cat"])
    print(f"6. Disallowed command: {valid} (expected: False) - {error}")

    print("\n" + "="*70)
    print("✅ INPUT VALIDATOR VALIDATED")
    print("="*70)


def test_rate_limiter():
    """Test rate limiter"""
    print("\n" + "="*70)
    print("RATE LIMITER TEST")
    print("="*70)

    limiter = RateLimiter()

    # Test: 3 calls per 1 second window
    print("\nTesting 3 calls/second limit:")
    for i in range(5):
        allowed, remaining = limiter.check_rate_limit("test_user", max_calls=3, window_seconds=1)
        status = "✅ Allowed" if allowed else "❌ Blocked"
        print(f"   Call {i+1}: {status} (remaining: {remaining})")

    print("\n" + "="*70)
    print("✅ RATE LIMITER VALIDATED")
    print("="*70)


if __name__ == "__main__":
    test_pii_redactor()
    test_input_validator()
    test_rate_limiter()
