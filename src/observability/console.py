"""
Structured console output helpers.

Provides a thin layer over print() that standardises formatting across the
pipeline without pulling in a full logging framework.

    from observability.console import console
    console.section("First Pass")
    console.step("Initializing DataLoader...")
    console.detail("loaded 42 000 sequences")
    console.success("first_pass saved")
    console.warn("neg_ctx skipped: file not found")
    console.error("candidates.pt missing")
"""

from typing import Optional


class Console:
    INDENT = "  "

    def section(self, title: str) -> None:
        """Print a prominent section header."""
        print(f"--- {title} ---")

    def step(self, message: str) -> None:
        """Print a top-level pipeline step (no indent)."""
        print(message)

    def detail(self, message: str) -> None:
        """Print an indented detail line."""
        print(f"{self.INDENT}{message}")

    def success(self, message: str) -> None:
        """Print an indented success line with a checkmark."""
        print(f"{self.INDENT}\u2713 {message}")

    def warn(self, message: str) -> None:
        """Print an indented warning line."""
        print(f"{self.INDENT}\u2717 {message}")

    def error(self, message: str, prefix: Optional[str] = None) -> None:
        """Print an error line with optional prefix."""
        tag = f"[{prefix}] " if prefix else ""
        print(f"{tag}Error: {message}")

    def kv(self, key: str, value: object) -> None:
        """Print an indented key-value pair."""
        print(f"{self.INDENT}{key}: {value}")

    def blank(self) -> None:
        """Print an empty line."""
        print("")


console = Console()
