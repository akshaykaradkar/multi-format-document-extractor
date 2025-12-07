"""
Abstract base class for all document parsers.

Defines the interface that all parsers must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from ..schemas import RawExtraction


class BaseParser(ABC):
    """
    Abstract base class for document parsers.

    All format-specific parsers inherit from this class and implement
    the parse() method to extract order data from their respective formats.
    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the parser with a file path.

        Args:
            file_path: Path to the document to parse
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is readable."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    @abstractmethod
    def parse(self) -> RawExtraction:
        """
        Parse the document and extract order data.

        Returns:
            RawExtraction object containing extracted data

        Raises:
            ParseError: If parsing fails
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions."""
        pass

    def is_supported(self) -> bool:
        """Check if the file extension is supported by this parser."""
        return self.file_path.suffix.lower() in self.supported_extensions


class ParseError(Exception):
    """Exception raised when document parsing fails."""

    def __init__(self, message: str, file_path: Path = None, original_error: Exception = None):
        self.message = message
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(self.message)
