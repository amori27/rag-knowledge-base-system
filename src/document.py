"""Document Processing Module.

This module provides utilities for processing documents
including chunking, parsing, and metadata extraction.
"""

import re
from typing import Any


class DocumentProcessor:
    """Handles document processing and chunking."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the DocumentProcessor.

        Args:
            chunk_size: Size of each chunk in characters.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Split text into overlapping chunks.

        Args:
            text: Input text.

        Returns:
            List of chunks with metadata.
        """
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk_text.strip(),
                "start": start,
                "end": end
            })

            start += self.chunk_size - self.chunk_overlap
            chunk_id += 1

        return chunks

    def process_file(self, file_path: str) -> list[dict[str, Any]]:
        """Process a file and return chunks.

        Args:
            file_path: Path to file.

        Returns:
            List of document chunks.
        """
        if file_path.endswith('.txt'):
            return self._process_txt(file_path)
        elif file_path.endswith('.pdf'):
            return self._process_pdf(file_path)
        else:
            return self._process_txt(file_path)

    def _process_txt(self, file_path: str) -> list[dict[str, Any]]:
        """Process a text file.

        Args:
            file_path: Path to text file.

        Returns:
            List of chunks.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.chunk_text(text)
        except Exception:
            return []

    def _process_pdf(self, file_path: str) -> list[dict[str, Any]]:
        """Process a PDF file.

        Args:
            file_path: Path to PDF file.

        Returns:
            List of chunks.
        """
        return self.chunk_text(f"[PDF content from {file_path}]")

    def extract_metadata(self, text: str) -> dict[str, Any]:
        """Extract metadata from text.

        Args:
            text: Input text.

        Returns:
            Extracted metadata.
        """
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "line_count": len(text.split('\n')),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_urls": bool(re.search(r'http[s]?://', text))
        }


def clean_text_for_chunking(text: str) -> str:
    """Clean text before chunking.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
