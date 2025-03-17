import numpy as np
import re
from numpy.typing import NDArray
from typing import Optional, List
from abc import ABC, abstractmethod
from .utils import split_text


class BaseContextPartitioner(ABC):
    """
    A base class for partitioning a context into sources.

    Attributes:
        context (str):
            The context to partition.

    Methods:
        num_sources(self) -> int:
            Property. The number of sources within the context.
        split_context(self) -> None:
            Split the context into sources.
        get_source(self, index: int) -> str:
            Get a represention of the source corresponding to a given index.
        get_context(self, mask: Optional[NDArray] = None) -> str:
            Get a version of the context ablated according to the given mask.
        sources(self) -> List[str]:
            Property. A list of all sources within the context.
    """

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    @abstractmethod
    def num_sources(self) -> int:
        """The number of sources."""

    @abstractmethod
    def split_context(self) -> None:
        """Split the context into sources."""

    @abstractmethod
    def get_source(self, index: int) -> str:
        """Get a represention of the source corresponding to a given index."""

    @abstractmethod
    def get_context(self, mask: Optional[NDArray] = None):
        """Get a version of the context ablated according to the given mask."""

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]


class SimpleContextPartitioner(BaseContextPartitioner):
    """
    A simple context partitioner that splits the context into sources based on
    a separator.
    """

    def __init__(self, context: str, source_type: str = "sentence") -> None:
        super().__init__(context)
        self.source_type = source_type
        self._cache = {}

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        parts, separators, _ = split_text(self.context, self.source_type)
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += separator
            context += part
        return context
    
class SentencePeriodPartitioner(BaseContextPartitioner):
    """
    A context partitioner that splits the context into sources based on periods (`.`)
    and filters out sentences with fewer than 3 words.
    """

    def __init__(self, context: str) -> None:
        super().__init__(context)
        self._cache = {}

    def _preprocess_context(self, context: str) -> str:
        """
        Preprocess the context to remove unexpected formats, extra spaces, and non-standard characters.

        Args:
            context (str): The input context.

        Returns:
            str: The preprocessed context.
        """
        # Remove extra spaces, line breaks, and non-standard characters
        context = re.sub(r'\s+', ' ', context)  # Replace multiple spaces with a single space
        context = context.strip()  # Remove leading/trailing spaces
        return context

    def split_context(self):
        """Split text into parts and cache the parts and separators."""
        # Preprocess the context
        preprocessed_context = self._preprocess_context(self.context)

        # Split the context into parts based on periods
        parts, separators, _ = split_text(preprocessed_context, split_by="period")

        # Filter out parts with fewer than 3 words
        filtered_parts = []
        filtered_separators = []
        for part, separator in zip(parts, separators):
            if len(part.split()) >= 3:  # Keep only parts with 3 or more words
                filtered_parts.append(part)
                filtered_separators.append(separator)

        # Cache the filtered parts and separators
        self._cache["parts"] = filtered_parts
        self._cache["separators"] = filtered_separators

    @property
    def parts(self):
        if self._cache.get("parts") is None:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self):
        if self._cache.get("separators") is None:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[NDArray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        separators = np.array(self.separators)[mask]
        parts = np.array(self.parts)[mask]
        context = ""
        for i, (separator, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += separator
            context += part
        return context

class ParagraphPartitioner(BaseContextPartitioner):
    """
    A context partitioner that splits the context into sources based on paragraphs.
    Paragraphs are assumed to be separated by double line breaks.
    """
    def __init__(self, context: str) -> None:
        super().__init__(context)
        self._cache = {}

    def split_context(self):
        # Split on double newlines and filter out empty paragraphs.
        paragraphs = [p.strip() for p in self.context.split("\n\n") if p.strip()]
        # Create a list of separators. For simplicity, we assume "\n\n" between paragraphs.
        # The first paragraph has no separator.
        separators = [""] + ["\n\n" for _ in range(len(paragraphs) - 1)]
        self._cache["parts"] = paragraphs
        self._cache["separators"] = separators

    @property
    def parts(self) -> List[str]:
        if "parts" not in self._cache:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self) -> List[str]:
        if "separators" not in self._cache:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[np.ndarray] = None):
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        # Select only the parts and separators that are not masked out.
        parts = np.array(self.parts)[mask]
        separators = np.array(self.separators)[mask]
        context = ""
        for i, (sep, part) in enumerate(zip(separators, parts)):
            if i > 0:
                context += sep
            context += part
        return context

class CustomPartitioner(BaseContextPartitioner):
    def __init__(self, context: str) -> None:
        super().__init__(context)
        self._cache = {}

    def _preprocess_context(self, context: str) -> str:
        # Replace double newlines with a single newline
        context = context.replace("\n\n", "\n")
        # Replace multiple consecutive spaces with one space
        context = re.sub(r'\s+', ' ', context)
        return context.strip()

    def split_context(self) -> None:
        preprocessed = self._preprocess_context(self.context)
        # Split by periods and newline, while keeping the delimiters
        tokens = re.split(r'([.\n])', preprocessed)
        parts = []
        separators = []
        i = 0
        while i < len(tokens):
            # Get the text segment and trim it
            segment = tokens[i].strip()
            delimiter = ""
            if i + 1 < len(tokens):
                delimiter = tokens[i + 1]
            # Only add non-empty segments
            if segment:
                parts.append(segment + delimiter)
                separators.append(delimiter)
            i += 2  # move to the next text segment (skip delimiter)
        self._cache["parts"] = parts
        self._cache["separators"] = separators

    @property
    def parts(self) -> List[str]:
        if "parts" not in self._cache:
            self.split_context()
        return self._cache["parts"]

    @property
    def separators(self) -> List[str]:
        if "separators" not in self._cache:
            self.split_context()
        return self._cache["separators"]

    @property
    def num_sources(self) -> int:
        return len(self.parts)

    def get_source(self, index: int) -> str:
        return self.parts[index]

    def get_context(self, mask: Optional[np.ndarray] = None) -> str:
        if mask is None:
            mask = np.ones(self.num_sources, dtype=bool)
        # Use NumPy arrays to filter parts and separators based on the mask.
        selected_parts = np.array(self.parts)[mask]
        selected_seps = np.array(self.separators)[mask]
        context = ""
        for i, (sep, part) in enumerate(zip(selected_seps, selected_parts)):
            if i > 0:
                context += sep
            context += part
        return context