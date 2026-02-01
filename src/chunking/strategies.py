"""Document chunking strategies for the RAG system."""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..models.base import Chunk, ChunkingConfig, ChunkingStrategy, Document
from ..utils.exceptions import ChunkingError
from ..utils.helpers import clean_text


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into chunks."""
        pass
    
    def _create_chunk(
        self,
        document: Document,
        content: str,
        chunk_index: int,
        start_char: int,
        end_char: int
    ) -> Chunk:
        """Create a chunk object with metadata."""
        return Chunk(
            document_id=document.id,
            content=clean_text(content),
            chunk_index=chunk_index,
            start_char=start_char,
            end_char=end_char,
            metadata=document.metadata
        )
    
    def _validate_chunk_size(self, content: str) -> bool:
        """Validate chunk meets size requirements."""
        content_length = len(content.strip())
        return (
            self.config.min_chunk_size <= content_length <= self.config.max_chunk_size
        )


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking strategy with overlap."""
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into fixed-size chunks with overlap."""
        text = document.content
        chunks = []
        chunk_index = 0
        
        start = 0
        while start < len(text):
            # Calculate end position
            end = min(start + self.config.chunk_size, len(text))
            
            # Extract chunk content
            chunk_content = text[start:end]
            
            # Skip chunks that are too small (except the last one)
            if len(chunk_content.strip()) < self.config.min_chunk_size and end < len(text):
                start += self.config.chunk_size - self.config.chunk_overlap
                continue
            
            # Create chunk
            if chunk_content.strip():
                chunk = self._create_chunk(
                    document=document,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start += self.config.chunk_size - self.config.chunk_overlap
        
        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive chunking strategy that respects text structure."""
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document recursively using multiple separators."""
        text = document.content
        chunks = []
        
        # Split text using recursive approach
        text_chunks = self._recursive_split(text, self.config.separators)
        
        # Create chunk objects
        current_pos = 0
        for chunk_index, chunk_content in enumerate(text_chunks):
            if chunk_content.strip():
                start_char = text.find(chunk_content, current_pos)
                if start_char == -1:
                    start_char = current_pos
                
                end_char = start_char + len(chunk_content)
                
                chunk = self._create_chunk(
                    document=document,
                    content=chunk_content,
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=end_char
                )
                chunks.append(chunk)
                current_pos = end_char
        
        return chunks
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using multiple separators."""
        if not separators:
            return [text] if text.strip() else []
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Split by current separator
        splits = text.split(separator)
        result = []
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            # If split is too large, try next separator
            if len(split) > self.config.chunk_size and remaining_separators:
                result.extend(self._recursive_split(split, remaining_separators))
            else:
                # If still too large and no more separators, force split
                if len(split) > self.config.max_chunk_size:
                    result.extend(self._force_split(split))
                else:
                    result.append(split)
        
        # Merge small chunks
        return self._merge_small_chunks(result)
    
    def _force_split(self, text: str) -> List[str]:
        """Force split text when it's too large."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - self.config.chunk_overlap
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge consecutive small chunks."""
        if not chunks:
            return []
        
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # If current chunk is too small and merging won't exceed max size
            if (
                len(current_chunk) < self.config.min_chunk_size and
                len(current_chunk) + len(next_chunk) <= self.config.max_chunk_size
            ):
                current_chunk += " " + next_chunk
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        merged.append(current_chunk)
        return merged


class SemanticChunker(BaseChunker):
    """Semantic chunking based on sentence boundaries and coherence."""
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into semantically coherent chunks."""
        text = document.content
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = ""
        current_sentences = []
        chunk_index = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.config.min_chunk_size:
                    start_char, end_char = self._find_chunk_positions(
                        text, current_sentences
                    )
                    
                    chunk = self._create_chunk(
                        document=document,
                        content=current_chunk,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence
                current_sentences = [sentence]
        
        # Handle remaining chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            start_char, end_char = self._find_chunk_positions(text, current_sentences)
            
            chunk = self._create_chunk(
                document=document,
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns."""
        # Simple sentence splitting pattern
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_chunk_positions(self, text: str, sentences: List[str]) -> Tuple[int, int]:
        """Find start and end positions of chunk in original text."""
        chunk_text = " ".join(sentences)
        start_char = text.find(sentences[0])
        if start_char == -1:
            start_char = 0
        
        end_char = start_char + len(chunk_text)
        return start_char, min(end_char, len(text))


class ParagraphChunker(BaseChunker):
    """Paragraph-based chunking strategy."""
    
    def chunk_document(self, document: Document) -> List[Chunk]:
        """Split document into paragraph-based chunks."""
        text = document.content
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        current_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Find paragraph position in original text
            para_start = text.find(paragraph, current_pos)
            if para_start == -1:
                para_start = current_pos
            
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.config.min_chunk_size:
                    chunk_start = text.find(current_chunk.split('\n\n')[0], current_pos)
                    chunk_end = chunk_start + len(current_chunk)
                    
                    chunk = self._create_chunk(
                        document=document,
                        content=current_chunk,
                        chunk_index=chunk_index,
                        start_char=chunk_start,
                        end_char=chunk_end
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with current paragraph
                current_chunk = paragraph
            
            current_pos = para_start + len(paragraph)
        
        # Handle remaining chunk
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunk_start = text.find(current_chunk.split('\n\n')[0])
            chunk_end = chunk_start + len(current_chunk)
            
            chunk = self._create_chunk(
                document=document,
                content=current_chunk,
                chunk_index=chunk_index,
                start_char=chunk_start,
                end_char=chunk_end
            )
            chunks.append(chunk)
        
        return chunks


class ChunkerFactory:
    """Factory for creating chunking strategies."""
    
    _strategies = {
        ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
        ChunkingStrategy.RECURSIVE: RecursiveChunker,
        ChunkingStrategy.SEMANTIC: SemanticChunker,
        ChunkingStrategy.PARAGRAPH: ParagraphChunker,
    }
    
    @classmethod
    def create_chunker(cls, config: ChunkingConfig) -> BaseChunker:
        """Create chunker instance based on strategy."""
        strategy_class = cls._strategies.get(config.strategy)
        
        if not strategy_class:
            raise ChunkingError(f"Unknown chunking strategy: {config.strategy}")
        
        return strategy_class(config)
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available chunking strategies."""
        return list(cls._strategies.keys())


def chunk_document(document: Document, config: ChunkingConfig) -> List[Chunk]:
    """Convenience function to chunk a document."""
    chunker = ChunkerFactory.create_chunker(config)
    return chunker.chunk_document(document)