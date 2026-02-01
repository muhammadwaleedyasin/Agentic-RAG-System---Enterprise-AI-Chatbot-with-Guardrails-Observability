"""Unit tests for chunking strategies."""

import pytest
from unittest.mock import MagicMock
from src.chunking.strategies import (
    ChunkingStrategy, 
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    SemanticChunker,
    RecursiveCharacterChunker
)


class TestFixedSizeChunker:
    """Test cases for FixedSizeChunker."""

    @pytest.fixture
    def chunker(self):
        """Create FixedSizeChunker instance."""
        return FixedSizeChunker(chunk_size=100, overlap=20)

    @pytest.mark.unit
    def test_fixed_size_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20

    @pytest.mark.unit
    def test_chunk_short_text(self, chunker):
        """Test chunking of short text."""
        text = "This is a short text that should not be chunked."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["metadata"]["chunk_index"] == 0

    @pytest.mark.unit
    def test_chunk_long_text(self, chunker):
        """Test chunking of long text."""
        text = "A" * 250  # 250 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 3  # With overlap
        assert len(chunks[0]["content"]) == 100
        assert len(chunks[1]["content"]) == 100
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[1]["metadata"]["chunk_index"] == 1

    @pytest.mark.unit
    def test_chunk_with_overlap(self, chunker):
        """Test chunking with overlap."""
        text = "0123456789" * 15  # 150 characters
        chunks = chunker.chunk_text(text, overlap=10)
        
        assert len(chunks) == 2
        # Second chunk should start 10 characters before the end of first chunk
        assert chunks[1]["content"].startswith(chunks[0]["content"][-10:])


class TestSentenceChunker:
    """Test cases for SentenceChunker."""

    @pytest.fixture
    def chunker(self):
        """Create SentenceChunker instance."""
        return SentenceChunker(max_chunk_size=200)

    @pytest.mark.unit
    def test_sentence_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.max_chunk_size == 200

    @pytest.mark.unit
    def test_chunk_by_sentences(self, chunker):
        """Test chunking by sentences."""
        text = """
        This is the first sentence. This is the second sentence.
        This is the third sentence. This is the fourth sentence.
        This is the fifth sentence.
        """
        
        chunks = chunker.chunk_text(text.strip())
        
        assert len(chunks) >= 1
        # Check that sentences are not broken
        for chunk in chunks:
            assert chunk["content"].count('.') >= 1  # At least one complete sentence

    @pytest.mark.unit
    def test_respect_max_chunk_size(self, chunker):
        """Test that chunks respect maximum size."""
        long_sentences = [
            "This is a very long sentence that contains many words and should be handled properly by the sentence chunker." * 2
        ]
        
        for sentence in long_sentences:
            chunks = chunker.chunk_text(sentence)
            for chunk in chunks:
                assert len(chunk["content"]) <= chunker.max_chunk_size * 1.1  # Allow 10% tolerance


class TestParagraphChunker:
    """Test cases for ParagraphChunker."""

    @pytest.fixture
    def chunker(self):
        """Create ParagraphChunker instance."""
        return ParagraphChunker(max_chunk_size=500)

    @pytest.mark.unit
    def test_paragraph_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.max_chunk_size == 500

    @pytest.mark.unit
    def test_chunk_by_paragraphs(self, chunker):
        """Test chunking by paragraphs."""
        text = """
        This is the first paragraph with multiple sentences.
        It contains important information about the topic.

        This is the second paragraph that discusses different aspects.
        It also contains relevant information.

        This is the third paragraph with concluding thoughts.
        It summarizes the main points discussed.
        """
        
        chunks = chunker.chunk_text(text.strip())
        
        assert len(chunks) >= 1
        # Paragraphs should be preserved as units when possible
        for chunk in chunks:
            assert '\n\n' not in chunk["content"] or len(chunk["content"]) > chunker.max_chunk_size

    @pytest.mark.unit
    def test_handle_large_paragraphs(self, chunker):
        """Test handling of paragraphs larger than max chunk size."""
        large_paragraph = "This is a very long paragraph. " * 50  # Create large paragraph
        
        chunks = chunker.chunk_text(large_paragraph)
        
        # Should split large paragraph
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["content"]) <= chunker.max_chunk_size * 1.1


class TestSemanticChunker:
    """Test cases for SemanticChunker."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = MagicMock()
        service.embed_text.return_value = [0.1] * 384
        return service

    @pytest.fixture
    def chunker(self, mock_embedding_service):
        """Create SemanticChunker instance."""
        return SemanticChunker(
            embedding_service=mock_embedding_service,
            similarity_threshold=0.7,
            max_chunk_size=300
        )

    @pytest.mark.unit
    def test_semantic_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.similarity_threshold == 0.7
        assert chunker.max_chunk_size == 300
        assert chunker.embedding_service is not None

    @pytest.mark.unit
    def test_semantic_chunking(self, chunker):
        """Test semantic-based chunking."""
        text = """
        Machine learning is a subset of artificial intelligence.
        It enables computers to learn from data without explicit programming.
        
        Natural language processing is another AI field.
        It focuses on understanding human language.
        
        Computer vision deals with image and video analysis.
        It allows machines to interpret visual information.
        """
        
        # Mock similarity calculations
        chunker.embedding_service.embed_text.side_effect = [
            [0.1] * 384,  # ML sentence 1
            [0.12] * 384,  # ML sentence 2 (similar)
            [0.5] * 384,   # NLP sentence 1 (different)
            [0.52] * 384,  # NLP sentence 2 (similar to NLP)
            [0.8] * 384,   # CV sentence 1 (different)
            [0.82] * 384   # CV sentence 2 (similar to CV)
        ]
        
        chunks = chunker.chunk_text(text.strip())
        
        assert len(chunks) >= 1
        # Verify that embedding service was called
        assert chunker.embedding_service.embed_text.call_count > 0


class TestRecursiveCharacterChunker:
    """Test cases for RecursiveCharacterChunker."""

    @pytest.fixture
    def chunker(self):
        """Create RecursiveCharacterChunker instance."""
        return RecursiveCharacterChunker(
            chunk_size=100,
            overlap=20,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @pytest.mark.unit
    def test_recursive_chunker_initialization(self, chunker):
        """Test chunker initialization."""
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
        assert len(chunker.separators) == 5

    @pytest.mark.unit
    def test_recursive_splitting(self, chunker):
        """Test recursive splitting behavior."""
        text = """
        First paragraph with multiple sentences. This is another sentence.

        Second paragraph here. It also has multiple sentences.

        Third paragraph content. Final sentence here.
        """
        
        chunks = chunker.chunk_text(text.strip())
        
        assert len(chunks) >= 1
        # Should prefer paragraph breaks over other separators
        for chunk in chunks:
            content = chunk["content"]
            # Check that natural breaks are preserved when possible
            if len(content) < chunker.chunk_size:
                assert not content.endswith(" paragraph")  # Shouldn't break mid-word

    @pytest.mark.unit
    def test_fallback_to_character_splitting(self, chunker):
        """Test fallback to character splitting for long words."""
        # Text with very long word that can't be split naturally
        text = "normalword " + "x" * 200 + " normalword"
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 2  # Should be split
        # Verify that even the long word is handled
        total_content = "".join(chunk["content"] for chunk in chunks)
        assert "x" * 200 in total_content

    @pytest.mark.unit
    def test_overlap_preservation(self, chunker):
        """Test that overlap is preserved correctly."""
        text = "word " * 50  # 250 characters (5 chars per word)
        
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check overlap between consecutive chunks
            for i in range(len(chunks) - 1):
                chunk1_end = chunks[i]["content"][-chunker.overlap:]
                chunk2_start = chunks[i+1]["content"][:chunker.overlap]
                # Should have some overlap
                assert any(word in chunk2_start for word in chunk1_end.split())


class TestChunkingUtilities:
    """Test utility functions for chunking."""

    @pytest.mark.unit
    def test_chunk_metadata_generation(self):
        """Test chunk metadata generation."""
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Test content for metadata generation."
        
        chunks = chunker.chunk_text(text, document_id="test_doc")
        
        chunk = chunks[0]
        assert "chunk_index" in chunk["metadata"]
        assert "start_pos" in chunk["metadata"]
        assert "end_pos" in chunk["metadata"]
        assert "chunk_size" in chunk["metadata"]
        assert chunk["metadata"]["document_id"] == "test_doc"

    @pytest.mark.unit
    def test_chunk_position_tracking(self):
        """Test chunk position tracking."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=10)
        text = "A" * 120  # 120 characters
        
        chunks = chunker.chunk_text(text)
        
        # Verify position tracking
        assert chunks[0]["metadata"]["start_pos"] == 0
        assert chunks[0]["metadata"]["end_pos"] == 50
        
        if len(chunks) > 1:
            # Second chunk should start at position accounting for overlap
            expected_start = 50 - 10  # chunk_size - overlap
            assert chunks[1]["metadata"]["start_pos"] == expected_start

    @pytest.mark.unit
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        chunker = FixedSizeChunker(chunk_size=100)
        
        # Empty string
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
        
        # Only whitespace
        chunks = chunker.chunk_text("   \n  \t  ")
        assert len(chunks) == 0 or all(not chunk["content"].strip() for chunk in chunks)

    @pytest.mark.unit
    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=0)
        
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=-1)

    @pytest.mark.unit
    def test_overlap_validation(self):
        """Test overlap validation."""
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)  # Overlap equals chunk size
        
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=150)  # Overlap exceeds chunk size

    @pytest.mark.unit
    def test_chunking_preserves_content(self):
        """Test that chunking preserves all content."""
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)  # No overlap for exact comparison
        text = "This is test content that will be chunked and should be preserved completely."
        
        chunks = chunker.chunk_text(text)
        reconstructed = "".join(chunk["content"] for chunk in chunks)
        
        assert reconstructed == text

    @pytest.mark.unit 
    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        chunker = FixedSizeChunker(chunk_size=50)
        
        # Text with unicode characters
        text = "Hello 世界! This is a test with émojis 🚀 and spëcial characters."
        
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        # Verify unicode is preserved
        reconstructed = "".join(chunk["content"] for chunk in chunks)
        assert "世界" in reconstructed
        assert "🚀" in reconstructed
        assert "spëcial" in reconstructed

    @pytest.mark.unit
    def test_chunking_strategy_factory(self):
        """Test chunking strategy factory pattern."""
        strategies = {
            "fixed": FixedSizeChunker,
            "sentence": SentenceChunker,
            "paragraph": ParagraphChunker,
            "recursive": RecursiveCharacterChunker
        }
        
        for strategy_name, strategy_class in strategies.items():
            # Test that each strategy can be instantiated
            if strategy_name == "semantic":
                continue  # Skip semantic as it requires embedding service
            
            chunker = strategy_class(chunk_size=100)
            assert isinstance(chunker, ChunkingStrategy)
            
            # Test basic functionality
            chunks = chunker.chunk_text("Test content for strategy " + strategy_name)
            assert len(chunks) >= 1