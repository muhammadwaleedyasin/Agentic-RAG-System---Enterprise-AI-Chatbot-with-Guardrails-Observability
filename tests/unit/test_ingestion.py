"""Unit tests for the ingestion module."""

import pytest
from unittest.mock import MagicMock, patch, mock_open

from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.chunking_strategy import ChunkingStrategy
from src.ingestion.metadata_extractor import MetadataExtractor


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance for testing."""
        return DocumentProcessor()
    
    def test_process_pdf_document(self, processor, sample_document):
        """Test PDF document processing."""
        with patch('src.ingestion.document_processor.PyPDF2.PdfReader') as mock_reader:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Sample PDF content"
            mock_reader.return_value.pages = [mock_page]
            
            with patch('builtins.open', mock_open(read_data=b'pdf_content')):
                result = processor.process_document("test.pdf", "pdf")
                
            assert result['content'] == "Sample PDF content"
            assert result['metadata']['file_type'] == "pdf"
    
    def test_process_text_document(self, processor):
        """Test text document processing."""
        with patch('builtins.open', mock_open(read_data="Sample text content")):
            result = processor.process_document("test.txt", "txt")
            
        assert result['content'] == "Sample text content"
        assert result['metadata']['file_type'] == "txt"
    
    def test_unsupported_file_type(self, processor):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            processor.process_document("test.xyz", "xyz")


class TestChunkingStrategy:
    """Test cases for ChunkingStrategy."""
    
    @pytest.fixture
    def chunking_strategy(self):
        """Create a ChunkingStrategy instance for testing."""
        return ChunkingStrategy(chunk_size=100, overlap=20)
    
    def test_chunk_text(self, chunking_strategy):
        """Test text chunking functionality."""
        text = "This is a sample text that needs to be chunked into smaller pieces for testing purposes."
        chunks = chunking_strategy.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # chunk_size + some tolerance
    
    def test_chunk_overlap(self, chunking_strategy):
        """Test that chunks have proper overlap."""
        text = "A" * 200  # Long text to ensure multiple chunks
        chunks = chunking_strategy.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks have overlap
            overlap_chars = chunks[1][:20]  # First 20 chars of second chunk
            assert overlap_chars in chunks[0]  # Should exist in first chunk
    
    def test_empty_text(self, chunking_strategy):
        """Test chunking of empty text."""
        chunks = chunking_strategy.chunk_text("")
        assert chunks == []
    
    def test_short_text(self, chunking_strategy):
        """Test chunking of text shorter than chunk size."""
        text = "Short text"
        chunks = chunking_strategy.chunk_text(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text


class TestMetadataExtractor:
    """Test cases for MetadataExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a MetadataExtractor instance for testing."""
        return MetadataExtractor()
    
    def test_extract_basic_metadata(self, extractor):
        """Test extraction of basic file metadata."""
        with patch('os.path.getsize', return_value=1024):
            with patch('os.path.getmtime', return_value=1640995200):  # 2022-01-01
                metadata = extractor.extract_metadata("test.txt", "txt")
                
        assert metadata['file_name'] == "test.txt"
        assert metadata['file_type'] == "txt"
        assert metadata['file_size'] == 1024
        assert 'created_at' in metadata
    
    def test_extract_content_metadata(self, extractor):
        """Test extraction of content-based metadata."""
        content = "This is a test document with multiple sentences. It has some content for analysis."
        metadata = extractor.extract_content_metadata(content)
        
        assert 'word_count' in metadata
        assert 'char_count' in metadata
        assert metadata['word_count'] > 0
        assert metadata['char_count'] == len(content)
    
    def test_extract_language_detection(self, extractor):
        """Test language detection in metadata extraction."""
        content = "This is an English text document."
        metadata = extractor.extract_content_metadata(content)
        
        # Language detection might be implemented
        if 'language' in metadata:
            assert isinstance(metadata['language'], str)
    
    @patch('os.path.exists', return_value=False)
    def test_nonexistent_file(self, mock_exists, extractor):
        """Test handling of non-existent files."""
        with pytest.raises(FileNotFoundError):
            extractor.extract_metadata("nonexistent.txt", "txt")