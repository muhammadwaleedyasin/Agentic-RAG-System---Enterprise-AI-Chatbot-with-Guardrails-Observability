"""Unit tests for DocumentProcessor."""

import pytest
from unittest.mock import MagicMock, patch, mock_open
import io
from pathlib import Path

from src.core.document_processor import DocumentProcessor
from src.utils.exceptions import DocumentProcessingError


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""

    @pytest.fixture
    def document_processor(self):
        """Create DocumentProcessor instance for testing."""
        return DocumentProcessor()

    @pytest.mark.unit
    def test_document_processor_initialization(self, document_processor):
        """Test DocumentProcessor initialization."""
        assert document_processor.supported_formats is not None
        assert len(document_processor.supported_formats) > 0
        assert 'txt' in document_processor.supported_formats
        assert 'pdf' in document_processor.supported_formats

    @pytest.mark.unit
    def test_process_text_file_success(self, document_processor):
        """Test successful text file processing."""
        text_content = "This is a test document.\nWith multiple lines.\nAnd some content."
        
        with patch('builtins.open', mock_open(read_data=text_content)):
            result = document_processor.process_file("test.txt")
            
            assert result["content"] == text_content
            assert result["metadata"]["file_type"] == "txt"
            assert result["metadata"]["file_name"] == "test.txt"
            assert "file_size" in result["metadata"]

    @pytest.mark.unit
    def test_process_pdf_file_success(self, document_processor):
        """Test successful PDF file processing."""
        with patch('src.core.document_processor.PyPDF2.PdfReader') as mock_pdf:
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF content here"
            mock_pdf.return_value.pages = [mock_page, mock_page]
            
            with patch('builtins.open', mock_open(read_data=b"fake pdf data")):
                result = document_processor.process_file("test.pdf")
                
                assert result["content"] == "PDF content herePDF content here"
                assert result["metadata"]["file_type"] == "pdf"
                assert result["metadata"]["file_name"] == "test.pdf"
                assert result["metadata"]["num_pages"] == 2

    @pytest.mark.unit
    def test_process_docx_file_success(self, document_processor):
        """Test successful DOCX file processing."""
        with patch('src.core.document_processor.Document') as mock_docx:
            mock_paragraph1 = MagicMock()
            mock_paragraph1.text = "First paragraph"
            mock_paragraph2 = MagicMock()
            mock_paragraph2.text = "Second paragraph"
            
            mock_docx.return_value.paragraphs = [mock_paragraph1, mock_paragraph2]
            
            result = document_processor.process_file("test.docx")
            
            assert result["content"] == "First paragraph\nSecond paragraph"
            assert result["metadata"]["file_type"] == "docx"
            assert result["metadata"]["file_name"] == "test.docx"

    @pytest.mark.unit
    def test_process_markdown_file_success(self, document_processor):
        """Test successful Markdown file processing."""
        md_content = """# Header 1
        
## Header 2

This is **bold** text and *italic* text.

- List item 1
- List item 2

[Link](http://example.com)
"""
        
        with patch('builtins.open', mock_open(read_data=md_content)):
            result = document_processor.process_file("test.md")
            
            assert "Header 1" in result["content"]
            assert "bold" in result["content"]
            assert result["metadata"]["file_type"] == "md"
            assert result["metadata"]["file_name"] == "test.md"

    @pytest.mark.unit
    def test_process_unsupported_file_format(self, document_processor):
        """Test processing unsupported file format."""
        with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
            document_processor.process_file("test.xyz")

    @pytest.mark.unit
    def test_process_nonexistent_file(self, document_processor):
        """Test processing nonexistent file."""
        with patch('builtins.open', side_effect=FileNotFoundError):
            with pytest.raises(DocumentProcessingError, match="File not found"):
                document_processor.process_file("nonexistent.txt")

    @pytest.mark.unit
    def test_process_corrupted_pdf(self, document_processor):
        """Test processing corrupted PDF file."""
        with patch('src.core.document_processor.PyPDF2.PdfReader') as mock_pdf:
            mock_pdf.side_effect = Exception("PDF parsing error")
            
            with patch('builtins.open', mock_open(read_data=b"corrupted pdf")):
                with pytest.raises(DocumentProcessingError, match="Failed to process PDF"):
                    document_processor.process_file("corrupted.pdf")

    @pytest.mark.unit
    def test_process_empty_file(self, document_processor):
        """Test processing empty file."""
        with patch('builtins.open', mock_open(read_data="")):
            with pytest.raises(DocumentProcessingError, match="File is empty"):
                document_processor.process_file("empty.txt")

    @pytest.mark.unit
    def test_extract_metadata_from_file(self, document_processor):
        """Test metadata extraction from file."""
        file_path = "test_document.txt"
        file_size = 1024
        
        with patch('os.path.getsize', return_value=file_size):
            with patch('os.path.getmtime', return_value=1640995200.0):  # 2022-01-01
                metadata = document_processor._extract_metadata(file_path, "txt")
                
                assert metadata["file_name"] == "test_document.txt"
                assert metadata["file_type"] == "txt"
                assert metadata["file_size"] == file_size
                assert "created_at" in metadata

    @pytest.mark.unit
    def test_clean_text_content(self, document_processor):
        """Test text content cleaning."""
        dirty_text = """
        This is    a   text   with   extra   spaces.
        
        
        And    multiple   blank   lines.
        
        Also\ttabs\tand\nnewlines\r\n.
        """
        
        cleaned = document_processor._clean_text(dirty_text)
        
        # Should normalize whitespace and remove excessive blank lines
        assert "   " not in cleaned  # No triple spaces
        assert "\t" not in cleaned   # No tabs
        assert "\r" not in cleaned   # No carriage returns
        assert cleaned.strip() != ""

    @pytest.mark.unit
    def test_process_file_with_encoding_issues(self, document_processor):
        """Test processing file with encoding issues."""
        # Mock file with encoding issues
        with patch('builtins.open', side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid')):
            # Should try alternative encodings
            with patch('builtins.open', mock_open(read_data="Content with special chars")):
                result = document_processor.process_file("encoded.txt")
                assert result["content"] == "Content with special chars"

    @pytest.mark.unit
    def test_process_large_file(self, document_processor):
        """Test processing large file."""
        large_content = "This is a large file. " * 10000  # ~230KB
        
        with patch('builtins.open', mock_open(read_data=large_content)):
            with patch('os.path.getsize', return_value=230000):
                result = document_processor.process_file("large.txt")
                
                assert len(result["content"]) > 200000
                assert result["metadata"]["file_size"] == 230000

    @pytest.mark.unit
    def test_process_file_with_custom_metadata(self, document_processor):
        """Test processing file with custom metadata."""
        custom_metadata = {
            "author": "John Doe",
            "category": "technical",
            "tags": ["python", "machine learning"]
        }
        
        with patch('builtins.open', mock_open(read_data="Test content")):
            result = document_processor.process_file("test.txt", metadata=custom_metadata)
            
            assert result["metadata"]["author"] == "John Doe"
            assert result["metadata"]["category"] == "technical"
            assert result["metadata"]["tags"] == ["python", "machine learning"]

    @pytest.mark.unit
    def test_batch_process_files(self, document_processor):
        """Test batch processing of multiple files."""
        file_paths = ["doc1.txt", "doc2.txt", "doc3.pdf"]
        
        with patch.object(document_processor, 'process_file') as mock_process:
            mock_process.return_value = {
                "content": "Test content",
                "metadata": {"file_type": "txt"}
            }
            
            results = document_processor.batch_process_files(file_paths)
            
            assert len(results) == 3
            assert mock_process.call_count == 3

    @pytest.mark.unit
    def test_process_file_from_bytes(self, document_processor):
        """Test processing file from byte content."""
        file_bytes = b"This is byte content for testing."
        filename = "test.txt"
        
        result = document_processor.process_bytes(file_bytes, filename)
        
        assert result["content"] == "This is byte content for testing."
        assert result["metadata"]["file_name"] == "test.txt"
        assert result["metadata"]["file_type"] == "txt"

    @pytest.mark.unit
    def test_process_file_from_stream(self, document_processor):
        """Test processing file from stream."""
        content = "Stream content for testing."
        stream = io.StringIO(content)
        
        result = document_processor.process_stream(stream, "test.txt", "txt")
        
        assert result["content"] == content
        assert result["metadata"]["file_name"] == "test.txt"
        assert result["metadata"]["file_type"] == "txt"

    @pytest.mark.unit
    def test_get_supported_formats(self, document_processor):
        """Test getting supported file formats."""
        formats = document_processor.get_supported_formats()
        
        assert isinstance(formats, list)
        assert "txt" in formats
        assert "pdf" in formats
        assert "docx" in formats
        assert "md" in formats

    @pytest.mark.unit
    def test_validate_file_format(self, document_processor):
        """Test file format validation."""
        # Valid formats
        assert document_processor.is_supported_format("test.txt") is True
        assert document_processor.is_supported_format("document.pdf") is True
        assert document_processor.is_supported_format("report.docx") is True
        
        # Invalid formats
        assert document_processor.is_supported_format("image.jpg") is False
        assert document_processor.is_supported_format("data.xlsx") is False

    @pytest.mark.unit
    def test_extract_text_statistics(self, document_processor):
        """Test text statistics extraction."""
        text = """This is a test document. It has multiple sentences.
        And multiple paragraphs as well.
        
        This is another paragraph with more content."""
        
        stats = document_processor._get_text_statistics(text)
        
        assert stats["char_count"] == len(text)
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0
        assert stats["paragraph_count"] == 2

    @pytest.mark.unit
    def test_content_type_detection(self, document_processor):
        """Test automatic content type detection."""
        # Technical content
        tech_content = "Machine learning algorithms use neural networks and deep learning."
        tech_type = document_processor._detect_content_type(tech_content)
        assert "technical" in tech_type.lower() or "ml" in tech_type.lower()
        
        # Business content
        business_content = "Revenue increased by 15% this quarter due to improved sales strategies."
        business_type = document_processor._detect_content_type(business_content)
        assert "business" in business_type.lower() or "financial" in business_type.lower()

    @pytest.mark.unit
    def test_language_detection(self, document_processor):
        """Test language detection in documents."""
        english_text = "This is an English document with standard vocabulary."
        detected_lang = document_processor._detect_language(english_text)
        assert detected_lang == "en" or detected_lang == "english"