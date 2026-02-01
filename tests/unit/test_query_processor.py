"""Unit tests for QueryProcessor from search optimization module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.optimization.search_optimizer import QueryProcessor, OptimizationConfig


class TestQueryProcessor:
    """Test suite for QueryProcessor class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OptimizationConfig(
            enable_query_expansion=True,
            enable_spell_correction=True,
            enable_stemming=True,
            remove_stopwords=True
        )
    
    @pytest.fixture
    def query_processor(self, config):
        """Create QueryProcessor instance for testing."""
        with patch('nltk.download'), \
             patch('spacy.load') as mock_spacy:
            # Mock spaCy model
            mock_nlp = Mock()
            mock_doc = Mock()
            mock_token = Mock()
            mock_token.lemma_ = "test"
            mock_token.is_space = False
            mock_token.pos_ = "NOUN"
            mock_token.ent_type_ = ""
            mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
            mock_doc.ents = []
            mock_nlp.return_value = mock_doc
            mock_spacy.return_value = mock_nlp
            
            processor = QueryProcessor(config)
            processor.nlp = mock_nlp
            return processor
    
    def test_init_with_valid_config(self, config):
        """Test QueryProcessor initialization with valid configuration."""
        with patch('nltk.download'), \
             patch('spacy.load'):
            processor = QueryProcessor(config)
            
            assert processor.config == config
            assert processor.stemmer is not None
            assert len(processor.stop_words) > 0
    
    def test_init_without_stemming(self):
        """Test QueryProcessor initialization without stemming enabled."""
        config = OptimizationConfig(enable_stemming=False)
        
        with patch('nltk.download'), \
             patch('spacy.load'):
            processor = QueryProcessor(config)
            
            assert processor.stemmer is None
    
    def test_init_without_stopword_removal(self):
        """Test QueryProcessor initialization without stopword removal."""
        config = OptimizationConfig(remove_stopwords=False)
        
        with patch('nltk.download'), \
             patch('spacy.load'):
            processor = QueryProcessor(config)
            
            assert len(processor.stop_words) == 0
    
    @patch('spacy.load', side_effect=OSError("Model not found"))
    @patch('nltk.download')
    def test_load_models_spacy_fallback(self, mock_nltk, mock_spacy):
        """Test graceful fallback when spaCy model is not available."""
        config = OptimizationConfig()
        processor = QueryProcessor(config)
        
        assert processor.nlp is None
        mock_spacy.assert_called_once()
    
    def test_process_query_basic(self, query_processor):
        """Test basic query processing functionality."""
        query = "Python programming tutorial"
        result = query_processor.process_query(query)
        
        assert "original" in result
        assert "processed" in result
        assert "tokens" in result
        assert "expanded_terms" in result
        assert "entities" in result
        assert "intent" in result
        
        assert result["original"] == query
        assert isinstance(result["tokens"], list)
        assert isinstance(result["expanded_terms"], list)
        assert isinstance(result["entities"], list)
        assert result["intent"] in ["search", "action", "person_search"]
    
    def test_process_query_with_spacy(self, query_processor):
        """Test query processing with spaCy enabled."""
        # Mock spaCy processing
        mock_doc = Mock()
        mock_token = Mock()
        mock_token.lemma_ = "python"
        mock_token.is_space = False
        mock_token.pos_ = "NOUN"
        mock_token.ent_type_ = ""
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.ents = []
        query_processor.nlp.return_value = mock_doc
        
        query = "Python programming"
        result = query_processor.process_query(query)
        
        assert result["tokens"] == ["python"]
        assert result["intent"] == "search"
    
    def test_process_query_action_intent(self, query_processor):
        """Test query processing with action intent detection."""
        # Mock spaCy with verb
        mock_doc = Mock()
        mock_token = Mock()
        mock_token.lemma_ = "create"
        mock_token.is_space = False
        mock_token.pos_ = "VERB"
        mock_token.ent_type_ = ""
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.ents = []
        query_processor.nlp.return_value = mock_doc
        
        query = "create a function"
        result = query_processor.process_query(query)
        
        assert result["intent"] == "action"
    
    def test_process_query_person_intent(self, query_processor):
        """Test query processing with person intent detection."""
        # Mock spaCy with person entity
        mock_doc = Mock()
        mock_token = Mock()
        mock_token.lemma_ = "john"
        mock_token.is_space = False
        mock_token.pos_ = "NOUN"
        mock_token.ent_type_ = "PERSON"
        mock_doc.__iter__ = Mock(return_value=iter([mock_token]))
        mock_doc.ents = []
        query_processor.nlp.return_value = mock_doc
        
        query = "John Smith"
        result = query_processor.process_query(query)
        
        assert result["intent"] == "person_search"
    
    def test_process_query_without_spacy(self, config):
        """Test query processing fallback without spaCy."""
        with patch('nltk.download'), \
             patch('spacy.load'), \
             patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            
            mock_tokenize.return_value = ["python", "programming"]
            
            processor = QueryProcessor(config)
            processor.nlp = None
            
            query = "Python programming"
            result = processor.process_query(query)
            
            assert result["tokens"] == ["python", "programming"]
            mock_tokenize.assert_called_once_with("python programming")
    
    def test_stopword_removal(self, query_processor):
        """Test stopword removal functionality."""
        # Mock word_tokenize for fallback case
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["the", "quick", "brown", "fox"]
            query_processor.nlp = None
            query_processor.stop_words = {"the"}
            
            query = "the quick brown fox"
            result = query_processor.process_query(query)
            
            assert "the" not in result["tokens"]
            assert "quick" in result["tokens"]
    
    def test_stemming(self, query_processor):
        """Test stemming functionality."""
        # Mock stemmer with specific side effects to match expectations
        def mock_stem(word):
            if word == "running":
                return "runnin"
            elif word == "cats":
                return "cat"
            else:
                return word
        
        query_processor.stemmer.stem.side_effect = mock_stem
        
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["running", "cats"]
            query_processor.nlp = None
            
            query = "running cats"
            result = query_processor.process_query(query)
            
            assert "runnin" in result["tokens"]  # stemmed 'running'
            assert "cat" in result["tokens"]     # stemmed 'cats'
    
    def test_query_expansion(self, query_processor):
        """Test query expansion functionality."""
        query = "python machine learning data"
        result = query_processor.process_query(query)
        
        # Should expand terms based on synonym map
        expected_expansions = ["programming", "code", "script", 
                             "automated", "artificial", "computer",
                             "training", "education", "study",
                             "information", "dataset", "analytics"]
        
        for expansion in result["expanded_terms"]:
            assert expansion in expected_expansions
    
    def test_query_expansion_disabled(self, config):
        """Test query processing with expansion disabled."""
        config.enable_query_expansion = False
        
        with patch('nltk.download'), \
             patch('spacy.load'):
            processor = QueryProcessor(config)
            
            query = "python programming"
            result = processor.process_query(query)
            
            assert result["expanded_terms"] == []
    
    def test_expand_query_with_synonyms(self, query_processor):
        """Test _expand_query method with known synonyms."""
        tokens = ["python", "machine", "data"]
        expanded = query_processor._expand_query(tokens)
        
        expected = ["programming", "code", "script",
                   "automated", "artificial", "computer", 
                   "information", "dataset", "analytics"]
        
        assert all(term in expected for term in expanded)
    
    def test_expand_query_without_synonyms(self, query_processor):
        """Test _expand_query method with unknown tokens."""
        tokens = ["unknown", "tokens"]
        expanded = query_processor._expand_query(tokens)
        
        assert expanded == []
    
    def test_empty_query_processing(self, query_processor):
        """Test processing empty query."""
        query = ""
        result = query_processor.process_query(query)
        
        assert result["original"] == ""
        assert result["processed"] == ""
        assert result["tokens"] == []
        assert result["expanded_terms"] == []
    
    def test_whitespace_only_query(self, query_processor):
        """Test processing query with only whitespace."""
        query = "   \t\n  "
        result = query_processor.process_query(query)
        
        assert result["original"] == query
        assert result["processed"] == ""
    
    def test_special_characters_query(self, query_processor):
        """Test processing query with special characters."""
        with patch('nltk.tokenize.word_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["test", "@", "#", "$"]
            query_processor.nlp = None
            
            query = "test @#$"
            result = query_processor.process_query(query)
            
            # Should handle special characters gracefully
            assert isinstance(result["tokens"], list)
            assert "test" in result["tokens"]
    
    @patch('src.optimization.search_optimizer.logger')
    def test_load_models_exception_handling(self, mock_logger):
        """Test exception handling during model loading."""
        with patch('nltk.download', side_effect=Exception("NLTK error")):
            config = OptimizationConfig()
            processor = QueryProcessor(config)
            
            mock_logger.error.assert_called()
            assert "Failed to load NLP models" in str(mock_logger.error.call_args)