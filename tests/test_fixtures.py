"""Test fixtures and data generators for comprehensive testing."""

import pytest
import asyncio
import tempfile
import json
import csv
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import random
import string


class TestDataGenerator:
    """Generate test data for various testing scenarios."""
    
    @staticmethod
    def generate_sample_documents(count: int = 100) -> List[Dict[str, Any]]:
        """Generate sample documents for testing."""
        documents = []
        topics = [
            "machine learning", "artificial intelligence", "data science",
            "deep learning", "natural language processing", "computer vision",
            "neural networks", "statistics", "python programming", "databases"
        ]
        
        for i in range(count):
            topic = random.choice(topics)
            content = f"""
            This is document {i} about {topic}. 
            {topic.title()} is an important field in computer science and data analysis.
            It involves various techniques and methodologies for analyzing and processing data.
            The applications of {topic} are vast and include areas such as automation,
            prediction, classification, and optimization. Many companies and researchers
            are actively working on improving {topic} technologies to solve real-world problems.
            """
            
            documents.append({
                "id": f"doc_{i:04d}",
                "title": f"Introduction to {topic.title()} - Part {i}",
                "content": content.strip(),
                "metadata": {
                    "topic": topic,
                    "author": f"Author_{i % 10}",
                    "created_at": (datetime.now() - timedelta(days=i)).isoformat(),
                    "word_count": len(content.split()),
                    "category": "technical",
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                    "tags": [topic, "tutorial", "guide"]
                }
            })
        
        return documents
    
    @staticmethod
    def generate_qa_pairs(count: int = 50) -> List[Dict[str, Any]]:
        """Generate question-answer pairs for evaluation."""
        qa_templates = [
            {
                "question": "What is {topic}?",
                "answer": "{topic.title()} is a field of study that focuses on...",
                "contexts": [
                    "{topic.title()} is an important concept in computer science.",
                    "The applications of {topic} include various domains."
                ]
            },
            {
                "question": "How does {topic} work?",
                "answer": "{topic.title()} works by using algorithms and techniques to...",
                "contexts": [
                    "{topic.title()} involves complex algorithms and data processing.",
                    "The methodology of {topic} includes several key steps."
                ]
            },
            {
                "question": "What are the applications of {topic}?",
                "answer": "Applications of {topic} include automation, prediction, and analysis.",
                "contexts": [
                    "{topic.title()} has many practical applications in industry.",
                    "Companies use {topic} for various business solutions."
                ]
            }
        ]
        
        topics = [
            "machine learning", "artificial intelligence", "data science",
            "deep learning", "computer vision", "natural language processing"
        ]
        
        qa_pairs = []
        for i in range(count):
            template = random.choice(qa_templates)
            topic = random.choice(topics)
            
            qa_pair = {
                "id": f"qa_{i:04d}",
                "question": template["question"].format(topic=topic),
                "ground_truth": template["answer"].format(topic=topic),
                "contexts": [ctx.format(topic=topic) for ctx in template["contexts"]],
                "metadata": {
                    "topic": topic,
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "question_type": random.choice(["factual", "explanatory", "comparative"])
                }
            }
            qa_pairs.append(qa_pair)
        
        return qa_pairs
    
    @staticmethod
    def generate_embeddings(count: int, dimension: int = 384) -> List[List[float]]:
        """Generate mock embeddings for testing."""
        return [
            np.random.normal(0, 1, dimension).tolist()
            for _ in range(count)
        ]
    
    @staticmethod
    def generate_conversation_history(length: int = 10) -> List[Dict[str, str]]:
        """Generate conversation history for testing."""
        questions = [
            "What is machine learning?",
            "How does deep learning work?", 
            "Can you explain neural networks?",
            "What are the applications of AI?",
            "How is data preprocessing done?",
            "What is the difference between supervised and unsupervised learning?",
            "How do you evaluate model performance?",
            "What are some common ML algorithms?",
            "How does natural language processing work?",
            "What is computer vision?"
        ]
        
        answers = [
            "Machine learning is a subset of AI that enables computers to learn from data.",
            "Deep learning uses neural networks with multiple layers to process data.",
            "Neural networks are computational models inspired by biological neural networks.",
            "AI applications include automation, prediction, classification, and optimization.",
            "Data preprocessing involves cleaning, transforming, and preparing data for analysis.",
            "Supervised learning uses labeled data, while unsupervised learning finds patterns.",
            "Model performance is evaluated using metrics like accuracy, precision, and recall.",
            "Common algorithms include decision trees, SVM, random forest, and neural networks.",
            "NLP enables computers to understand, interpret, and generate human language.",
            "Computer vision allows machines to interpret and understand visual information."
        ]
        
        history = []
        for i in range(min(length, len(questions))):
            history.extend([
                {"role": "user", "content": questions[i]},
                {"role": "assistant", "content": answers[i]}
            ])
        
        return history
    
    @staticmethod
    def generate_performance_test_queries(count: int = 100) -> List[str]:
        """Generate queries for performance testing."""
        query_templates = [
            "What is {concept}?",
            "How does {concept} work?",
            "Explain {concept} in simple terms",
            "What are the benefits of {concept}?",
            "Compare {concept1} and {concept2}",
            "What are the challenges in {concept}?",
            "How to implement {concept}?",
            "What are the best practices for {concept}?"
        ]
        
        concepts = [
            "machine learning", "deep learning", "neural networks", "data science",
            "artificial intelligence", "computer vision", "natural language processing",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "data mining", "big data", "cloud computing", "cybersecurity"
        ]
        
        queries = []
        for i in range(count):
            template = random.choice(query_templates)
            if "{concept1}" in template and "{concept2}" in template:
                concept1, concept2 = random.sample(concepts, 2)
                query = template.format(concept1=concept1, concept2=concept2)
            else:
                concept = random.choice(concepts)
                query = template.format(concept=concept)
            
            queries.append(query)
        
        return queries


class TestFileManager:
    """Manage test files and temporary resources."""
    
    def __init__(self):
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, content: str, suffix: str = '.txt') -> str:
        """Create temporary file with content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False)
        temp_file.write(content)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_temp_json_file(self, data: Dict[str, Any]) -> str:
        """Create temporary JSON file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(data, temp_file, indent=2)
        temp_file.close()
        
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_temp_csv_file(self, data: List[Dict[str, Any]]) -> str:
        """Create temporary CSV file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='')
        
        if data:
            fieldnames = data[0].keys()
            writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        temp_file.close()
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def create_temp_directory(self) -> str:
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup(self):
        """Clean up all temporary files and directories."""
        import os
        import shutil
        
        for file_path in self.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error cleaning up file {file_path}: {e}")
        
        for dir_path in self.temp_dirs:
            try:
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error cleaning up directory {dir_path}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def test_file_manager():
    """Provide test file manager with automatic cleanup."""
    manager = TestFileManager()
    yield manager
    manager.cleanup()


@pytest.fixture
def sample_documents(test_data_generator):
    """Provide sample documents for testing."""
    return test_data_generator.generate_sample_documents(20)


@pytest.fixture
def sample_qa_pairs(test_data_generator):
    """Provide sample QA pairs for evaluation."""
    return test_data_generator.generate_qa_pairs(10)


@pytest.fixture
def sample_embeddings(test_data_generator):
    """Provide sample embeddings."""
    return test_data_generator.generate_embeddings(20, 384)


@pytest.fixture
def sample_conversation_history(test_data_generator):
    """Provide sample conversation history."""
    return test_data_generator.generate_conversation_history(5)


@pytest.fixture
def performance_test_queries(test_data_generator):
    """Provide queries for performance testing."""
    return test_data_generator.generate_performance_test_queries(50)


@pytest.fixture
def evaluation_dataset():
    """Comprehensive evaluation dataset for RAG testing."""
    return [
        {
            "question": "What is machine learning and how does it work?",
            "ground_truth": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.",
            "contexts": [
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention."
            ],
            "question_type": "explanatory",
            "difficulty": "medium"
        },
        {
            "question": "What are neural networks?",
            "ground_truth": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections.",
            "contexts": [
                "Neural networks are a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
                "A neural network contains layers of interconnected nodes. Each node is a perceptron and is similar to a multiple linear regression."
            ],
            "question_type": "factual",
            "difficulty": "easy"
        },
        {
            "question": "Compare supervised and unsupervised learning approaches in machine learning.",
            "ground_truth": "Supervised learning uses labeled training data to learn a mapping from inputs to outputs, while unsupervised learning finds hidden patterns in data without labeled examples.",
            "contexts": [
                "Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs, referred to as 'training data'.",
                "Unsupervised learning algorithms take a set of data that contains only inputs, and finds structure in the data, like grouping or clustering of data points."
            ],
            "question_type": "comparative", 
            "difficulty": "hard"
        },
        {
            "question": "What is deep learning?",
            "ground_truth": "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data.",
            "contexts": [
                "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
                "Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing."
            ],
            "question_type": "explanatory",
            "difficulty": "medium"
        },
        {
            "question": "How does natural language processing work?",
            "ground_truth": "Natural language processing (NLP) works by using computational techniques to analyze, understand, and generate human language, combining linguistics, computer science, and artificial intelligence.",
            "contexts": [
                "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
                "NLP combines computational linguistics with statistical, machine learning, and deep learning models to enable computers to process human language in the form of text or voice data."
            ],
            "question_type": "explanatory",
            "difficulty": "medium"
        }
    ]


@pytest.fixture
def mock_embedding_responses():
    """Mock embedding service responses for testing."""
    return {
        "single_text": [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.6, 0.7, 0.8, 0.9],  # 384 dimensions
        "batch_texts": [
            [0.1, 0.2, 0.3, 0.4, 0.5] * 76 + [0.6, 0.7, 0.8, 0.9],
            [0.2, 0.3, 0.4, 0.5, 0.6] * 76 + [0.7, 0.8, 0.9, 1.0],
            [0.3, 0.4, 0.5, 0.6, 0.7] * 76 + [0.8, 0.9, 1.0, 0.1]
        ]
    }


@pytest.fixture
def mock_vector_search_results():
    """Mock vector store search results for testing."""
    return [
        {
            "id": "doc_001",
            "content": "Machine learning is a powerful technique for data analysis and pattern recognition.",
            "metadata": {
                "title": "Introduction to Machine Learning",
                "author": "AI Expert",
                "created_at": "2023-01-15T10:00:00Z",
                "category": "tutorial"
            },
            "score": 0.95
        },
        {
            "id": "doc_002", 
            "content": "Deep learning uses neural networks with multiple layers to solve complex problems.",
            "metadata": {
                "title": "Deep Learning Fundamentals",
                "author": "Neural Network Specialist",
                "created_at": "2023-02-20T14:30:00Z",
                "category": "advanced"
            },
            "score": 0.87
        },
        {
            "id": "doc_003",
            "content": "Natural language processing enables computers to understand human language.",
            "metadata": {
                "title": "NLP Basics",
                "author": "Language Processing Expert", 
                "created_at": "2023-03-10T09:15:00Z",
                "category": "intermediate"
            },
            "score": 0.82
        }
    ]


@pytest.fixture
def mock_llm_responses():
    """Mock LLM provider responses for testing."""
    return {
        "simple_query": {
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed for every scenario.",
            "usage": {
                "prompt_tokens": 45,
                "completion_tokens": 28,
                "total_tokens": 73
            }
        },
        "complex_query": {
            "content": "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes called neurons, organized in layers. Each connection between neurons has a weight that determines the strength of the signal. During training, these weights are adjusted to minimize prediction errors through algorithms like backpropagation.",
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 67,
                "total_tokens": 187
            }
        },
        "with_context": {
            "content": "Based on the provided context, machine learning is a method that automates analytical model building by learning from data patterns. It's particularly effective for tasks like prediction, classification, and pattern recognition in large datasets.",
            "usage": {
                "prompt_tokens": 89,
                "completion_tokens": 42,
                "total_tokens": 131
            }
        }
    }


@pytest.fixture
def performance_benchmarks():
    """Baseline performance benchmarks for regression testing."""
    return {
        "embedding_service": {
            "single_text_max_time": 0.5,  # seconds
            "batch_text_max_time": 2.0,   # seconds
            "max_memory_mb": 100
        },
        "vector_store": {
            "search_max_time": 0.3,       # seconds
            "insertion_max_time": 1.0,    # seconds
            "max_memory_mb": 50
        },
        "llm_provider": {
            "response_max_time": 5.0,     # seconds
            "streaming_max_time": 8.0,    # seconds
            "max_memory_mb": 200
        },
        "rag_pipeline": {
            "end_to_end_max_time": 10.0,  # seconds
            "max_memory_mb": 300,
            "min_success_rate": 0.95
        }
    }


@pytest.fixture
def test_configuration():
    """Test configuration settings."""
    return {
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_store": {
            "type": "chroma",
            "collection_name": "test_collection",
            "persist_directory": "./test_chroma_db"
        },
        "llm_provider": {
            "type": "openrouter",
            "model": "openai/gpt-3.5-turbo",
            "api_key": "test-api-key",
            "temperature": 0.7,
            "max_tokens": 500
        },
        "rag_settings": {
            "top_k_retrieval": 5,
            "similarity_threshold": 0.7,
            "max_context_length": 2000,
            "include_citations": True
        }
    }


class AsyncContextManager:
    """Async context manager for testing async resources."""
    
    def __init__(self, resource):
        self.resource = resource
        
    async def __aenter__(self):
        # Setup async resource
        return self.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup async resource
        if hasattr(self.resource, 'close'):
            await self.resource.close()


@pytest.fixture
async def async_test_client():
    """Async test client for API testing."""
    from httpx import AsyncClient
    
    async with AsyncClient() as client:
        yield client


# Utility functions for test fixtures

def create_test_document_files(file_manager: TestFileManager, count: int = 5) -> List[str]:
    """Create temporary test document files."""
    files = []
    
    for i in range(count):
        content = f"""
        Test Document {i}
        
        This is a test document created for testing purposes.
        It contains sample content about artificial intelligence and machine learning.
        
        Key topics covered:
        - Introduction to AI
        - Machine learning algorithms  
        - Neural networks
        - Applications in industry
        
        Document ID: test_doc_{i:03d}
        Created for testing the document processing pipeline.
        """
        
        file_path = file_manager.create_temp_file(content.strip(), '.txt')
        files.append(file_path)
    
    return files


def create_evaluation_dataset_file(file_manager: TestFileManager, 
                                 qa_pairs: List[Dict[str, Any]]) -> str:
    """Create evaluation dataset file."""
    return file_manager.create_temp_json_file({
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "dataset_size": len(qa_pairs),
        "data": qa_pairs
    })


@pytest.fixture
def test_document_files(test_file_manager):
    """Provide temporary test document files."""
    return create_test_document_files(test_file_manager, 5)


@pytest.fixture
def evaluation_dataset_file(test_file_manager, sample_qa_pairs):
    """Provide evaluation dataset file."""
    return create_evaluation_dataset_file(test_file_manager, sample_qa_pairs)