"""RAG evaluation tests using Ragas framework."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
from typing import List, Dict, Any

from tests.evaluation.ragas_evaluator import RagasEvaluator
from src.core.rag_pipeline import RAGPipeline


class TestRagasEvaluation:
    """Test cases for RAG evaluation using Ragas."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create mock RAG pipeline for evaluation."""
        pipeline = MagicMock(spec=RAGPipeline)
        pipeline.generate_response = AsyncMock()
        return pipeline

    @pytest.fixture
    def ragas_evaluator(self, mock_rag_pipeline):
        """Create RagasEvaluator instance for testing."""
        return RagasEvaluator(rag_pipeline=mock_rag_pipeline)

    @pytest.fixture
    def sample_evaluation_dataset(self):
        """Create sample evaluation dataset."""
        return [
            {
                "question": "What is machine learning?",
                "ground_truth": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
                "contexts": [
                    "Machine learning is a method of data analysis that automates analytical model building.",
                    "It is a branch of artificial intelligence based on the idea that systems can learn from data."
                ]
            },
            {
                "question": "Explain deep learning",
                "ground_truth": "Deep learning is a machine learning technique inspired by the structure and function of the brain called artificial neural networks.",
                "contexts": [
                    "Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
                    "It is particularly effective for tasks like image recognition and natural language processing."
                ]
            },
            {
                "question": "What is natural language processing?",
                "ground_truth": "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
                "contexts": [
                    "NLP combines computational linguistics with statistical and machine learning models.",
                    "It enables computers to process and analyze large amounts of natural language data."
                ]
            }
        ]

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_faithfulness_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test faithfulness evaluation of RAG responses."""
        # Mock RAG pipeline responses
        ragas_evaluator.rag_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {
                'answer': 'Machine learning is a subset of AI that enables computers to learn from data automatically.',
                'sources': [{'content': 'Machine learning is a method of data analysis that automates analytical model building.'}],
                'confidence_score': 0.9
            }),
            type('RAGResponse', (), {
                'answer': 'Deep learning uses neural networks with multiple layers for complex pattern recognition.',
                'sources': [{'content': 'Deep learning uses neural networks with multiple layers to model and understand complex patterns.'}],
                'confidence_score': 0.85
            }),
            type('RAGResponse', (), {
                'answer': 'NLP helps computers understand and process human language using AI techniques.',
                'sources': [{'content': 'NLP combines computational linguistics with statistical and machine learning models.'}],
                'confidence_score': 0.88
            })
        ]
        
        # Run faithfulness evaluation
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'faithfulness': 0.85,
                'answer_relevancy': 0.90,
                'context_precision': 0.82,
                'context_recall': 0.78
            }
            
            results = await ragas_evaluator.evaluate_faithfulness(sample_evaluation_dataset)
            
            assert 'faithfulness' in results
            assert results['faithfulness'] >= 0.8
            assert results['faithfulness'] <= 1.0

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_answer_relevancy_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test answer relevancy evaluation."""
        # Mock responses
        ragas_evaluator.rag_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {
                'answer': 'Machine learning enables computers to learn patterns from data without explicit programming.',
                'sources': [{'content': 'ML context'}],
                'confidence_score': 0.9
            }),
            type('RAGResponse', (), {
                'answer': 'Deep learning is a subset of machine learning using neural networks with multiple layers.',
                'sources': [{'content': 'DL context'}],
                'confidence_score': 0.88
            }),
            type('RAGResponse', (), {
                'answer': 'NLP is an AI field focused on enabling computers to understand human language.',
                'sources': [{'content': 'NLP context'}],
                'confidence_score': 0.87
            })
        ]
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'answer_relevancy': 0.88,
                'faithfulness': 0.85,
                'context_precision': 0.80,
                'context_recall': 0.75
            }
            
            results = await ragas_evaluator.evaluate_answer_relevancy(sample_evaluation_dataset)
            
            assert 'answer_relevancy' in results
            assert results['answer_relevancy'] >= 0.8

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_context_precision_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test context precision evaluation."""
        # Mock retrieval results with varying relevance
        ragas_evaluator.rag_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {
                'answer': 'ML answer',
                'sources': [
                    {'content': 'Highly relevant ML content', 'score': 0.95},
                    {'content': 'Somewhat relevant content', 'score': 0.7}
                ]
            }),
            type('RAGResponse', (), {
                'answer': 'DL answer',
                'sources': [
                    {'content': 'Very relevant DL content', 'score': 0.9},
                    {'content': 'Less relevant content', 'score': 0.6}
                ]
            }),
            type('RAGResponse', (), {
                'answer': 'NLP answer',
                'sources': [
                    {'content': 'Perfect NLP content', 'score': 0.98},
                    {'content': 'Good NLP context', 'score': 0.85}
                ]
            })
        ]
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'context_precision': 0.83,
                'context_recall': 0.79,
                'faithfulness': 0.86,
                'answer_relevancy': 0.89
            }
            
            results = await ragas_evaluator.evaluate_context_precision(sample_evaluation_dataset)
            
            assert 'context_precision' in results
            assert results['context_precision'] >= 0.7

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_context_recall_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test context recall evaluation."""
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'context_recall': 0.81,
                'context_precision': 0.84,
                'faithfulness': 0.87,
                'answer_relevancy': 0.90
            }
            
            results = await ragas_evaluator.evaluate_context_recall(sample_evaluation_dataset)
            
            assert 'context_recall' in results
            assert results['context_recall'] >= 0.7

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test comprehensive RAG evaluation with all metrics."""
        # Mock comprehensive responses
        ragas_evaluator.rag_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {
                'answer': 'Comprehensive ML answer with good context alignment.',
                'sources': [
                    {'content': 'High quality ML context', 'score': 0.92},
                    {'content': 'Supporting ML information', 'score': 0.87}
                ],
                'confidence_score': 0.91
            }),
            type('RAGResponse', (), {
                'answer': 'Detailed deep learning explanation based on neural networks.',
                'sources': [
                    {'content': 'Excellent DL context', 'score': 0.94},
                    {'content': 'Neural network details', 'score': 0.89}
                ],
                'confidence_score': 0.89
            }),
            type('RAGResponse', (), {
                'answer': 'NLP enables AI systems to process human language effectively.',
                'sources': [
                    {'content': 'Perfect NLP explanation', 'score': 0.96},
                    {'content': 'Language processing context', 'score': 0.88}
                ],
                'confidence_score': 0.92
            })
        ]
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'faithfulness': 0.87,
                'answer_relevancy': 0.91,
                'context_precision': 0.85,
                'context_recall': 0.82,
                'answer_similarity': 0.88,
                'answer_correctness': 0.86
            }
            
            results = await ragas_evaluator.evaluate_comprehensive(sample_evaluation_dataset)
            
            # Verify all key metrics are present
            required_metrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']
            for metric in required_metrics:
                assert metric in results
                assert results[metric] >= 0.8
            
            # Verify overall score calculation
            if 'overall_score' in results:
                assert results['overall_score'] >= 0.8

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_evaluation_with_custom_metrics(self, ragas_evaluator):
        """Test evaluation with custom metrics."""
        custom_dataset = [
            {
                "question": "What is the capital of France?",
                "ground_truth": "The capital of France is Paris.",
                "contexts": ["Paris is the capital and largest city of France."],
                "custom_metadata": {"difficulty": "easy", "category": "geography"}
            }
        ]
        
        # Mock custom metric evaluation
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'faithfulness': 0.95,
                'answer_relevancy': 0.93,
                'custom_coherence': 0.91,
                'custom_completeness': 0.89
            }
            
            results = await ragas_evaluator.evaluate_with_custom_metrics(
                custom_dataset, 
                custom_metrics=['custom_coherence', 'custom_completeness']
            )
            
            assert 'custom_coherence' in results
            assert 'custom_completeness' in results

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_batch_evaluation_performance(self, ragas_evaluator):
        """Test performance of batch evaluation."""
        # Create larger dataset for performance testing
        large_dataset = []
        for i in range(50):  # 50 samples
            large_dataset.append({
                "question": f"Question {i} about AI topic {i % 5}",
                "ground_truth": f"Ground truth answer for question {i}",
                "contexts": [f"Context {i} with relevant information", f"Additional context {i}"]
            })
        
        # Mock batch responses
        mock_responses = []
        for i in range(50):
            mock_responses.append(
                type('RAGResponse', (), {
                    'answer': f'Generated answer for question {i}',
                    'sources': [{'content': f'Source {i}', 'score': 0.8 + (i % 10) * 0.02}],
                    'confidence_score': 0.85
                })
            )
        
        ragas_evaluator.rag_pipeline.generate_response.side_effect = mock_responses
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'faithfulness': 0.84,
                'answer_relevancy': 0.87,
                'context_precision': 0.81,
                'context_recall': 0.79
            }
            
            import time
            start_time = time.time()
            
            results = await ragas_evaluator.evaluate_batch(large_dataset, batch_size=10)
            
            evaluation_time = time.time() - start_time
            
            # Verify results
            assert all(metric in results for metric in ['faithfulness', 'answer_relevancy'])
            assert evaluation_time < 30  # Should complete within reasonable time (mocked)

    @pytest.mark.evaluation
    def test_evaluation_report_generation(self, ragas_evaluator):
        """Test evaluation report generation."""
        evaluation_results = {
            'faithfulness': 0.85,
            'answer_relevancy': 0.89,
            'context_precision': 0.82,
            'context_recall': 0.78,
            'overall_score': 0.84
        }
        
        report = ragas_evaluator.generate_evaluation_report(
            evaluation_results,
            dataset_size=100,
            evaluation_time=45.5
        )
        
        # Verify report structure
        assert 'summary' in report
        assert 'metrics' in report
        assert 'recommendations' in report
        
        # Verify summary information
        assert report['summary']['dataset_size'] == 100
        assert report['summary']['evaluation_time'] == 45.5
        assert report['summary']['overall_score'] == 0.84
        
        # Verify recommendations are provided
        assert len(report['recommendations']) > 0

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_comparative_evaluation(self, ragas_evaluator, sample_evaluation_dataset):
        """Test comparative evaluation between different RAG configurations."""
        # Create second RAG pipeline for comparison
        second_pipeline = MagicMock(spec=RAGPipeline)
        second_pipeline.generate_response = AsyncMock()
        
        # Mock responses for both pipelines
        ragas_evaluator.rag_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {'answer': 'Pipeline 1 answer', 'sources': [], 'confidence_score': 0.8}),
            type('RAGResponse', (), {'answer': 'Pipeline 1 answer', 'sources': [], 'confidence_score': 0.8}),
            type('RAGResponse', (), {'answer': 'Pipeline 1 answer', 'sources': [], 'confidence_score': 0.8})
        ]
        
        second_pipeline.generate_response.side_effect = [
            type('RAGResponse', (), {'answer': 'Pipeline 2 answer', 'sources': [], 'confidence_score': 0.9}),
            type('RAGResponse', (), {'answer': 'Pipeline 2 answer', 'sources': [], 'confidence_score': 0.9}),
            type('RAGResponse', (), {'answer': 'Pipeline 2 answer', 'sources': [], 'confidence_score': 0.9})
        ]
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.side_effect = [
                # Results for pipeline 1
                {
                    'faithfulness': 0.82,
                    'answer_relevancy': 0.85,
                    'context_precision': 0.79,
                    'context_recall': 0.75
                },
                # Results for pipeline 2
                {
                    'faithfulness': 0.88,
                    'answer_relevancy': 0.91,
                    'context_precision': 0.84,
                    'context_recall': 0.81
                }
            ]
            
            comparison = await ragas_evaluator.compare_pipelines(
                [ragas_evaluator.rag_pipeline, second_pipeline],
                sample_evaluation_dataset,
                pipeline_names=['Pipeline 1', 'Pipeline 2']
            )
            
            # Verify comparison results
            assert 'Pipeline 1' in comparison
            assert 'Pipeline 2' in comparison
            assert comparison['Pipeline 2']['faithfulness'] > comparison['Pipeline 1']['faithfulness']

    @pytest.mark.evaluation
    @pytest.mark.asyncio
    async def test_evaluation_with_different_question_types(self, ragas_evaluator):
        """Test evaluation across different question types."""
        diverse_dataset = [
            {
                "question": "What is machine learning?",  # Factual
                "question_type": "factual",
                "ground_truth": "ML definition",
                "contexts": ["ML context"]
            },
            {
                "question": "How do neural networks work?",  # Explanatory
                "question_type": "explanatory", 
                "ground_truth": "Neural network explanation",
                "contexts": ["NN context"]
            },
            {
                "question": "Compare supervised vs unsupervised learning",  # Comparative
                "question_type": "comparative",
                "ground_truth": "Comparison of learning types",
                "contexts": ["Learning types context"]
            }
        ]
        
        with patch('ragas.evaluate') as mock_ragas_evaluate:
            mock_ragas_evaluate.return_value = {
                'faithfulness': 0.86,
                'answer_relevancy': 0.89,
                'context_precision': 0.83,
                'context_recall': 0.80
            }
            
            results = await ragas_evaluator.evaluate_by_question_type(diverse_dataset)
            
            # Verify results are broken down by question type
            assert 'factual' in results
            assert 'explanatory' in results
            assert 'comparative' in results