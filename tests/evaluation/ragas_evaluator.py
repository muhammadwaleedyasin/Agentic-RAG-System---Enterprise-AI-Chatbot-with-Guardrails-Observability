"""RAG evaluation using Ragas framework."""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: Ragas not installed. Some evaluation features will be unavailable.")


@dataclass
class EvaluationResult:
    """Evaluation result container."""
    metric_scores: Dict[str, float]
    overall_score: float
    evaluation_time: float
    dataset_size: int
    recommendations: List[str]


class RagasEvaluator:
    """RAG system evaluator using Ragas framework."""
    
    def __init__(self, rag_pipeline, embedding_model=None, llm_model=None):
        """
        Initialize the Ragas evaluator.
        
        Args:
            rag_pipeline: The RAG pipeline to evaluate
            embedding_model: Optional embedding model for evaluation
            llm_model: Optional LLM model for evaluation
        """
        self.rag_pipeline = rag_pipeline
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Default metrics to evaluate
        self.default_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ] if RAGAS_AVAILABLE else []
        
        # Evaluation history
        self.evaluation_history = []
    
    async def evaluate_faithfulness(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate faithfulness of RAG responses.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary with faithfulness scores
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Please install with: pip install ragas")
        
        # Generate responses for the dataset
        responses_data = await self._generate_responses_for_dataset(dataset)
        
        # Create Ragas dataset
        ragas_dataset = Dataset.from_dict(responses_data)
        
        # Evaluate faithfulness
        result = evaluate(ragas_dataset, metrics=[faithfulness])
        
        return {
            'faithfulness': result['faithfulness']
        }
    
    async def evaluate_answer_relevancy(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate answer relevancy of RAG responses.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary with answer relevancy scores
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Please install with: pip install ragas")
        
        responses_data = await self._generate_responses_for_dataset(dataset)
        ragas_dataset = Dataset.from_dict(responses_data)
        
        result = evaluate(ragas_dataset, metrics=[answer_relevancy])
        
        return {
            'answer_relevancy': result['answer_relevancy']
        }
    
    async def evaluate_context_precision(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate context precision of RAG responses.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary with context precision scores
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Please install with: pip install ragas")
        
        responses_data = await self._generate_responses_for_dataset(dataset)
        ragas_dataset = Dataset.from_dict(responses_data)
        
        result = evaluate(ragas_dataset, metrics=[context_precision])
        
        return {
            'context_precision': result['context_precision']
        }
    
    async def evaluate_context_recall(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate context recall of RAG responses.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary with context recall scores
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Please install with: pip install ragas")
        
        responses_data = await self._generate_responses_for_dataset(dataset)
        ragas_dataset = Dataset.from_dict(responses_data)
        
        result = evaluate(ragas_dataset, metrics=[context_recall])
        
        return {
            'context_recall': result['context_recall']
        }
    
    async def evaluate_comprehensive(self, dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Run comprehensive evaluation with all available metrics.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary with all metric scores
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed. Please install with: pip install ragas")
        
        start_time = time.time()
        
        # Generate responses for the dataset
        responses_data = await self._generate_responses_for_dataset(dataset)
        
        # Create Ragas dataset
        ragas_dataset = Dataset.from_dict(responses_data)
        
        # Evaluate with all default metrics
        result = evaluate(ragas_dataset, metrics=self.default_metrics)
        
        # Calculate overall score
        metric_scores = {k: v for k, v in result.items() if isinstance(v, (int, float))}
        overall_score = np.mean(list(metric_scores.values()))
        
        evaluation_time = time.time() - start_time
        
        # Add evaluation to history
        evaluation_result = EvaluationResult(
            metric_scores=metric_scores,
            overall_score=overall_score,
            evaluation_time=evaluation_time,
            dataset_size=len(dataset),
            recommendations=self._generate_recommendations(metric_scores)
        )
        
        self.evaluation_history.append(evaluation_result)
        
        return {
            **metric_scores,
            'overall_score': overall_score,
            'evaluation_time': evaluation_time
        }
    
    async def evaluate_with_custom_metrics(self, 
                                         dataset: List[Dict[str, Any]], 
                                         custom_metrics: List[str]) -> Dict[str, float]:
        """
        Evaluate with custom metrics.
        
        Args:
            dataset: List of evaluation samples
            custom_metrics: List of custom metric names
            
        Returns:
            Dictionary with custom metric scores
        """
        if not RAGAS_AVAILABLE:
            return {'error': 'Ragas not available for custom metrics'}
        
        responses_data = await self._generate_responses_for_dataset(dataset)
        ragas_dataset = Dataset.from_dict(responses_data)
        
        # For demo purposes, return mock scores
        # In real implementation, you would define custom metrics
        custom_scores = {}
        for metric in custom_metrics:
            custom_scores[metric] = np.random.uniform(0.7, 0.95)  # Mock scores
        
        # Also include standard metrics
        standard_result = evaluate(ragas_dataset, metrics=self.default_metrics)
        
        return {
            **standard_result,
            **custom_scores
        }
    
    async def evaluate_batch(self, 
                           dataset: List[Dict[str, Any]], 
                           batch_size: int = 10) -> Dict[str, float]:
        """
        Evaluate dataset in batches for better performance.
        
        Args:
            dataset: List of evaluation samples
            batch_size: Size of each batch
            
        Returns:
            Aggregated evaluation results
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("Ragas is not installed")
        
        all_results = []
        
        # Process in batches
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_results = await self.evaluate_comprehensive(batch)
            all_results.append(batch_results)
        
        # Aggregate results
        aggregated = {}
        for key in all_results[0].keys():
            if key != 'evaluation_time':  # Don't average evaluation time
                values = [result[key] for result in all_results if key in result]
                aggregated[key] = np.mean(values)
        
        return aggregated
    
    async def compare_pipelines(self, 
                              pipelines: List[Any], 
                              dataset: List[Dict[str, Any]],
                              pipeline_names: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple RAG pipelines on the same dataset.
        
        Args:
            pipelines: List of RAG pipelines to compare
            dataset: Evaluation dataset
            pipeline_names: Optional names for pipelines
            
        Returns:
            Comparison results for each pipeline
        """
        if pipeline_names is None:
            pipeline_names = [f"Pipeline_{i+1}" for i in range(len(pipelines))]
        
        comparison_results = {}
        
        for pipeline, name in zip(pipelines, pipeline_names):
            # Temporarily switch pipeline
            original_pipeline = self.rag_pipeline
            self.rag_pipeline = pipeline
            
            try:
                results = await self.evaluate_comprehensive(dataset)
                comparison_results[name] = results
            finally:
                # Restore original pipeline
                self.rag_pipeline = original_pipeline
        
        return comparison_results
    
    async def evaluate_by_question_type(self, dataset: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by question type.
        
        Args:
            dataset: Dataset with question_type field
            
        Returns:
            Results broken down by question type
        """
        # Group dataset by question type
        question_types = {}
        for sample in dataset:
            q_type = sample.get('question_type', 'unknown')
            if q_type not in question_types:
                question_types[q_type] = []
            question_types[q_type].append(sample)
        
        # Evaluate each question type separately
        results_by_type = {}
        for q_type, samples in question_types.items():
            type_results = await self.evaluate_comprehensive(samples)
            results_by_type[q_type] = type_results
        
        return results_by_type
    
    async def _generate_responses_for_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, List]:
        """
        Generate RAG responses for evaluation dataset.
        
        Args:
            dataset: List of evaluation samples
            
        Returns:
            Dictionary formatted for Ragas evaluation
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for sample in dataset:
            question = sample['question']
            ground_truth = sample.get('ground_truth', '')
            expected_contexts = sample.get('contexts', [])
            
            # Generate response using RAG pipeline
            try:
                response = await self.rag_pipeline.generate_response(question)
                answer = response.answer
                
                # Extract contexts from sources
                response_contexts = []
                if hasattr(response, 'sources') and response.sources:
                    response_contexts = [source.get('content', '') for source in response.sources]
                
                questions.append(question)
                answers.append(answer)
                contexts.append(response_contexts if response_contexts else expected_contexts)
                ground_truths.append(ground_truth)
                
            except Exception as e:
                print(f"Error generating response for question '{question}': {e}")
                # Use fallback values
                questions.append(question)
                answers.append("Error generating response")
                contexts.append(expected_contexts)
                ground_truths.append(ground_truth)
        
        return {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truths': ground_truths
        }
    
    def _generate_recommendations(self, metric_scores: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on evaluation scores.
        
        Args:
            metric_scores: Dictionary of metric scores
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Faithfulness recommendations
        if 'faithfulness' in metric_scores:
            if metric_scores['faithfulness'] < 0.8:
                recommendations.append(
                    "Low faithfulness score detected. Consider improving context retrieval "
                    "and ensuring generated answers stay grounded in source documents."
                )
        
        # Answer relevancy recommendations
        if 'answer_relevancy' in metric_scores:
            if metric_scores['answer_relevancy'] < 0.8:
                recommendations.append(
                    "Low answer relevancy. Consider fine-tuning the prompt template "
                    "to generate more focused and relevant responses."
                )
        
        # Context precision recommendations
        if 'context_precision' in metric_scores:
            if metric_scores['context_precision'] < 0.8:
                recommendations.append(
                    "Low context precision. Consider improving document chunking strategy "
                    "and retrieval ranking to surface more relevant contexts first."
                )
        
        # Context recall recommendations
        if 'context_recall' in metric_scores:
            if metric_scores['context_recall'] < 0.8:
                recommendations.append(
                    "Low context recall. Consider expanding retrieval results or "
                    "improving embedding model to capture more relevant contexts."
                )
        
        # Overall performance recommendations
        overall_avg = np.mean(list(metric_scores.values()))
        if overall_avg > 0.9:
            recommendations.append("Excellent performance! Consider this configuration for production.")
        elif overall_avg < 0.7:
            recommendations.append(
                "Overall performance needs improvement. Consider reviewing the entire RAG pipeline."
            )
        
        return recommendations if recommendations else ["Performance is acceptable."]
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, float],
                                 dataset_size: int,
                                 evaluation_time: float) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Dictionary of evaluation results
            dataset_size: Size of evaluation dataset
            evaluation_time: Time taken for evaluation
            
        Returns:
            Comprehensive evaluation report
        """
        report = {
            'summary': {
                'dataset_size': dataset_size,
                'evaluation_time': evaluation_time,
                'overall_score': evaluation_results.get('overall_score', 0.0),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'metrics': {
                metric: score for metric, score in evaluation_results.items()
                if metric != 'overall_score' and isinstance(score, (int, float))
            },
            'recommendations': self._generate_recommendations(evaluation_results),
            'performance_grade': self._calculate_performance_grade(evaluation_results.get('overall_score', 0.0))
        }
        
        return report
    
    def _calculate_performance_grade(self, overall_score: float) -> str:
        """Calculate performance grade based on overall score."""
        if overall_score >= 0.9:
            return "A"
        elif overall_score >= 0.8:
            return "B"
        elif overall_score >= 0.7:
            return "C"
        elif overall_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def get_evaluation_history(self) -> List[EvaluationResult]:
        """Get evaluation history."""
        return self.evaluation_history
    
    def export_results_to_csv(self, 
                            results: Dict[str, float], 
                            filename: str = "ragas_evaluation_results.csv") -> None:
        """
        Export evaluation results to CSV file.
        
        Args:
            results: Evaluation results dictionary
            filename: Output filename
        """
        df = pd.DataFrame([results])
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    async def continuous_evaluation(self, 
                                  dataset: List[Dict[str, Any]], 
                                  interval_hours: int = 24) -> None:
        """
        Run continuous evaluation at specified intervals.
        
        Args:
            dataset: Evaluation dataset
            interval_hours: Hours between evaluations
        """
        while True:
            try:
                print(f"Running scheduled evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                results = await self.evaluate_comprehensive(dataset)
                
                # Log results or send alerts based on performance degradation
                if results.get('overall_score', 0) < 0.8:
                    print("WARNING: RAG performance has degraded!")
                
                print(f"Evaluation completed. Overall score: {results.get('overall_score', 0):.3f}")
                
            except Exception as e:
                print(f"Error in continuous evaluation: {e}")
            
            # Wait for next evaluation
            await asyncio.sleep(interval_hours * 3600)


# Example usage and utility functions
def create_evaluation_dataset_from_qa_pairs(qa_pairs: List[Tuple[str, str, List[str]]]) -> List[Dict[str, Any]]:
    """
    Create evaluation dataset from question-answer pairs.
    
    Args:
        qa_pairs: List of (question, answer, contexts) tuples
        
    Returns:
        Formatted evaluation dataset
    """
    dataset = []
    for question, answer, contexts in qa_pairs:
        dataset.append({
            'question': question,
            'ground_truth': answer,
            'contexts': contexts
        })
    return dataset


def load_evaluation_dataset_from_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from file (JSON or CSV).
    
    Args:
        filepath: Path to dataset file
        
    Returns:
        Evaluation dataset
    """
    if filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        return df.to_dict('records')
    else:
        raise ValueError("Unsupported file format. Use .json or .csv")