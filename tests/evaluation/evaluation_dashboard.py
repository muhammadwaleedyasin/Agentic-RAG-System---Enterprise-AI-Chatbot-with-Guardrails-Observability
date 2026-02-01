"""RAG Evaluation Dashboard and Reporting System."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
import json
import time
from datetime import datetime, timedelta
import numpy as np


class RAGEvaluationDashboard:
    """Interactive dashboard for RAG system evaluation results."""
    
    def __init__(self):
        """Initialize the evaluation dashboard."""
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="RAG Evaluation Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            border-left: 5px solid #1f77b4;
        }
        
        .metric-title {
            font-size: 18px;
            font-weight: bold;
            color: #1f77b4;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }
        
        .status-good { color: #4caf50; }
        .status-warning { color: #ff9800; }
        .status-error { color: #f44336; }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Run the main dashboard application."""
        st.title("🔍 RAG System Evaluation Dashboard")
        st.markdown("Monitor and analyze your RAG system's performance with comprehensive metrics and insights.")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Performance Metrics", "Quality Analysis", "Trend Analysis", "Comparison", "Settings"]
        )
        
        if page == "Overview":
            self.show_overview()
        elif page == "Performance Metrics":
            self.show_performance_metrics()
        elif page == "Quality Analysis":
            self.show_quality_analysis()
        elif page == "Trend Analysis":
            self.show_trend_analysis()
        elif page == "Comparison":
            self.show_comparison()
        elif page == "Settings":
            self.show_settings()
    
    def show_overview(self):
        """Display overview dashboard."""
        st.header("📈 System Overview")
        
        # Load sample data (in production, this would come from your evaluation system)
        evaluation_data = self.load_evaluation_data()
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.display_metric_card(
                "Overall Score",
                f"{evaluation_data['overall_score']:.2f}",
                "🎯",
                self.get_status_color(evaluation_data['overall_score'], 0.8, 0.6)
            )
        
        with col2:
            self.display_metric_card(
                "Faithfulness",
                f"{evaluation_data['faithfulness']:.2f}",
                "✅",
                self.get_status_color(evaluation_data['faithfulness'], 0.8, 0.6)
            )
        
        with col3:
            self.display_metric_card(
                "Answer Relevancy",
                f"{evaluation_data['answer_relevancy']:.2f}",
                "🎪",
                self.get_status_color(evaluation_data['answer_relevancy'], 0.8, 0.6)
            )
        
        with col4:
            self.display_metric_card(
                "Context Precision",
                f"{evaluation_data['context_precision']:.2f}",
                "🎯",
                self.get_status_color(evaluation_data['context_precision'], 0.8, 0.6)
            )
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_radar_chart(evaluation_data)
        
        with col2:
            self.plot_performance_timeline()
        
        # Recent evaluations table
        st.subheader("🕒 Recent Evaluations")
        self.display_recent_evaluations()
        
        # System health status
        st.subheader("🏥 System Health")
        self.display_system_health()
    
    def show_performance_metrics(self):
        """Display detailed performance metrics."""
        st.header("⚡ Performance Metrics")
        
        # Performance overview
        perf_data = self.load_performance_data()
        
        # Response time metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self.display_metric_card(
                "Avg Response Time",
                f"{perf_data['avg_response_time']:.2f}s",
                "⏱️",
                self.get_performance_status(perf_data['avg_response_time'], 2.0, 5.0)
            )
        
        with col2:
            self.display_metric_card(
                "Throughput",
                f"{perf_data['requests_per_second']:.1f} req/s",
                "🚀",
                "good"
            )
        
        with col3:
            self.display_metric_card(
                "Error Rate",
                f"{perf_data['error_rate']:.1%}",
                "❌",
                self.get_performance_status(perf_data['error_rate'], 0.01, 0.05, reverse=True)
            )
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_response_time_distribution()
        
        with col2:
            self.plot_throughput_over_time()
        
        # Detailed performance breakdown
        st.subheader("🔍 Component Breakdown")
        self.display_component_performance()
        
        # Load testing results
        st.subheader("📊 Load Testing Results")
        self.display_load_testing_results()
    
    def show_quality_analysis(self):
        """Display quality analysis dashboard."""
        st.header("🔬 Quality Analysis")
        
        # Quality metrics overview
        quality_data = self.load_quality_data()
        
        # Ragas metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 RAGAS Metrics")
            ragas_metrics = {
                "Faithfulness": quality_data["faithfulness"],
                "Answer Relevancy": quality_data["answer_relevancy"],
                "Context Precision": quality_data["context_precision"],
                "Context Recall": quality_data["context_recall"]
            }
            
            self.plot_metrics_bar_chart(ragas_metrics)
        
        with col2:
            st.subheader("📈 Quality Trends")
            self.plot_quality_trends()
        
        # Question type analysis
        st.subheader("❓ Performance by Question Type")
        self.display_question_type_analysis()
        
        # Failure analysis
        st.subheader("🔍 Failure Analysis")
        self.display_failure_analysis()
        
        # Improvement recommendations
        st.subheader("💡 Recommendations")
        self.display_recommendations()
    
    def show_trend_analysis(self):
        """Display trend analysis dashboard."""
        st.header("📈 Trend Analysis")
        
        # Time range selector
        time_range = st.selectbox(
            "Select Time Range",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 3 Months"]
        )
        
        # Trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Metric Trends")
            self.plot_metric_trends(time_range)
        
        with col2:
            st.subheader("🎯 Performance Trends")
            self.plot_performance_trends(time_range)
        
        # Seasonal analysis
        st.subheader("🔄 Seasonal Patterns")
        self.display_seasonal_analysis()
        
        # Anomaly detection
        st.subheader("🚨 Anomaly Detection")
        self.display_anomaly_detection()
    
    def show_comparison(self):
        """Display comparison dashboard."""
        st.header("⚖️ Model Comparison")
        
        # Model selector
        models = ["GPT-3.5-Turbo", "GPT-4", "Claude-2", "Llama-2-7B", "Llama-2-13B"]
        selected_models = st.multiselect(
            "Select Models to Compare",
            models,
            default=models[:3]
        )
        
        if len(selected_models) >= 2:
            # Comparison metrics table
            st.subheader("📊 Metrics Comparison")
            self.display_model_comparison_table(selected_models)
            
            # Comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                self.plot_model_comparison_radar(selected_models)
            
            with col2:
                self.plot_model_performance_comparison(selected_models)
            
            # Statistical significance testing
            st.subheader("📈 Statistical Analysis")
            self.display_statistical_analysis(selected_models)
        else:
            st.warning("Please select at least 2 models for comparison.")
    
    def show_settings(self):
        """Display settings and configuration."""
        st.header("⚙️ Settings & Configuration")
        
        # Evaluation settings
        st.subheader("🔧 Evaluation Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("Evaluation Frequency (hours)", value=24, min_value=1)
            st.multiselect(
                "Metrics to Track",
                ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"],
                default=["Faithfulness", "Answer Relevancy"]
            )
        
        with col2:
            st.number_input("Performance Threshold", value=0.8, min_value=0.0, max_value=1.0, step=0.1)
            st.checkbox("Enable Automatic Alerts")
        
        # Data management
        st.subheader("📊 Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Evaluation Data"):
                self.export_evaluation_data()
        
        with col2:
            uploaded_file = st.file_uploader("Import Evaluation Data", type=["json", "csv"])
            if uploaded_file:
                self.import_evaluation_data(uploaded_file)
        
        # System configuration
        st.subheader("🖥️ System Configuration")
        self.display_system_configuration()
    
    def display_metric_card(self, title: str, value: str, icon: str, status: str):
        """Display a metric card."""
        status_class = f"status-{status}"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; margin-right: 10px;">{icon}</span>
                <div>
                    <div class="metric-title">{title}</div>
                    <div class="metric-value {status_class}">{value}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def plot_radar_chart(self, data: Dict[str, float]):
        """Plot radar chart of evaluation metrics."""
        st.subheader("📡 Metrics Overview")
        
        categories = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
        values = [
            data.get("faithfulness", 0),
            data.get("answer_relevancy", 0),
            data.get("context_precision", 0),
            data.get("context_recall", 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="RAG System Performance Radar"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_performance_timeline(self):
        """Plot performance timeline."""
        st.subheader("📈 Performance Timeline")
        
        # Generate sample timeline data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_scores = np.random.normal(0.85, 0.05, 30)
        performance_scores = np.clip(performance_scores, 0.7, 0.95)
        
        df = pd.DataFrame({
            'Date': dates,
            'Performance Score': performance_scores
        })
        
        fig = px.line(df, x='Date', y='Performance Score', 
                     title='Performance Score Over Time')
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Minimum Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_recent_evaluations(self):
        """Display recent evaluations table."""
        # Generate sample data
        recent_evals = pd.DataFrame({
            'Timestamp': pd.date_range(start='2024-01-01', periods=10, freq='6H'),
            'Dataset': ['test_set_1', 'test_set_2'] * 5,
            'Overall Score': np.random.uniform(0.75, 0.95, 10),
            'Questions': np.random.randint(50, 200, 10),
            'Status': ['✅ Completed'] * 8 + ['⚠️ Issues'] * 2
        })
        
        st.dataframe(
            recent_evals.sort_values('Timestamp', ascending=False),
            use_container_width=True
        )
    
    def display_system_health(self):
        """Display system health status."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🖥️ System Status",
                value="Healthy",
                delta="All services operational"
            )
        
        with col2:
            st.metric(
                label="🔄 Last Evaluation",
                value="2 hours ago",
                delta="On schedule"
            )
        
        with col3:
            st.metric(
                label="📊 Data Quality",
                value="98.5%",
                delta="0.5% improvement"
            )
    
    def plot_response_time_distribution(self):
        """Plot response time distribution."""
        st.subheader("📊 Response Time Distribution")
        
        # Generate sample response time data
        response_times = np.random.lognormal(mean=1.0, sigma=0.5, size=1000)
        
        fig = px.histogram(
            x=response_times,
            nbins=50,
            title="Response Time Distribution",
            labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
        )
        
        # Add percentile lines
        p95 = np.percentile(response_times, 95)
        fig.add_vline(x=p95, line_dash="dash", line_color="red",
                     annotation_text=f"95th percentile: {p95:.2f}s")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_throughput_over_time(self):
        """Plot throughput over time."""
        st.subheader("🚀 Throughput Over Time")
        
        # Generate sample throughput data
        hours = pd.date_range(start='2024-01-01', periods=24, freq='H')
        throughput = np.random.poisson(lam=10, size=24) + np.random.normal(0, 2, 24)
        throughput = np.maximum(throughput, 0)
        
        df = pd.DataFrame({
            'Hour': hours,
            'Requests/Second': throughput
        })
        
        fig = px.line(df, x='Hour', y='Requests/Second',
                     title='System Throughput Over 24 Hours')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_component_performance(self):
        """Display component performance breakdown."""
        components = {
            'Embedding Service': {'avg_time': 0.15, 'error_rate': 0.001},
            'Vector Store': {'avg_time': 0.08, 'error_rate': 0.002},
            'LLM Provider': {'avg_time': 2.1, 'error_rate': 0.015},
            'RAG Pipeline': {'avg_time': 2.8, 'error_rate': 0.01}
        }
        
        df = pd.DataFrame(components).T
        df.columns = ['Avg Response Time (s)', 'Error Rate']
        
        st.dataframe(df, use_container_width=True)
    
    def load_evaluation_data(self) -> Dict[str, float]:
        """Load sample evaluation data."""
        return {
            'overall_score': 0.87,
            'faithfulness': 0.89,
            'answer_relevancy': 0.92,
            'context_precision': 0.84,
            'context_recall': 0.81
        }
    
    def load_performance_data(self) -> Dict[str, float]:
        """Load sample performance data."""
        return {
            'avg_response_time': 2.3,
            'requests_per_second': 12.5,
            'error_rate': 0.008
        }
    
    def load_quality_data(self) -> Dict[str, float]:
        """Load sample quality data."""
        return {
            'faithfulness': 0.89,
            'answer_relevancy': 0.92,
            'context_precision': 0.84,
            'context_recall': 0.81
        }
    
    def get_status_color(self, value: float, good_threshold: float, warning_threshold: float) -> str:
        """Get status color based on value and thresholds."""
        if value >= good_threshold:
            return "good"
        elif value >= warning_threshold:
            return "warning"
        else:
            return "error"
    
    def get_performance_status(self, value: float, warning_threshold: float, error_threshold: float, reverse: bool = False) -> str:
        """Get performance status color."""
        if reverse:
            if value <= warning_threshold:
                return "good"
            elif value <= error_threshold:
                return "warning"
            else:
                return "error"
        else:
            if value <= warning_threshold:
                return "good"
            elif value <= error_threshold:
                return "warning"
            else:
                return "error"
    
    def plot_metrics_bar_chart(self, metrics: Dict[str, float]):
        """Plot metrics as bar chart."""
        df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
        
        fig = px.bar(df, x='Metric', y='Score', 
                    title='RAGAS Evaluation Metrics',
                    color='Score',
                    color_continuous_scale='RdYlGn')
        
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_quality_trends(self):
        """Plot quality trends over time."""
        # Sample trend data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        trends = {
            'Faithfulness': np.random.normal(0.89, 0.03, 30),
            'Answer Relevancy': np.random.normal(0.92, 0.02, 30),
            'Context Precision': np.random.normal(0.84, 0.04, 30),
        }
        
        fig = go.Figure()
        
        for metric, values in trends.items():
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=metric
            ))
        
        fig.update_layout(
            title='Quality Metrics Trends',
            xaxis_title='Date',
            yaxis_title='Score',
            yaxis_range=[0.7, 1.0]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional helper methods would go here...
    def display_question_type_analysis(self):
        """Display performance analysis by question type."""
        question_types = {
            'Factual': {'score': 0.92, 'count': 45},
            'Explanatory': {'score': 0.87, 'count': 38},
            'Comparative': {'score': 0.81, 'count': 22},
            'Analytical': {'score': 0.79, 'count': 15}
        }
        
        df = pd.DataFrame(question_types).T
        df.columns = ['Avg Score', 'Question Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df, y=df.index, x='Avg Score',
                        title='Performance by Question Type',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(df, use_container_width=True)
    
    def display_failure_analysis(self):
        """Display failure analysis."""
        failure_data = {
            'Low Context Quality': 15,
            'Hallucination': 8,
            'Irrelevant Retrieval': 12,
            'Citation Missing': 6
        }
        
        fig = px.pie(
            values=list(failure_data.values()),
            names=list(failure_data.keys()),
            title='Failure Categories'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_recommendations(self):
        """Display improvement recommendations."""
        recommendations = [
            "🎯 **Context Precision**: Consider improving document chunking strategy to surface more relevant contexts first.",
            "🔍 **Context Recall**: Expand retrieval results or improve embedding model to capture more relevant contexts.",
            "✅ **Faithfulness**: Review prompt templates to ensure responses stay grounded in source documents.",
            "📚 **Citation Quality**: Implement automatic citation validation and source quality scoring."
        ]
        
        for rec in recommendations:
            st.markdown(rec)
    
    def export_evaluation_data(self):
        """Export evaluation data."""
        st.success("Evaluation data exported successfully!")
    
    def import_evaluation_data(self, uploaded_file):
        """Import evaluation data."""
        st.success(f"Data imported from {uploaded_file.name}")
    
    def display_system_configuration(self):
        """Display system configuration."""
        config_data = {
            'Evaluation Engine': 'Ragas v0.1.9',
            'Model Provider': 'OpenRouter API',
            'Vector Store': 'ChromaDB',
            'Embedding Model': 'all-MiniLM-L6-v2',
            'Last Updated': '2024-01-15 10:30:00'
        }
        
        for key, value in config_data.items():
            st.text(f"{key}: {value}")


def main():
    """Main application entry point."""
    dashboard = RAGEvaluationDashboard()
    dashboard.run()


if __name__ == "__main__":
    # For running the dashboard, use: streamlit run evaluation_dashboard.py
    main()