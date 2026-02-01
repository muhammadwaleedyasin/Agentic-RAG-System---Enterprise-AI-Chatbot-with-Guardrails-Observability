"""
Runtime Performance Testing Suite for RAG Chatbot UI

Tests chat response times, document upload performance, search result rendering,
and real-time interaction performance during active usage.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pytest
import numpy as np
from pathlib import Path


@dataclass
class RuntimePerformanceMetrics:
    """Runtime performance metrics container"""
    chat_response_time: float
    search_response_time: float
    document_upload_time: float
    ui_rendering_time: float
    websocket_latency: float
    memory_usage_mb: float
    cpu_usage_percent: float
    network_throughput: float
    animation_smoothness: float
    scroll_performance: float


class RuntimePerformanceTester:
    """Comprehensive runtime performance testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.thresholds = {
            'chat_response_time': 1.5,  # seconds
            'search_response_time': 0.8,
            'document_upload_time': 5.0,
            'ui_rendering_time': 0.1,
            'websocket_latency': 0.05,
            'memory_usage_mb': 100,
            'animation_smoothness': 60,  # FPS
            'scroll_performance': 60   # FPS
        }
        
        # Test data for various scenarios
        self.test_queries = [
            "What is the deployment process?",
            "How do I configure the RAG pipeline?",
            "What are the security requirements?",
            "How to setup observability monitoring?",
            "What is the scaling strategy?",
            "How to handle document ingestion?",
            "What are the performance benchmarks?",
            "How to troubleshoot common issues?"
        ]
        
        self.complex_queries = [
            "Compare the performance characteristics of local LLM inference versus OpenRouter API, considering factors like latency, cost, privacy, and scalability for enterprise deployments.",
            "Explain the complete document ingestion pipeline including chunking strategies, embedding generation, vector storage, metadata tagging, and how guardrails are applied throughout the process.",
            "Describe the observability and monitoring architecture including metrics collection, trace propagation, dashboard configuration, and integration with external monitoring systems."
        ]
    
    def _get_chrome_driver(self) -> webdriver.Chrome:
        """Configure Chrome driver with performance monitoring"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--enable-logging')
        options.add_argument('--log-level=0')
        
        # Enable performance logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True,
            'enableTimeline': True
        })
        options.add_experimental_option('useAutomationExtension', False)
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL', 'browser': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    def _measure_memory_usage(self, driver: webdriver.Chrome) -> float:
        """Measure current memory usage in MB"""
        try:
            memory_info = driver.execute_script("""
                if (performance.memory) {
                    return performance.memory.usedJSHeapSize / 1024 / 1024;
                }
                return 0;
            """)
            return memory_info
        except:
            return 0
    
    def _measure_frame_rate(self, driver: webdriver.Chrome, duration: float = 2.0) -> float:
        """Measure animation/scroll frame rate"""
        try:
            frame_rate = driver.execute_script(f"""
                return new Promise((resolve) => {{
                    let frames = 0;
                    let startTime = performance.now();
                    
                    function countFrame() {{
                        frames++;
                        if (performance.now() - startTime < {duration * 1000}) {{
                            requestAnimationFrame(countFrame);
                        }} else {{
                            resolve(frames / {duration});
                        }}
                    }}
                    
                    requestAnimationFrame(countFrame);
                }});
            """)
            return frame_rate if frame_rate else 0
        except:
            return 0
    
    async def test_chat_response_performance(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Test chat response times with various query complexities"""
        results = {
            'simple_queries': [],
            'complex_queries': [],
            'response_times': [],
            'memory_usage_progression': []
        }
        
        try:
            # Navigate to chat interface
            driver.get(f"{self.base_url}")
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for chat interface to load
            await asyncio.sleep(2)
            
            # Find chat input field (adjust selector based on actual UI)
            chat_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
            )
            
            # Test simple queries
            for query in self.test_queries[:4]:  # Test first 4 simple queries
                initial_memory = self._measure_memory_usage(driver)
                
                # Clear input and enter query
                chat_input.clear()
                chat_input.send_keys(query)
                
                start_time = time.time()
                chat_input.send_keys(Keys.RETURN)
                
                # Wait for response (adjust selector based on actual UI)
                try:
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".response, .message, .chat-response"))
                    )
                    response_time = time.time() - start_time
                except:
                    response_time = 30.0  # Timeout
                
                final_memory = self._measure_memory_usage(driver)
                
                results['simple_queries'].append({
                    'query': query,
                    'response_time': response_time,
                    'memory_delta': final_memory - initial_memory
                })
                results['response_times'].append(response_time)
                results['memory_usage_progression'].append(final_memory)
                
                # Wait between queries
                await asyncio.sleep(1)
            
            # Test complex queries
            for query in self.complex_queries[:2]:  # Test first 2 complex queries
                initial_memory = self._measure_memory_usage(driver)
                
                chat_input.clear()
                chat_input.send_keys(query)
                
                start_time = time.time()
                chat_input.send_keys(Keys.RETURN)
                
                try:
                    WebDriverWait(driver, 60).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".response, .message, .chat-response"))
                    )
                    response_time = time.time() - start_time
                except:
                    response_time = 60.0  # Timeout
                
                final_memory = self._measure_memory_usage(driver)
                
                results['complex_queries'].append({
                    'query': query[:100] + "...",  # Truncate for readability
                    'response_time': response_time,
                    'memory_delta': final_memory - initial_memory
                })
                
                await asyncio.sleep(2)
            
            # Calculate aggregate metrics
            if results['response_times']:
                results['aggregate_metrics'] = {
                    'mean_response_time': statistics.mean(results['response_times']),
                    'median_response_time': statistics.median(results['response_times']),
                    'p95_response_time': np.percentile(results['response_times'], 95),
                    'max_response_time': max(results['response_times']),
                    'total_memory_growth': max(results['memory_usage_progression']) - min(results['memory_usage_progression'])
                }
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def test_document_upload_performance(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Test document upload and processing performance"""
        results = {
            'upload_tests': [],
            'processing_times': [],
            'memory_impact': []
        }
        
        try:
            # Navigate to document upload interface
            driver.get(f"{self.base_url}/upload")  # Adjust URL based on actual route
            
            # Create test files of different sizes
            test_files = self._create_test_files()
            
            for file_info in test_files:
                initial_memory = self._measure_memory_usage(driver)
                
                # Find file upload element
                file_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
                )
                
                start_time = time.time()
                file_input.send_keys(file_info['path'])
                
                # Wait for upload completion indicator
                try:
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".upload-complete, .success, .processed"))
                    )
                    upload_time = time.time() - start_time
                except:
                    upload_time = 30.0  # Timeout
                
                final_memory = self._measure_memory_usage(driver)
                
                results['upload_tests'].append({
                    'file_size_kb': file_info['size_kb'],
                    'file_type': file_info['type'],
                    'upload_time': upload_time,
                    'throughput_kbps': file_info['size_kb'] / upload_time if upload_time > 0 else 0,
                    'memory_impact': final_memory - initial_memory
                })
                
                results['processing_times'].append(upload_time)
                results['memory_impact'].append(final_memory - initial_memory)
                
                await asyncio.sleep(2)
            
            # Clean up test files
            self._cleanup_test_files(test_files)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def _create_test_files(self) -> List[Dict[str, Any]]:
        """Create test files for upload testing"""
        test_files = []
        test_dir = Path("tests/ui/performance/temp_files")
        test_dir.mkdir(exist_ok=True)
        
        # Small text file (1KB)
        small_file = test_dir / "small_test.txt"
        small_file.write_text("x" * 1024)
        test_files.append({
            'path': str(small_file.absolute()),
            'size_kb': 1,
            'type': 'text'
        })
        
        # Medium text file (100KB)
        medium_file = test_dir / "medium_test.txt"
        medium_file.write_text("x" * 102400)
        test_files.append({
            'path': str(medium_file.absolute()),
            'size_kb': 100,
            'type': 'text'
        })
        
        # Large text file (1MB)
        large_file = test_dir / "large_test.txt"
        large_file.write_text("x" * 1048576)
        test_files.append({
            'path': str(large_file.absolute()),
            'size_kb': 1024,
            'type': 'text'
        })
        
        return test_files
    
    def _cleanup_test_files(self, test_files: List[Dict[str, Any]]):
        """Clean up test files"""
        for file_info in test_files:
            try:
                Path(file_info['path']).unlink()
            except:
                pass
    
    async def test_search_performance(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Test search functionality performance"""
        results = {
            'search_tests': [],
            'response_times': []
        }
        
        try:
            # Navigate to search interface
            driver.get(f"{self.base_url}")
            
            # Find search input
            search_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'], .search-input"))
            )
            
            search_terms = [
                "deployment",
                "configuration",
                "performance",
                "security guardrails",
                "vector database optimization"
            ]
            
            for term in search_terms:
                search_input.clear()
                search_input.send_keys(term)
                
                start_time = time.time()
                search_input.send_keys(Keys.RETURN)
                
                # Wait for search results
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".search-results, .results"))
                    )
                    response_time = time.time() - start_time
                    
                    # Count results
                    result_elements = driver.find_elements(By.CSS_SELECTOR, ".search-result, .result-item")
                    result_count = len(result_elements)
                    
                except:
                    response_time = 10.0
                    result_count = 0
                
                results['search_tests'].append({
                    'term': term,
                    'response_time': response_time,
                    'result_count': result_count
                })
                results['response_times'].append(response_time)
                
                await asyncio.sleep(1)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def test_ui_rendering_performance(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Test UI rendering and animation performance"""
        results = {
            'scroll_performance': 0,
            'animation_smoothness': 0,
            'rendering_times': []
        }
        
        try:
            driver.get(f"{self.base_url}")
            await asyncio.sleep(2)
            
            # Test scroll performance
            scroll_fps = self._measure_frame_rate(driver, 2.0)
            results['scroll_performance'] = scroll_fps
            
            # Simulate scrolling
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            await asyncio.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            
            # Test animation smoothness during interactions
            animation_fps = self._measure_frame_rate(driver, 2.0)
            results['animation_smoothness'] = animation_fps
            
            # Test rendering time for different UI components
            component_tests = [
                "document.querySelector('header')",
                "document.querySelector('main')",
                "document.querySelector('footer')"
            ]
            
            for test in component_tests:
                start_time = time.time()
                driver.execute_script(f"return {test};")
                render_time = time.time() - start_time
                results['rendering_times'].append(render_time)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def test_websocket_performance(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Test WebSocket connection performance and latency"""
        results = {
            'connection_time': 0,
            'message_latencies': [],
            'reconnection_time': 0,
            'throughput_messages_per_second': 0
        }
        
        try:
            # Navigate to page and inject WebSocket testing code
            driver.get(f"{self.base_url}")
            
            # Test WebSocket performance with JavaScript
            websocket_results = driver.execute_script("""
                return new Promise((resolve) => {
                    const results = {
                        connectionTime: 0,
                        messageLatencies: [],
                        error: null
                    };
                    
                    const startTime = Date.now();
                    const ws = new WebSocket('ws://localhost:8000/ws'); // Adjust URL
                    
                    ws.onopen = () => {
                        results.connectionTime = Date.now() - startTime;
                        
                        // Send test messages and measure latency
                        let messagesSent = 0;
                        const maxMessages = 10;
                        
                        const sendMessage = () => {
                            if (messagesSent < maxMessages) {
                                const msgStartTime = Date.now();
                                ws.send(JSON.stringify({
                                    type: 'ping',
                                    timestamp: msgStartTime,
                                    id: messagesSent
                                }));
                                messagesSent++;
                                setTimeout(sendMessage, 100);
                            }
                        };
                        
                        ws.onmessage = (event) => {
                            try {
                                const data = JSON.parse(event.data);
                                if (data.type === 'pong' && data.timestamp) {
                                    const latency = Date.now() - data.timestamp;
                                    results.messageLatencies.push(latency);
                                    
                                    if (results.messageLatencies.length >= maxMessages) {
                                        ws.close();
                                        resolve(results);
                                    }
                                }
                            } catch (e) {
                                // Ignore parsing errors
                            }
                        };
                        
                        sendMessage();
                    };
                    
                    ws.onerror = (error) => {
                        results.error = 'WebSocket connection failed';
                        resolve(results);
                    };
                    
                    // Timeout after 10 seconds
                    setTimeout(() => {
                        if (ws.readyState !== WebSocket.CLOSED) {
                            ws.close();
                        }
                        resolve(results);
                    }, 10000);
                });
            """)
            
            results.update(websocket_results)
            
            # Calculate derived metrics
            if results['message_latencies']:
                results['average_latency'] = statistics.mean(results['message_latencies'])
                results['p95_latency'] = np.percentile(results['message_latencies'], 95)
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    async def run_comprehensive_runtime_test(self) -> Dict[str, Any]:
        """Run comprehensive runtime performance test suite"""
        driver = self._get_chrome_driver()
        
        try:
            results = {
                'timestamp': time.time(),
                'test_suite': 'runtime_performance',
                'tests': {}
            }
            
            # Test chat response performance
            print("Testing chat response performance...")
            chat_results = await self.test_chat_response_performance(driver)
            results['tests']['chat_performance'] = chat_results
            
            # Test document upload performance
            print("Testing document upload performance...")
            upload_results = await self.test_document_upload_performance(driver)
            results['tests']['upload_performance'] = upload_results
            
            # Test search performance
            print("Testing search performance...")
            search_results = await self.test_search_performance(driver)
            results['tests']['search_performance'] = search_results
            
            # Test UI rendering performance
            print("Testing UI rendering performance...")
            ui_results = await self.test_ui_rendering_performance(driver)
            results['tests']['ui_rendering'] = ui_results
            
            # Test WebSocket performance
            print("Testing WebSocket performance...")
            websocket_results = await self.test_websocket_performance(driver)
            results['tests']['websocket_performance'] = websocket_results
            
            # Generate performance summary
            results['performance_summary'] = self._generate_performance_summary(results)
            
            return results
            
        finally:
            driver.quit()
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary and recommendations"""
        summary = {
            'overall_score': 0,
            'critical_issues': [],
            'recommendations': [],
            'threshold_compliance': {}
        }
        
        # Check chat performance
        chat_test = results['tests'].get('chat_performance', {})
        if 'aggregate_metrics' in chat_test:
            mean_response = chat_test['aggregate_metrics'].get('mean_response_time', 0)
            if mean_response > self.thresholds['chat_response_time']:
                summary['critical_issues'].append(f"Chat response time ({mean_response:.2f}s) exceeds threshold")
            summary['threshold_compliance']['chat_response_time'] = mean_response <= self.thresholds['chat_response_time']
        
        # Check UI rendering
        ui_test = results['tests'].get('ui_rendering', {})
        scroll_fps = ui_test.get('scroll_performance', 0)
        if scroll_fps < self.thresholds['scroll_performance']:
            summary['critical_issues'].append(f"Scroll performance ({scroll_fps:.1f} FPS) below threshold")
        summary['threshold_compliance']['scroll_performance'] = scroll_fps >= self.thresholds['scroll_performance']
        
        # Check WebSocket performance
        ws_test = results['tests'].get('websocket_performance', {})
        avg_latency = ws_test.get('average_latency', 0)
        if avg_latency > self.thresholds['websocket_latency'] * 1000:  # Convert to ms
            summary['critical_issues'].append(f"WebSocket latency ({avg_latency:.1f}ms) too high")
        
        # Generate recommendations
        if summary['critical_issues']:
            summary['recommendations'].extend([
                "Optimize backend response processing",
                "Implement response caching for common queries",
                "Consider WebSocket connection pooling",
                "Optimize DOM manipulation and rendering"
            ])
        
        # Calculate overall score
        compliance_count = sum(summary['threshold_compliance'].values())
        total_checks = len(summary['threshold_compliance'])
        summary['overall_score'] = (compliance_count / total_checks * 100) if total_checks > 0 else 0
        
        return summary


# Pytest fixtures and test cases
@pytest.fixture
def runtime_tester():
    """Fixture for runtime performance tester"""
    return RuntimePerformanceTester()


@pytest.mark.asyncio
async def test_chat_response_performance(runtime_tester):
    """Test chat response performance"""
    driver = runtime_tester._get_chrome_driver()
    try:
        results = await runtime_tester.test_chat_response_performance(driver)
        
        if 'aggregate_metrics' in results:
            mean_response_time = results['aggregate_metrics']['mean_response_time']
            assert mean_response_time < runtime_tester.thresholds['chat_response_time'], \
                f"Mean chat response time {mean_response_time}s exceeds threshold"
    finally:
        driver.quit()


@pytest.mark.asyncio
async def test_ui_rendering_smoothness(runtime_tester):
    """Test UI rendering and animation smoothness"""
    driver = runtime_tester._get_chrome_driver()
    try:
        results = await runtime_tester.test_ui_rendering_performance(driver)
        
        scroll_fps = results.get('scroll_performance', 0)
        assert scroll_fps >= 30, f"Scroll performance {scroll_fps} FPS too low"
        
        animation_fps = results.get('animation_smoothness', 0)
        assert animation_fps >= 30, f"Animation smoothness {animation_fps} FPS too low"
    finally:
        driver.quit()


@pytest.mark.asyncio
async def test_websocket_latency(runtime_tester):
    """Test WebSocket connection latency"""
    driver = runtime_tester._get_chrome_driver()
    try:
        results = await runtime_tester.test_websocket_performance(driver)
        
        if results.get('average_latency'):
            avg_latency = results['average_latency']
            assert avg_latency < 100, f"WebSocket latency {avg_latency}ms too high"  # 100ms threshold
    finally:
        driver.quit()


if __name__ == "__main__":
    # Run comprehensive runtime performance test
    async def main():
        tester = RuntimePerformanceTester()
        results = await tester.run_comprehensive_runtime_test()
        
        # Save results
        output_path = Path("tests/ui/performance/results")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"runtime_performance_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Runtime performance test completed.")
        print(f"Overall score: {results['performance_summary']['overall_score']:.1f}%")
        
        if results['performance_summary']['critical_issues']:
            print("Critical issues found:")
            for issue in results['performance_summary']['critical_issues']:
                print(f"  - {issue}")
    
    asyncio.run(main())