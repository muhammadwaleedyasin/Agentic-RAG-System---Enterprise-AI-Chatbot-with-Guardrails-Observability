"""
Performance Benchmarking and Reporting Suite for RAG Chatbot UI

Comprehensive performance benchmarking with detailed reporting,
trend analysis, and optimization recommendations.
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import csv


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    test_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: float
    browser: str
    device_type: str
    baseline_value: Optional[float] = None
    threshold: Optional[float] = None
    status: str = "unknown"  # pass, fail, warning


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_name: str
    trend_direction: str  # improving, degrading, stable
    change_percentage: float
    current_value: float
    baseline_value: float
    confidence: float
    recommendations: List[str]


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking and analysis"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.benchmark_config = {
            'page_load_time': {'threshold': 2.0, 'unit': 'seconds', 'baseline': None},
            'first_contentful_paint': {'threshold': 1.5, 'unit': 'seconds', 'baseline': None},
            'largest_contentful_paint': {'threshold': 2.5, 'unit': 'seconds', 'baseline': None},
            'time_to_interactive': {'threshold': 3.0, 'unit': 'seconds', 'baseline': None},
            'cumulative_layout_shift': {'threshold': 0.1, 'unit': 'score', 'baseline': None},
            'first_input_delay': {'threshold': 0.1, 'unit': 'seconds', 'baseline': None},
            'total_blocking_time': {'threshold': 0.3, 'unit': 'seconds', 'baseline': None},
            'memory_usage': {'threshold': 100, 'unit': 'MB', 'baseline': None},
            'network_requests': {'threshold': 50, 'unit': 'count', 'baseline': None},
            'transfer_size': {'threshold': 2048, 'unit': 'KB', 'baseline': None},
            'dom_nodes': {'threshold': 1500, 'unit': 'count', 'baseline': None},
            'event_listeners': {'threshold': 500, 'unit': 'count', 'baseline': None}
        }
        
        self.test_scenarios = [
            {'name': 'cold_start', 'description': 'First page load with empty cache'},
            {'name': 'warm_cache', 'description': 'Page load with populated cache'},
            {'name': 'heavy_interaction', 'description': 'Multiple chat interactions'},
            {'name': 'document_upload', 'description': 'Document upload and processing'},
            {'name': 'search_operations', 'description': 'Multiple search operations'},
            {'name': 'extended_session', 'description': 'Extended usage session'}
        ]
    
    def _get_chrome_driver(self, enable_lighthouse: bool = False) -> webdriver.Chrome:
        """Configure Chrome driver for performance benchmarking"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        if enable_lighthouse:
            # Enable Lighthouse integration
            options.add_argument('--enable-automation')
            options.add_argument('--disable-background-timer-throttling')
            options.add_argument('--disable-backgrounding-occluded-windows')
            options.add_argument('--disable-renderer-backgrounding')
        
        # Enable performance monitoring
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True,
            'enableTimeline': True
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    async def run_benchmark_scenario(self, scenario: Dict[str, str], browser: str = 'chrome', device: str = 'desktop') -> List[BenchmarkResult]:
        """Run a specific benchmark scenario"""
        driver = self._get_chrome_driver()
        results = []
        
        try:
            if scenario['name'] == 'cold_start':
                results = await self._benchmark_cold_start(driver, browser, device)
            elif scenario['name'] == 'warm_cache':
                results = await self._benchmark_warm_cache(driver, browser, device)
            elif scenario['name'] == 'heavy_interaction':
                results = await self._benchmark_heavy_interaction(driver, browser, device)
            elif scenario['name'] == 'document_upload':
                results = await self._benchmark_document_upload(driver, browser, device)
            elif scenario['name'] == 'search_operations':
                results = await self._benchmark_search_operations(driver, browser, device)
            elif scenario['name'] == 'extended_session':
                results = await self._benchmark_extended_session(driver, browser, device)
        
        finally:
            driver.quit()
        
        return results
    
    async def _benchmark_cold_start(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark cold start performance"""
        results = []
        
        # Clear cache and storage
        driver.execute_cdp_cmd('Network.clearBrowserCache', {})
        driver.execute_script("localStorage.clear(); sessionStorage.clear();")
        
        # Measure page load performance
        start_time = time.time()
        driver.get(self.base_url)
        
        # Wait for page to be fully loaded
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get comprehensive performance metrics
        performance_metrics = self._extract_comprehensive_metrics(driver)
        
        # Convert to BenchmarkResult objects
        timestamp = time.time()
        for metric_name, value in performance_metrics.items():
            config = self.benchmark_config.get(metric_name, {})
            
            status = "pass"
            if config.get('threshold') and value > config['threshold']:
                status = "fail"
            elif config.get('threshold') and value > config['threshold'] * 0.8:
                status = "warning"
            
            result = BenchmarkResult(
                test_name="cold_start",
                metric_name=metric_name,
                value=value,
                unit=config.get('unit', 'unknown'),
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=config.get('threshold'),
                status=status
            )
            results.append(result)
        
        return results
    
    async def _benchmark_warm_cache(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark warm cache performance"""
        results = []
        
        # First load to populate cache
        driver.get(self.base_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        await asyncio.sleep(2)
        
        # Second load with warm cache
        start_time = time.time()
        driver.refresh()
        
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        performance_metrics = self._extract_comprehensive_metrics(driver)
        
        timestamp = time.time()
        for metric_name, value in performance_metrics.items():
            config = self.benchmark_config.get(metric_name, {})
            
            # Warm cache should be faster
            warm_threshold = config.get('threshold', float('inf')) * 0.7 if config.get('threshold') else None
            
            status = "pass"
            if warm_threshold and value > warm_threshold:
                status = "fail"
            elif warm_threshold and value > warm_threshold * 0.8:
                status = "warning"
            
            result = BenchmarkResult(
                test_name="warm_cache",
                metric_name=metric_name,
                value=value,
                unit=config.get('unit', 'unknown'),
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=warm_threshold,
                status=status
            )
            results.append(result)
        
        return results
    
    async def _benchmark_heavy_interaction(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark performance under heavy interaction"""
        results = []
        
        driver.get(self.base_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get initial metrics
        initial_metrics = self._extract_comprehensive_metrics(driver)
        
        # Perform heavy interactions
        start_time = time.time()
        
        # Simulate multiple chat interactions
        for i in range(10):
            try:
                chat_input = driver.find_element(By.CSS_SELECTOR, "input[type='text'], textarea")
                chat_input.clear()
                chat_input.send_keys(f"Heavy interaction test message {i}")
                chat_input.send_keys("\n")
                await asyncio.sleep(1)
            except Exception:
                continue
        
        # Get final metrics
        final_metrics = self._extract_comprehensive_metrics(driver)
        interaction_time = time.time() - start_time
        
        # Calculate metric deltas and rates
        timestamp = time.time()
        
        # Add interaction-specific metrics
        results.append(BenchmarkResult(
            test_name="heavy_interaction",
            metric_name="interaction_completion_time",
            value=interaction_time,
            unit="seconds",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=15.0,  # 15 seconds for 10 interactions
            status="pass" if interaction_time < 15.0 else "fail"
        ))
        
        # Memory growth during interactions
        memory_growth = final_metrics.get('memory_usage', 0) - initial_metrics.get('memory_usage', 0)
        results.append(BenchmarkResult(
            test_name="heavy_interaction",
            metric_name="memory_growth",
            value=memory_growth,
            unit="MB",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=25.0,  # 25MB growth limit
            status="pass" if memory_growth < 25.0 else "fail"
        ))
        
        return results
    
    async def _benchmark_document_upload(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark document upload performance"""
        results = []
        
        driver.get(self.base_url)
        
        # Create test document
        test_file_path = self._create_benchmark_document()
        
        try:
            # Navigate to upload page
            driver.get(f"{self.base_url}/upload")
            
            # Measure upload performance
            start_time = time.time()
            
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            file_input.send_keys(test_file_path)
            
            # Wait for upload completion
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".upload-complete, .success"))
                )
            except Exception:
                pass
            
            upload_time = time.time() - start_time
            
            # Get performance metrics
            performance_metrics = self._extract_comprehensive_metrics(driver)
            
            timestamp = time.time()
            
            # Add upload-specific metrics
            results.append(BenchmarkResult(
                test_name="document_upload",
                metric_name="upload_completion_time",
                value=upload_time,
                unit="seconds",
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=10.0,  # 10 seconds for document upload
                status="pass" if upload_time < 10.0 else "fail"
            ))
            
            # Add general performance metrics
            for metric_name, value in performance_metrics.items():
                config = self.benchmark_config.get(metric_name, {})
                
                result = BenchmarkResult(
                    test_name="document_upload",
                    metric_name=metric_name,
                    value=value,
                    unit=config.get('unit', 'unknown'),
                    timestamp=timestamp,
                    browser=browser,
                    device_type=device,
                    threshold=config.get('threshold'),
                    status="pass" if not config.get('threshold') or value <= config['threshold'] else "fail"
                )
                results.append(result)
        
        finally:
            # Clean up test file
            Path(test_file_path).unlink(missing_ok=True)
        
        return results
    
    async def _benchmark_search_operations(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark search operation performance"""
        results = []
        
        driver.get(self.base_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        search_terms = [
            "deployment",
            "configuration",
            "performance",
            "security",
            "troubleshooting"
        ]
        
        search_times = []
        
        for term in search_terms:
            try:
                search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search'], .search-input")
                
                start_time = time.time()
                search_input.clear()
                search_input.send_keys(term)
                search_input.send_keys("\n")
                
                # Wait for search results
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".search-results, .results"))
                    )
                except Exception:
                    pass
                
                search_time = time.time() - start_time
                search_times.append(search_time)
                
                await asyncio.sleep(1)
                
            except Exception:
                continue
        
        # Calculate search performance metrics
        timestamp = time.time()
        
        if search_times:
            avg_search_time = statistics.mean(search_times)
            max_search_time = max(search_times)
            
            results.append(BenchmarkResult(
                test_name="search_operations",
                metric_name="average_search_time",
                value=avg_search_time,
                unit="seconds",
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=2.0,  # 2 seconds average
                status="pass" if avg_search_time < 2.0 else "fail"
            ))
            
            results.append(BenchmarkResult(
                test_name="search_operations",
                metric_name="max_search_time",
                value=max_search_time,
                unit="seconds",
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=5.0,  # 5 seconds max
                status="pass" if max_search_time < 5.0 else "fail"
            ))
        
        # Get general performance metrics
        performance_metrics = self._extract_comprehensive_metrics(driver)
        for metric_name, value in performance_metrics.items():
            config = self.benchmark_config.get(metric_name, {})
            
            result = BenchmarkResult(
                test_name="search_operations",
                metric_name=metric_name,
                value=value,
                unit=config.get('unit', 'unknown'),
                timestamp=timestamp,
                browser=browser,
                device_type=device,
                threshold=config.get('threshold'),
                status="pass" if not config.get('threshold') or value <= config['threshold'] else "fail"
            )
            results.append(result)
        
        return results
    
    async def _benchmark_extended_session(self, driver: webdriver.Chrome, browser: str, device: str) -> List[BenchmarkResult]:
        """Benchmark extended session performance"""
        results = []
        
        driver.get(self.base_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Get initial metrics
        initial_metrics = self._extract_comprehensive_metrics(driver)
        start_time = time.time()
        
        # Simulate extended usage (5 minutes)
        session_duration = 300  # 5 minutes
        end_time = start_time + session_duration
        
        interaction_count = 0
        while time.time() < end_time:
            # Perform various interactions
            try:
                if interaction_count % 3 == 0:
                    # Chat interaction
                    chat_input = driver.find_element(By.CSS_SELECTOR, "input[type='text'], textarea")
                    chat_input.clear()
                    chat_input.send_keys(f"Extended session message {interaction_count}")
                    chat_input.send_keys("\n")
                
                elif interaction_count % 3 == 1:
                    # Scroll page
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    await asyncio.sleep(0.5)
                    driver.execute_script("window.scrollTo(0, 0);")
                
                else:
                    # Search interaction
                    search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search'], .search-input")
                    search_input.clear()
                    search_input.send_keys("test query")
                    search_input.send_keys("\n")
                
                interaction_count += 1
                await asyncio.sleep(5)  # 5 seconds between interactions
                
            except Exception:
                await asyncio.sleep(5)
                continue
        
        # Get final metrics
        final_metrics = self._extract_comprehensive_metrics(driver)
        actual_duration = time.time() - start_time
        
        timestamp = time.time()
        
        # Calculate session-specific metrics
        memory_growth = final_metrics.get('memory_usage', 0) - initial_metrics.get('memory_usage', 0)
        dom_growth = final_metrics.get('dom_nodes', 0) - initial_metrics.get('dom_nodes', 0)
        
        results.append(BenchmarkResult(
            test_name="extended_session",
            metric_name="session_duration",
            value=actual_duration,
            unit="seconds",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=session_duration,
            status="pass"
        ))
        
        results.append(BenchmarkResult(
            test_name="extended_session",
            metric_name="memory_growth_rate",
            value=memory_growth / (actual_duration / 60),  # MB per minute
            unit="MB/min",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=5.0,  # 5MB per minute max
            status="pass" if (memory_growth / (actual_duration / 60)) < 5.0 else "fail"
        ))
        
        results.append(BenchmarkResult(
            test_name="extended_session",
            metric_name="dom_growth_rate",
            value=dom_growth / (actual_duration / 60),  # nodes per minute
            unit="nodes/min",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=50.0,  # 50 nodes per minute max
            status="pass" if (dom_growth / (actual_duration / 60)) < 50.0 else "fail"
        ))
        
        results.append(BenchmarkResult(
            test_name="extended_session",
            metric_name="interactions_completed",
            value=interaction_count,
            unit="count",
            timestamp=timestamp,
            browser=browser,
            device_type=device,
            threshold=50,  # Expect at least 50 interactions
            status="pass" if interaction_count >= 50 else "warning"
        ))
        
        return results
    
    def _extract_comprehensive_metrics(self, driver: webdriver.Chrome) -> Dict[str, float]:
        """Extract comprehensive performance metrics"""
        try:
            return driver.execute_script("""
                const timing = performance.timing;
                const navigation = performance.getEntriesByType('navigation')[0];
                const paintEntries = performance.getEntriesByType('paint');
                
                // Core Web Vitals and performance metrics
                const metrics = {
                    page_load_time: (timing.loadEventEnd - timing.navigationStart) / 1000,
                    dom_content_loaded: (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000,
                    time_to_interactive: (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000,
                    first_contentful_paint: 0,
                    largest_contentful_paint: 0,
                    cumulative_layout_shift: 0,
                    first_input_delay: 0,
                    total_blocking_time: 0,
                    memory_usage: 0,
                    dom_nodes: document.querySelectorAll('*').length,
                    event_listeners: 0,
                    network_requests: 0,
                    transfer_size: 0
                };
                
                // Paint metrics
                paintEntries.forEach(entry => {
                    if (entry.name === 'first-contentful-paint') {
                        metrics.first_contentful_paint = entry.startTime / 1000;
                    }
                });
                
                // LCP (approximate)
                const lcpEntries = performance.getEntriesByType('largest-contentful-paint');
                if (lcpEntries.length > 0) {
                    metrics.largest_contentful_paint = lcpEntries[lcpEntries.length - 1].startTime / 1000;
                }
                
                // CLS (approximate)
                const clsEntries = performance.getEntriesByType('layout-shift');
                metrics.cumulative_layout_shift = clsEntries.reduce((sum, entry) => {
                    return sum + (entry.hadRecentInput ? 0 : entry.value);
                }, 0);
                
                // Memory usage
                if (performance.memory) {
                    metrics.memory_usage = performance.memory.usedJSHeapSize / 1024 / 1024;
                }
                
                // Network metrics
                const resourceEntries = performance.getEntriesByType('resource');
                metrics.network_requests = resourceEntries.length;
                metrics.transfer_size = resourceEntries.reduce((sum, entry) => {
                    return sum + (entry.transferSize || 0);
                }, 0) / 1024; // Convert to KB
                
                // Event listeners (approximate)
                let listenerCount = 0;
                document.querySelectorAll('*').forEach(el => {
                    const listeners = getEventListeners ? getEventListeners(el) : {};
                    listenerCount += Object.keys(listeners).length;
                });
                metrics.event_listeners = listenerCount;
                
                return metrics;
            """)
        except Exception:
            return {}
    
    def _create_benchmark_document(self) -> str:
        """Create a test document for upload benchmarking"""
        test_dir = Path("tests/ui/performance/temp")
        test_dir.mkdir(exist_ok=True)
        
        # Create a moderately sized document
        content = "This is a benchmark test document.\n" * 1000  # ~33KB
        test_file = test_dir / f"benchmark_doc_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text(content)
        
        return str(test_file.absolute())
    
    def save_benchmark_results(self, results: List[BenchmarkResult], output_dir: Path = None) -> str:
        """Save benchmark results to file"""
        if output_dir is None:
            output_dir = Path("tests/ui/performance/benchmarks")
        
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        timestamp = int(time.time())
        json_file = output_dir / f"benchmark_results_{timestamp}.json"
        
        results_data = {
            'timestamp': time.time(),
            'results': [asdict(result) for result in results],
            'summary': self._generate_benchmark_summary(results)
        }
        
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        # Save as CSV for analysis
        csv_file = output_dir / f"benchmark_results_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['test_name', 'metric_name', 'value', 'unit', 'timestamp', 'browser', 'device_type', 'status'])
            
            for result in results:
                writer.writerow([
                    result.test_name, result.metric_name, result.value, result.unit,
                    result.timestamp, result.browser, result.device_type, result.status
                ])
        
        return str(json_file)
    
    def _generate_benchmark_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate benchmark summary statistics"""
        if not results:
            return {}
        
        # Group by test and metric
        by_test = {}
        by_metric = {}
        
        for result in results:
            # By test
            if result.test_name not in by_test:
                by_test[result.test_name] = []
            by_test[result.test_name].append(result)
            
            # By metric
            if result.metric_name not in by_metric:
                by_metric[result.metric_name] = []
            by_metric[result.metric_name].append(result)
        
        # Status summary
        status_counts = {'pass': 0, 'fail': 0, 'warning': 0, 'unknown': 0}
        for result in results:
            status_counts[result.status] += 1
        
        # Performance summary by metric
        metric_summary = {}
        for metric_name, metric_results in by_metric.items():
            values = [r.value for r in metric_results]
            metric_summary[metric_name] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'count': len(values),
                'unit': metric_results[0].unit
            }
        
        # Test summary
        test_summary = {}
        for test_name, test_results in by_test.items():
            failed_metrics = [r.metric_name for r in test_results if r.status == 'fail']
            warning_metrics = [r.metric_name for r in test_results if r.status == 'warning']
            
            test_summary[test_name] = {
                'total_metrics': len(test_results),
                'passed_metrics': len([r for r in test_results if r.status == 'pass']),
                'failed_metrics': failed_metrics,
                'warning_metrics': warning_metrics,
                'success_rate': len([r for r in test_results if r.status == 'pass']) / len(test_results) * 100
            }
        
        return {
            'total_results': len(results),
            'status_summary': status_counts,
            'overall_success_rate': status_counts['pass'] / len(results) * 100,
            'metric_summary': metric_summary,
            'test_summary': test_summary,
            'critical_failures': [r.metric_name for r in results if r.status == 'fail' and r.metric_name in ['page_load_time', 'memory_usage']],
            'recommendations': self._generate_benchmark_recommendations(results)
        }
    
    def _generate_benchmark_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []
        
        # Check for performance issues
        failed_results = [r for r in results if r.status == 'fail']
        
        page_load_failures = [r for r in failed_results if 'page_load' in r.metric_name]
        if page_load_failures:
            recommendations.append("Optimize initial page load performance")
        
        memory_failures = [r for r in failed_results if 'memory' in r.metric_name]
        if memory_failures:
            recommendations.append("Investigate memory usage and potential leaks")
        
        network_failures = [r for r in failed_results if 'network' in r.metric_name or 'transfer' in r.metric_name]
        if network_failures:
            recommendations.append("Optimize network requests and payload sizes")
        
        interaction_failures = [r for r in failed_results if 'interaction' in r.metric_name or 'search' in r.metric_name]
        if interaction_failures:
            recommendations.append("Improve interactive performance and response times")
        
        # Check for warnings
        warning_results = [r for r in results if r.status == 'warning']
        if warning_results:
            recommendations.append("Address performance warnings to prevent future issues")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance benchmarks are within acceptable ranges")
        else:
            recommendations.extend([
                "Consider implementing performance monitoring in production",
                "Set up automated performance regression testing",
                "Review and optimize critical user paths"
            ])
        
        return recommendations
    
    async def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite"""
        all_results = []
        
        print("Running comprehensive performance benchmark suite...")
        
        # Run all benchmark scenarios
        for scenario in self.test_scenarios:
            print(f"Running scenario: {scenario['name']}")
            scenario_results = await self.run_benchmark_scenario(scenario)
            all_results.extend(scenario_results)
        
        # Save results
        results_file = self.save_benchmark_results(all_results)
        
        # Generate comprehensive report
        report = {
            'timestamp': time.time(),
            'benchmark_suite': 'comprehensive_performance',
            'total_scenarios': len(self.test_scenarios),
            'total_metrics': len(all_results),
            'results_file': results_file,
            'summary': self._generate_benchmark_summary(all_results)
        }
        
        return report


# Pytest fixtures and test cases
@pytest.fixture
def benchmark_suite():
    """Fixture for performance benchmark suite"""
    return PerformanceBenchmarkSuite()


@pytest.mark.asyncio
async def test_cold_start_benchmark(benchmark_suite):
    """Test cold start performance benchmark"""
    scenario = {'name': 'cold_start', 'description': 'First page load with empty cache'}
    results = await benchmark_suite.run_benchmark_scenario(scenario)
    
    assert len(results) > 0, "No benchmark results returned"
    
    # Check critical performance metrics
    page_load_results = [r for r in results if r.metric_name == 'page_load_time']
    assert len(page_load_results) > 0, "Page load time not measured"
    
    page_load_time = page_load_results[0].value
    assert page_load_time < 5.0, f"Page load time {page_load_time}s exceeds maximum threshold"


@pytest.mark.asyncio
async def test_memory_benchmark(benchmark_suite):
    """Test memory usage benchmark"""
    scenario = {'name': 'heavy_interaction', 'description': 'Multiple chat interactions'}
    results = await benchmark_suite.run_benchmark_scenario(scenario)
    
    memory_growth_results = [r for r in results if r.metric_name == 'memory_growth']
    if memory_growth_results:
        memory_growth = memory_growth_results[0].value
        assert memory_growth < 50.0, f"Memory growth {memory_growth}MB exceeds threshold"


if __name__ == "__main__":
    # Run comprehensive benchmark suite
    async def main():
        suite = PerformanceBenchmarkSuite()
        report = await suite.run_comprehensive_benchmark_suite()
        
        print("Performance benchmark suite completed.")
        print(f"Total scenarios: {report['total_scenarios']}")
        print(f"Total metrics: {report['total_metrics']}")
        print(f"Overall success rate: {report['summary']['overall_success_rate']:.1f}%")
        print(f"Results saved to: {report['results_file']}")
        
        if report['summary']['critical_failures']:
            print("\nCritical failures:")
            for failure in report['summary']['critical_failures']:
                print(f"  - {failure}")
        
        if report['summary']['recommendations']:
            print("\nRecommendations:")
            for rec in report['summary']['recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())