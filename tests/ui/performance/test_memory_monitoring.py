"""
Memory Usage and Resource Monitoring Suite for RAG Chatbot UI

Tests memory leak detection, CPU usage monitoring, network request optimization,
and resource consumption patterns during extended usage sessions.
"""

import asyncio
import time
import json
import psutil
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pytest
import numpy as np
from pathlib import Path
import threading
import queue


@dataclass
class ResourceMetrics:
    """Resource usage metrics container"""
    timestamp: float
    js_heap_used_mb: float
    js_heap_total_mb: float
    js_heap_limit_mb: float
    dom_nodes: int
    event_listeners: int
    cpu_usage_percent: float
    memory_usage_mb: float
    network_requests_count: int
    network_bytes_transferred: int
    active_websockets: int
    local_storage_size_kb: float
    session_storage_size_kb: float


class MemoryMonitor:
    """Continuous memory and resource monitoring"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_queue = queue.Queue()
        self.monitor_thread = None
    
    def start_monitoring(self, driver: webdriver.Chrome, interval: float = 1.0):
        """Start continuous monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(driver, interval)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        metrics = []
        while not self.metrics_queue.empty():
            metrics.append(self.metrics_queue.get())
        
        return metrics
    
    def _monitor_loop(self, driver: webdriver.Chrome, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics(driver)
                self.metrics_queue.put(metrics)
                time.sleep(interval)
            except Exception as e:
                # Continue monitoring even if individual collection fails
                time.sleep(interval)
    
    def _collect_metrics(self, driver: webdriver.Chrome) -> ResourceMetrics:
        """Collect comprehensive resource metrics"""
        
        # JavaScript heap metrics
        js_memory = driver.execute_script("""
            if (performance.memory) {
                return {
                    usedJSHeapSize: performance.memory.usedJSHeapSize / 1024 / 1024,
                    totalJSHeapSize: performance.memory.totalJSHeapSize / 1024 / 1024,
                    jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / 1024 / 1024
                };
            }
            return {usedJSHeapSize: 0, totalJSHeapSize: 0, jsHeapSizeLimit: 0};
        """)
        
        # DOM metrics
        dom_metrics = driver.execute_script("""
            return {
                domNodes: document.querySelectorAll('*').length,
                eventListeners: (function() {
                    let count = 0;
                    document.querySelectorAll('*').forEach(el => {
                        const listeners = getEventListeners ? getEventListeners(el) : {};
                        count += Object.keys(listeners).length;
                    });
                    return count;
                })()
            };
        """)
        
        # Network metrics
        network_metrics = driver.execute_script("""
            const entries = performance.getEntriesByType('resource');
            let totalBytes = 0;
            entries.forEach(entry => {
                totalBytes += entry.transferSize || 0;
            });
            
            return {
                requestCount: entries.length,
                totalBytes: totalBytes
            };
        """)
        
        # Storage metrics
        storage_metrics = driver.execute_script("""
            function getStorageSize(storage) {
                let total = 0;
                for (let key in storage) {
                    if (storage.hasOwnProperty(key)) {
                        total += storage[key].length + key.length;
                    }
                }
                return total;
            }
            
            return {
                localStorage: getStorageSize(localStorage) / 1024,
                sessionStorage: getStorageSize(sessionStorage) / 1024
            };
        """)
        
        # WebSocket connections
        websocket_count = driver.execute_script("""
            // This is an approximation - actual count would need tracking
            return window.activeWebSockets || 0;
        """)
        
        # System CPU and memory
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return ResourceMetrics(
            timestamp=time.time(),
            js_heap_used_mb=js_memory['usedJSHeapSize'],
            js_heap_total_mb=js_memory['totalJSHeapSize'],
            js_heap_limit_mb=js_memory['jsHeapSizeLimit'],
            dom_nodes=dom_metrics['domNodes'],
            event_listeners=dom_metrics['eventListeners'],
            cpu_usage_percent=cpu_percent,
            memory_usage_mb=memory_mb,
            network_requests_count=network_metrics['requestCount'],
            network_bytes_transferred=network_metrics['totalBytes'],
            active_websockets=websocket_count,
            local_storage_size_kb=storage_metrics['localStorage'],
            session_storage_size_kb=storage_metrics['sessionStorage']
        )


class MemoryLeakTester:
    """Memory leak detection and resource monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.thresholds = {
            'max_memory_growth_mb': 50,
            'max_dom_nodes': 5000,
            'max_event_listeners': 1000,
            'max_cpu_usage': 80,
            'memory_leak_threshold': 10  # MB growth without GC
        }
    
    def _get_chrome_driver(self) -> webdriver.Chrome:
        """Configure Chrome driver for memory monitoring"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--memory-pressure-off')
        
        # Enable detailed logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True,
            'enableTimeline': True
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL', 'browser': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    async def test_memory_leak_detection(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Test for memory leaks during extended usage"""
        driver = self._get_chrome_driver()
        monitor = MemoryMonitor()
        
        try:
            # Navigate to application
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Start monitoring
            monitor.start_monitoring(driver, interval=5.0)  # Monitor every 5 seconds
            
            # Simulate extended usage
            end_time = time.time() + (duration_minutes * 60)
            interaction_count = 0
            
            while time.time() < end_time:
                # Simulate various user interactions
                await self._simulate_user_interactions(driver, interaction_count)
                interaction_count += 1
                
                # Force garbage collection periodically
                if interaction_count % 10 == 0:
                    driver.execute_script("if (window.gc) window.gc();")
                
                await asyncio.sleep(10)  # Wait between interaction cycles
            
            # Stop monitoring and collect results
            metrics_history = monitor.stop_monitoring()
            
            # Analyze for memory leaks
            leak_analysis = self._analyze_memory_leaks(metrics_history)
            
            return {
                'duration_minutes': duration_minutes,
                'interaction_count': interaction_count,
                'metrics_history': [asdict(m) for m in metrics_history],
                'leak_analysis': leak_analysis,
                'final_metrics': asdict(metrics_history[-1]) if metrics_history else None
            }
            
        finally:
            monitor.stop_monitoring()
            driver.quit()
    
    async def _simulate_user_interactions(self, driver: webdriver.Chrome, cycle: int):
        """Simulate realistic user interactions"""
        try:
            # Chat interactions
            if cycle % 3 == 0:
                await self._simulate_chat_interaction(driver)
            
            # Navigation interactions
            elif cycle % 3 == 1:
                await self._simulate_navigation(driver)
            
            # Document interactions
            else:
                await self._simulate_document_interaction(driver)
                
        except Exception:
            # Continue even if individual interactions fail
            pass
    
    async def _simulate_chat_interaction(self, driver: webdriver.Chrome):
        """Simulate chat interactions"""
        try:
            # Find chat input
            chat_input = driver.find_element(By.CSS_SELECTOR, "input[type='text'], textarea")
            
            test_queries = [
                "What is the system architecture?",
                "How to configure security settings?",
                "Explain the deployment process",
                "What are the performance metrics?"
            ]
            
            query = test_queries[int(time.time()) % len(test_queries)]
            chat_input.clear()
            chat_input.send_keys(query)
            chat_input.send_keys(Keys.RETURN)
            
            # Wait for response
            await asyncio.sleep(3)
            
        except Exception:
            pass
    
    async def _simulate_navigation(self, driver: webdriver.Chrome):
        """Simulate navigation interactions"""
        try:
            # Scroll page
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            await asyncio.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            
            # Click various elements
            clickable_elements = driver.find_elements(By.CSS_SELECTOR, "button, a, .clickable")
            if clickable_elements:
                element = clickable_elements[0]
                element.click()
                await asyncio.sleep(1)
                
        except Exception:
            pass
    
    async def _simulate_document_interaction(self, driver: webdriver.Chrome):
        """Simulate document-related interactions"""
        try:
            # Search functionality
            search_input = driver.find_element(By.CSS_SELECTOR, "input[type='search'], .search-input")
            search_input.clear()
            search_input.send_keys("performance testing")
            search_input.send_keys(Keys.RETURN)
            await asyncio.sleep(2)
            
        except Exception:
            pass
    
    def _analyze_memory_leaks(self, metrics_history: List[ResourceMetrics]) -> Dict[str, Any]:
        """Analyze metrics for memory leaks"""
        if len(metrics_history) < 2:
            return {'error': 'Insufficient data for analysis'}
        
        initial_metrics = metrics_history[0]
        final_metrics = metrics_history[-1]
        
        # Calculate growth rates
        memory_growth = final_metrics.js_heap_used_mb - initial_metrics.js_heap_used_mb
        dom_growth = final_metrics.dom_nodes - initial_metrics.dom_nodes
        listener_growth = final_metrics.event_listeners - initial_metrics.event_listeners
        
        # Detect trends
        memory_values = [m.js_heap_used_mb for m in metrics_history]
        dom_values = [m.dom_nodes for m in metrics_history]
        
        # Calculate linear regression to detect consistent growth
        x = np.arange(len(memory_values))
        memory_slope = np.polyfit(x, memory_values, 1)[0] if len(memory_values) > 1 else 0
        dom_slope = np.polyfit(x, dom_values, 1)[0] if len(dom_values) > 1 else 0
        
        # Identify potential leaks
        potential_leaks = []
        
        if memory_growth > self.thresholds['memory_leak_threshold']:
            potential_leaks.append(f"JavaScript heap grew by {memory_growth:.2f}MB")
        
        if dom_growth > 500:
            potential_leaks.append(f"DOM nodes increased by {dom_growth}")
        
        if listener_growth > 100:
            potential_leaks.append(f"Event listeners increased by {listener_growth}")
        
        if memory_slope > 0.5:  # Growing more than 0.5MB per measurement
            potential_leaks.append("Consistent memory growth detected")
        
        return {
            'memory_growth_mb': memory_growth,
            'dom_node_growth': dom_growth,
            'event_listener_growth': listener_growth,
            'memory_growth_rate': memory_slope,
            'dom_growth_rate': dom_slope,
            'potential_leaks': potential_leaks,
            'leak_detected': len(potential_leaks) > 0,
            'severity': 'high' if len(potential_leaks) >= 2 else 'low' if potential_leaks else 'none'
        }
    
    async def test_resource_optimization(self) -> Dict[str, Any]:
        """Test resource usage optimization"""
        driver = self._get_chrome_driver()
        
        try:
            # Initial page load metrics
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            initial_metrics = MemoryMonitor()._collect_metrics(driver)
            
            # Perform intensive operations
            await self._perform_intensive_operations(driver)
            
            # Final metrics
            final_metrics = MemoryMonitor()._collect_metrics(driver)
            
            # Network optimization analysis
            network_analysis = self._analyze_network_optimization(driver)
            
            return {
                'initial_metrics': asdict(initial_metrics),
                'final_metrics': asdict(final_metrics),
                'resource_efficiency': self._calculate_resource_efficiency(initial_metrics, final_metrics),
                'network_optimization': network_analysis
            }
            
        finally:
            driver.quit()
    
    async def _perform_intensive_operations(self, driver: webdriver.Chrome):
        """Perform resource-intensive operations"""
        # Load multiple chat conversations
        for i in range(10):
            try:
                chat_input = driver.find_element(By.CSS_SELECTOR, "input[type='text'], textarea")
                chat_input.clear()
                chat_input.send_keys(f"Test query {i} with complex content and multiple parameters")
                chat_input.send_keys(Keys.RETURN)
                await asyncio.sleep(1)
            except Exception:
                pass
        
        # Intensive scrolling
        for _ in range(20):
            driver.execute_script("window.scrollBy(0, 100);")
            await asyncio.sleep(0.1)
    
    def _analyze_network_optimization(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """Analyze network request optimization"""
        network_data = driver.execute_script("""
            const entries = performance.getEntriesByType('resource');
            const analysis = {
                totalRequests: entries.length,
                totalBytes: 0,
                resourceTypes: {},
                cacheable: 0,
                compressed: 0,
                duplicates: 0
            };
            
            const urls = new Set();
            const duplicateUrls = new Set();
            
            entries.forEach(entry => {
                analysis.totalBytes += entry.transferSize || 0;
                
                const type = entry.initiatorType || 'other';
                analysis.resourceTypes[type] = (analysis.resourceTypes[type] || 0) + 1;
                
                // Check for duplicates
                if (urls.has(entry.name)) {
                    duplicateUrls.add(entry.name);
                    analysis.duplicates++;
                } else {
                    urls.add(entry.name);
                }
                
                // Check cache usage
                if (entry.transferSize === 0 && entry.decodedBodySize > 0) {
                    analysis.cacheable++;
                }
                
                // Check compression
                if (entry.encodedBodySize && entry.decodedBodySize) {
                    if (entry.encodedBodySize < entry.decodedBodySize) {
                        analysis.compressed++;
                    }
                }
            });
            
            return analysis;
        """)
        
        # Calculate optimization scores
        optimization_score = 0
        recommendations = []
        
        if network_data['cacheable'] / network_data['totalRequests'] > 0.5:
            optimization_score += 25
        else:
            recommendations.append("Improve caching for static resources")
        
        if network_data['compressed'] / network_data['totalRequests'] > 0.7:
            optimization_score += 25
        else:
            recommendations.append("Enable compression for more resources")
        
        if network_data['duplicates'] == 0:
            optimization_score += 25
        else:
            recommendations.append("Eliminate duplicate resource requests")
        
        if network_data['totalRequests'] < 50:
            optimization_score += 25
        else:
            recommendations.append("Reduce total number of network requests")
        
        return {
            'network_data': network_data,
            'optimization_score': optimization_score,
            'recommendations': recommendations
        }
    
    def _calculate_resource_efficiency(self, initial: ResourceMetrics, final: ResourceMetrics) -> Dict[str, Any]:
        """Calculate resource usage efficiency"""
        return {
            'memory_efficiency': max(0, 100 - ((final.js_heap_used_mb - initial.js_heap_used_mb) / initial.js_heap_used_mb * 100)),
            'dom_efficiency': max(0, 100 - ((final.dom_nodes - initial.dom_nodes) / max(initial.dom_nodes, 1) * 100)),
            'network_efficiency': 100 - min(100, (final.network_requests_count - initial.network_requests_count) / 10 * 10),
            'overall_cpu_usage': final.cpu_usage_percent
        }
    
    async def run_comprehensive_memory_test(self) -> Dict[str, Any]:
        """Run comprehensive memory and resource monitoring test"""
        results = {
            'timestamp': time.time(),
            'test_suite': 'memory_monitoring',
            'tests': {}
        }
        
        # Memory leak detection test
        print("Running memory leak detection test...")
        leak_results = await self.test_memory_leak_detection(duration_minutes=3)
        results['tests']['memory_leak_detection'] = leak_results
        
        # Resource optimization test
        print("Running resource optimization test...")
        optimization_results = await self.test_resource_optimization()
        results['tests']['resource_optimization'] = optimization_results
        
        # Generate summary
        results['summary'] = self._generate_memory_summary(results)
        
        return results
    
    def _generate_memory_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate memory test summary"""
        summary = {
            'overall_health': 'good',
            'critical_issues': [],
            'recommendations': []
        }
        
        # Check leak detection results
        leak_test = results['tests'].get('memory_leak_detection', {})
        if leak_test.get('leak_analysis', {}).get('leak_detected'):
            summary['overall_health'] = 'warning'
            summary['critical_issues'].extend(leak_test['leak_analysis']['potential_leaks'])
        
        # Check resource optimization
        opt_test = results['tests'].get('resource_optimization', {})
        if opt_test.get('network_optimization', {}).get('optimization_score', 100) < 75:
            summary['recommendations'].extend(opt_test['network_optimization']['recommendations'])
        
        return summary


# Pytest fixtures and test cases
@pytest.fixture
def memory_tester():
    """Fixture for memory leak tester"""
    return MemoryLeakTester()


@pytest.mark.asyncio
async def test_memory_leak_detection(memory_tester):
    """Test memory leak detection"""
    results = await memory_tester.test_memory_leak_detection(duration_minutes=2)
    
    leak_analysis = results.get('leak_analysis', {})
    assert not leak_analysis.get('leak_detected', False), \
        f"Memory leaks detected: {leak_analysis.get('potential_leaks', [])}"


@pytest.mark.asyncio
async def test_resource_efficiency(memory_tester):
    """Test resource usage efficiency"""
    results = await memory_tester.test_resource_optimization()
    
    efficiency = results.get('resource_efficiency', {})
    assert efficiency.get('memory_efficiency', 0) > 50, "Memory efficiency too low"
    assert efficiency.get('overall_cpu_usage', 100) < 80, "CPU usage too high"


if __name__ == "__main__":
    # Run comprehensive memory monitoring test
    async def main():
        tester = MemoryLeakTester()
        results = await tester.run_comprehensive_memory_test()
        
        # Save results
        output_path = Path("tests/ui/performance/results")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"memory_monitoring_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Memory monitoring test completed.")
        print(f"Overall health: {results['summary']['overall_health']}")
        
        if results['summary']['critical_issues']:
            print("Critical issues found:")
            for issue in results['summary']['critical_issues']:
                print(f"  - {issue}")
    
    asyncio.run(main())