"""
Page Load Performance Testing Suite for RAG Chatbot UI

Tests initial page load times, asset loading optimization, and progressive enhancement.
Validates performance metrics against established benchmarks.
"""

import asyncio
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import numpy as np
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    page_load_time: float
    dom_content_loaded: float
    first_contentful_paint: float
    largest_contentful_paint: float
    first_input_delay: float
    cumulative_layout_shift: float
    total_blocking_time: float
    network_requests: int
    resource_sizes: Dict[str, int]
    memory_usage: Dict[str, float]


class PageLoadPerformanceTester:
    """Comprehensive page load performance testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.thresholds = {
            'page_load_time': 2.0,  # seconds
            'first_contentful_paint': 1.5,
            'largest_contentful_paint': 2.5,
            'first_input_delay': 0.1,
            'cumulative_layout_shift': 0.1,
            'total_blocking_time': 0.3
        }
    
    def _get_chrome_driver(self, mobile: bool = False) -> webdriver.Chrome:
        """Configure Chrome driver with performance logging"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--enable-logging')
        options.add_argument('--log-level=0')
        
        if mobile:
            mobile_emulation = {
                "deviceMetrics": {"width": 375, "height": 667, "pixelRatio": 3.0},
                "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
            }
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Enable performance logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True,
            'enableTimeline': True
        })
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    def _get_firefox_driver(self) -> webdriver.Firefox:
        """Configure Firefox driver with performance logging"""
        options = FirefoxOptions()
        options.add_argument('--headless')
        
        # Enable performance API
        profile = webdriver.FirefoxProfile()
        profile.set_preference('dom.enable_performance', True)
        profile.set_preference('dom.enable_performance_navigation_timing', True)
        profile.set_preference('dom.enable_resource_timing', True)
        
        return webdriver.Firefox(options=options, firefox_profile=profile)
    
    def _extract_performance_metrics(self, driver: webdriver.Chrome) -> PerformanceMetrics:
        """Extract comprehensive performance metrics from browser"""
        
        # Get Navigation Timing API data
        navigation_timing = driver.execute_script("""
            const timing = performance.timing;
            const navigation = performance.getEntriesByType('navigation')[0];
            return {
                pageLoadTime: (timing.loadEventEnd - timing.navigationStart) / 1000,
                domContentLoaded: (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000,
                firstByte: (timing.responseStart - timing.navigationStart) / 1000,
                domInteractive: (timing.domInteractive - timing.navigationStart) / 1000,
                navigationStart: timing.navigationStart,
                loadComplete: timing.loadEventEnd,
                transferSize: navigation ? navigation.transferSize : 0,
                encodedBodySize: navigation ? navigation.encodedBodySize : 0
            };
        """)
        
        # Get Paint Timing API data
        paint_timing = driver.execute_script("""
            const paintEntries = performance.getEntriesByType('paint');
            const result = {};
            paintEntries.forEach(entry => {
                result[entry.name.replace('-', '_')] = entry.startTime / 1000;
            });
            return result;
        """)
        
        # Get Layout Shift metrics
        layout_shift = driver.execute_script("""
            try {
                const cls = performance.getEntriesByType('layout-shift')
                    .reduce((total, entry) => total + entry.value, 0);
                return cls;
            } catch (e) {
                return 0;
            }
        """)
        
        # Get Resource Timing data
        resource_timing = driver.execute_script("""
            const resources = performance.getEntriesByType('resource');
            const resourceSizes = {};
            let totalRequests = 0;
            
            resources.forEach(resource => {
                const type = resource.initiatorType || 'other';
                if (!resourceSizes[type]) resourceSizes[type] = 0;
                resourceSizes[type] += resource.transferSize || 0;
                totalRequests++;
            });
            
            return {
                resourceSizes: resourceSizes,
                totalRequests: totalRequests
            };
        """)
        
        # Get memory usage if available
        memory_info = driver.execute_script("""
            try {
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize / 1024 / 1024,
                        totalJSHeapSize: performance.memory.totalJSHeapSize / 1024 / 1024,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit / 1024 / 1024
                    };
                }
                return {};
            } catch (e) {
                return {};
            }
        """)
        
        return PerformanceMetrics(
            page_load_time=navigation_timing.get('pageLoadTime', 0),
            dom_content_loaded=navigation_timing.get('domContentLoaded', 0),
            first_contentful_paint=paint_timing.get('first_contentful_paint', 0),
            largest_contentful_paint=paint_timing.get('largest_contentful_paint', 0),
            first_input_delay=0,  # Would need user interaction to measure
            cumulative_layout_shift=layout_shift,
            total_blocking_time=0,  # Would need lighthouse for accurate measurement
            network_requests=resource_timing['totalRequests'],
            resource_sizes=resource_timing['resourceSizes'],
            memory_usage=memory_info
        )
    
    async def test_initial_page_load(self, browser: str = 'chrome', mobile: bool = False) -> PerformanceMetrics:
        """Test initial page load performance"""
        
        if browser == 'chrome':
            driver = self._get_chrome_driver(mobile=mobile)
        else:
            driver = self._get_firefox_driver()
        
        try:
            start_time = time.time()
            driver.get(self.base_url)
            
            # Wait for page to be fully loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for any async content to load
            await asyncio.sleep(2)
            
            metrics = self._extract_performance_metrics(driver)
            
            # Add test metadata
            metrics.browser = browser
            metrics.mobile = mobile
            metrics.test_timestamp = time.time()
            
            return metrics
            
        finally:
            driver.quit()
    
    async def test_asset_loading_performance(self) -> Dict[str, Any]:
        """Test asset loading optimization and caching"""
        driver = self._get_chrome_driver()
        
        try:
            # First load - measure cold cache performance
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            cold_metrics = self._extract_performance_metrics(driver)
            
            # Second load - measure warm cache performance
            driver.refresh()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            warm_metrics = self._extract_performance_metrics(driver)
            
            # Calculate cache effectiveness
            cache_improvement = {
                'page_load_improvement': (cold_metrics.page_load_time - warm_metrics.page_load_time) / cold_metrics.page_load_time * 100,
                'requests_reduction': cold_metrics.network_requests - warm_metrics.network_requests,
                'size_reduction': sum(cold_metrics.resource_sizes.values()) - sum(warm_metrics.resource_sizes.values())
            }
            
            return {
                'cold_cache': cold_metrics,
                'warm_cache': warm_metrics,
                'cache_effectiveness': cache_improvement
            }
            
        finally:
            driver.quit()
    
    async def test_progressive_enhancement(self) -> Dict[str, Any]:
        """Test progressive enhancement and critical path rendering"""
        driver = self._get_chrome_driver()
        
        try:
            # Disable JavaScript to test basic functionality
            driver.execute_cdp_cmd('Emulation.setScriptExecutionDisabled', {'value': True})
            
            driver.get(self.base_url)
            
            # Check if basic content is available without JS
            basic_content_available = True
            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "main"))
                )
            except:
                basic_content_available = False
            
            # Re-enable JavaScript
            driver.execute_cdp_cmd('Emulation.setScriptExecutionDisabled', {'value': False})
            driver.refresh()
            
            # Wait for full functionality
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            enhanced_metrics = self._extract_performance_metrics(driver)
            
            return {
                'basic_content_available': basic_content_available,
                'enhanced_metrics': enhanced_metrics,
                'progressive_enhancement_score': 100 if basic_content_available else 0
            }
            
        finally:
            driver.quit()
    
    def analyze_performance_trends(self, results: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze performance trends across multiple runs"""
        if not results:
            return {}
        
        metrics_arrays = {
            'page_load_time': [r.page_load_time for r in results],
            'dom_content_loaded': [r.dom_content_loaded for r in results],
            'first_contentful_paint': [r.first_contentful_paint for r in results],
            'largest_contentful_paint': [r.largest_contentful_paint for r in results],
            'cumulative_layout_shift': [r.cumulative_layout_shift for r in results],
            'network_requests': [r.network_requests for r in results]
        }
        
        analysis = {}
        for metric, values in metrics_arrays.items():
            if values:
                analysis[metric] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p95': np.percentile(values, 95),
                    'threshold_met': np.mean(values) <= self.thresholds.get(metric, float('inf'))
                }
        
        return analysis
    
    async def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        results = {
            'timestamp': time.time(),
            'test_suite': 'page_load_performance',
            'tests': {}
        }
        
        # Test Chrome desktop
        chrome_desktop = await self.test_initial_page_load('chrome', mobile=False)
        results['tests']['chrome_desktop'] = chrome_desktop
        
        # Test Chrome mobile
        chrome_mobile = await self.test_initial_page_load('chrome', mobile=True)
        results['tests']['chrome_mobile'] = chrome_mobile
        
        # Test Firefox desktop
        firefox_desktop = await self.test_initial_page_load('firefox', mobile=False)
        results['tests']['firefox_desktop'] = firefox_desktop
        
        # Test asset loading
        asset_performance = await self.test_asset_loading_performance()
        results['tests']['asset_loading'] = asset_performance
        
        # Test progressive enhancement
        progressive_enhancement = await self.test_progressive_enhancement()
        results['tests']['progressive_enhancement'] = progressive_enhancement
        
        # Generate recommendations
        results['recommendations'] = self._generate_performance_recommendations(results)
        
        return results
    
    def _generate_performance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        for test_name, test_result in results['tests'].items():
            if hasattr(test_result, 'page_load_time'):
                if test_result.page_load_time > self.thresholds['page_load_time']:
                    recommendations.append(f"Optimize page load time for {test_name} (current: {test_result.page_load_time:.2f}s)")
                
                if test_result.largest_contentful_paint > self.thresholds['largest_contentful_paint']:
                    recommendations.append(f"Improve LCP for {test_name} - consider image optimization and preloading")
                
                if test_result.cumulative_layout_shift > self.thresholds['cumulative_layout_shift']:
                    recommendations.append(f"Reduce layout shifts for {test_name} - add explicit dimensions to images")
        
        # Check asset loading efficiency
        asset_test = results['tests'].get('asset_loading', {})
        if isinstance(asset_test, dict) and 'cache_effectiveness' in asset_test:
            cache_improvement = asset_test['cache_effectiveness']['page_load_improvement']
            if cache_improvement < 30:  # Less than 30% improvement from caching
                recommendations.append("Improve caching strategy - consider longer cache headers for static assets")
        
        return recommendations


# Pytest fixtures and test cases
@pytest.fixture
def performance_tester():
    """Fixture for performance tester"""
    return PageLoadPerformanceTester()


@pytest.mark.asyncio
async def test_page_load_performance_chrome(performance_tester):
    """Test page load performance in Chrome"""
    metrics = await performance_tester.test_initial_page_load('chrome')
    
    assert metrics.page_load_time < performance_tester.thresholds['page_load_time'], \
        f"Page load time {metrics.page_load_time}s exceeds threshold {performance_tester.thresholds['page_load_time']}s"
    
    assert metrics.first_contentful_paint < performance_tester.thresholds['first_contentful_paint'], \
        f"FCP {metrics.first_contentful_paint}s exceeds threshold"


@pytest.mark.asyncio
async def test_mobile_performance(performance_tester):
    """Test mobile performance"""
    metrics = await performance_tester.test_initial_page_load('chrome', mobile=True)
    
    # Mobile should have reasonable performance
    assert metrics.page_load_time < 3.0, "Mobile page load time too slow"
    assert metrics.network_requests < 50, "Too many network requests for mobile"


@pytest.mark.asyncio
async def test_asset_caching_effectiveness(performance_tester):
    """Test asset caching effectiveness"""
    results = await performance_tester.test_asset_loading_performance()
    
    cache_improvement = results['cache_effectiveness']['page_load_improvement']
    assert cache_improvement > 20, f"Cache improvement {cache_improvement}% is below expected threshold"


@pytest.mark.asyncio
async def test_progressive_enhancement(performance_tester):
    """Test progressive enhancement"""
    results = await performance_tester.test_progressive_enhancement()
    
    assert results['basic_content_available'], "Basic content should be available without JavaScript"
    assert results['progressive_enhancement_score'] >= 50, "Progressive enhancement score too low"


if __name__ == "__main__":
    # Run comprehensive performance test
    async def main():
        tester = PageLoadPerformanceTester()
        results = await tester.run_comprehensive_performance_test()
        
        # Save results
        output_path = Path("tests/ui/performance/results")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"page_load_performance_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("Performance test completed. Results saved.")
        print(f"Recommendations: {results['recommendations']}")
    
    asyncio.run(main())