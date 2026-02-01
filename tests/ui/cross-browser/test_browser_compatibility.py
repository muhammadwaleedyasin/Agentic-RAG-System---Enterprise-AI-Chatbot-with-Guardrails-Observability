"""
Cross-Browser Compatibility Testing Suite for RAG Chatbot UI

Tests compatibility across Chrome, Firefox, Safari, Edge, and mobile browsers.
Validates functionality, performance, and visual consistency across different browser engines.
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, WebDriverException
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import io
import base64


@dataclass
class BrowserTestResult:
    """Browser test result container"""
    browser: str
    version: str
    platform: str
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    screenshot_path: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    feature_support: Optional[Dict[str, bool]] = None


@dataclass
class CompatibilityReport:
    """Comprehensive compatibility report"""
    timestamp: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    browser_results: Dict[str, List[BrowserTestResult]]
    feature_matrix: Dict[str, Dict[str, bool]]
    performance_comparison: Dict[str, Dict[str, float]]
    recommendations: List[str]


class CrossBrowserTester:
    """Cross-browser compatibility testing framework"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.browsers_config = {
            'chrome': {
                'versions': ['latest', 'latest-1'],
                'mobile': True
            },
            'firefox': {
                'versions': ['latest', 'esr'],
                'mobile': False
            },
            'edge': {
                'versions': ['latest'],
                'mobile': False
            },
            'safari': {
                'versions': ['latest'],
                'mobile': True
            }
        }
        
        self.core_features = [
            'page_load',
            'chat_interface',
            'document_upload',
            'search_functionality',
            'websocket_connection',
            'responsive_design',
            'accessibility_features',
            'local_storage',
            'session_storage',
            'modern_js_features'
        ]
        
        self.test_scenarios = [
            {
                'name': 'basic_functionality',
                'description': 'Test basic chat and navigation',
                'critical': True
            },
            {
                'name': 'document_processing',
                'description': 'Test document upload and processing',
                'critical': True
            },
            {
                'name': 'real_time_features',
                'description': 'Test WebSocket and real-time updates',
                'critical': False
            },
            {
                'name': 'responsive_behavior',
                'description': 'Test responsive design adaptation',
                'critical': True
            },
            {
                'name': 'performance_benchmarks',
                'description': 'Test performance across browsers',
                'critical': False
            }
        ]
    
    def _get_driver(self, browser: str, mobile: bool = False, version: str = 'latest') -> webdriver.Remote:
        """Get configured WebDriver for specified browser"""
        
        if browser == 'chrome':
            return self._get_chrome_driver(mobile, version)
        elif browser == 'firefox':
            return self._get_firefox_driver(version)
        elif browser == 'edge':
            return self._get_edge_driver(version)
        elif browser == 'safari':
            return self._get_safari_driver(mobile)
        else:
            raise ValueError(f"Unsupported browser: {browser}")
    
    def _get_chrome_driver(self, mobile: bool = False, version: str = 'latest') -> webdriver.Chrome:
        """Configure Chrome driver"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        
        if mobile:
            # iPhone 12 Pro simulation
            mobile_emulation = {
                "deviceMetrics": {"width": 390, "height": 844, "pixelRatio": 3.0},
                "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15"
            }
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        # Enable performance and network logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': True
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    def _get_firefox_driver(self, version: str = 'latest') -> webdriver.Firefox:
        """Configure Firefox driver"""
        options = FirefoxOptions()
        options.add_argument('--headless')
        
        # Firefox profile configuration
        profile = webdriver.FirefoxProfile()
        profile.set_preference('dom.webnotifications.enabled', False)
        profile.set_preference('dom.push.enabled', False)
        profile.set_preference('dom.disable_beforeunload', True)
        
        return webdriver.Firefox(options=options, firefox_profile=profile)
    
    def _get_edge_driver(self, version: str = 'latest') -> webdriver.Edge:
        """Configure Edge driver"""
        options = EdgeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        return webdriver.Edge(options=options)
    
    def _get_safari_driver(self, mobile: bool = False) -> webdriver.Safari:
        """Configure Safari driver (macOS only)"""
        try:
            # Safari driver configuration is limited
            return webdriver.Safari()
        except Exception:
            # Return None if Safari is not available
            return None
    
    async def test_browser_compatibility(self, browser: str, mobile: bool = False) -> List[BrowserTestResult]:
        """Test compatibility for a specific browser"""
        results = []
        driver = None
        
        try:
            driver = self._get_driver(browser, mobile)
            if driver is None:
                return [BrowserTestResult(
                    browser=browser,
                    version='unknown',
                    platform='mobile' if mobile else 'desktop',
                    test_name='driver_initialization',
                    success=False,
                    duration=0,
                    error_message=f"{browser} driver not available"
                )]
            
            # Get browser info
            browser_info = self._get_browser_info(driver)
            
            # Run test scenarios
            for scenario in self.test_scenarios:
                start_time = time.time()
                
                try:
                    success = await self._run_test_scenario(driver, scenario['name'])
                    duration = time.time() - start_time
                    
                    # Capture screenshot on failure
                    screenshot_path = None
                    if not success:
                        screenshot_path = self._capture_screenshot(driver, browser, scenario['name'])
                    
                    # Get performance metrics
                    perf_metrics = self._get_performance_metrics(driver)
                    
                    result = BrowserTestResult(
                        browser=browser,
                        version=browser_info['version'],
                        platform='mobile' if mobile else 'desktop',
                        test_name=scenario['name'],
                        success=success,
                        duration=duration,
                        screenshot_path=screenshot_path,
                        performance_metrics=perf_metrics
                    )
                    
                except Exception as e:
                    duration = time.time() - start_time
                    result = BrowserTestResult(
                        browser=browser,
                        version=browser_info['version'],
                        platform='mobile' if mobile else 'desktop',
                        test_name=scenario['name'],
                        success=False,
                        duration=duration,
                        error_message=str(e)
                    )
                
                results.append(result)
        
        finally:
            if driver:
                driver.quit()
        
        return results
    
    def _get_browser_info(self, driver: webdriver.Remote) -> Dict[str, str]:
        """Get browser information"""
        try:
            caps = driver.capabilities
            browser_name = caps.get('browserName', 'unknown')
            browser_version = caps.get('browserVersion', caps.get('version', 'unknown'))
            platform = caps.get('platformName', caps.get('platform', 'unknown'))
            
            return {
                'name': browser_name,
                'version': browser_version,
                'platform': platform
            }
        except Exception:
            return {'name': 'unknown', 'version': 'unknown', 'platform': 'unknown'}
    
    async def _run_test_scenario(self, driver: webdriver.Remote, scenario_name: str) -> bool:
        """Run a specific test scenario"""
        try:
            if scenario_name == 'basic_functionality':
                return await self._test_basic_functionality(driver)
            elif scenario_name == 'document_processing':
                return await self._test_document_processing(driver)
            elif scenario_name == 'real_time_features':
                return await self._test_real_time_features(driver)
            elif scenario_name == 'responsive_behavior':
                return await self._test_responsive_behavior(driver)
            elif scenario_name == 'performance_benchmarks':
                return await self._test_performance_benchmarks(driver)
            else:
                return False
        except Exception:
            return False
    
    async def _test_basic_functionality(self, driver: webdriver.Remote) -> bool:
        """Test basic functionality"""
        # Navigate to application
        driver.get(self.base_url)
        
        # Wait for page load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Test chat interface
        try:
            chat_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
            )
            
            chat_input.send_keys("Test message")
            chat_input.send_keys(Keys.RETURN)
            
            # Wait for response or indication of processing
            await asyncio.sleep(2)
            
            return True
        except TimeoutException:
            return False
    
    async def _test_document_processing(self, driver: webdriver.Remote) -> bool:
        """Test document upload and processing"""
        try:
            # Navigate to upload page or find upload element
            driver.get(f"{self.base_url}/upload")
            
            # Look for file upload element
            file_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # Create a test file
            test_file_path = self._create_test_document()
            file_input.send_keys(test_file_path)
            
            # Wait for upload indication
            await asyncio.sleep(3)
            
            # Clean up
            Path(test_file_path).unlink(missing_ok=True)
            
            return True
        except (TimeoutException, Exception):
            return False
    
    async def _test_real_time_features(self, driver: webdriver.Remote) -> bool:
        """Test WebSocket and real-time features"""
        try:
            driver.get(self.base_url)
            
            # Test WebSocket connection
            websocket_test = driver.execute_script("""
                return new Promise((resolve) => {
                    try {
                        const ws = new WebSocket('ws://localhost:8000/ws');
                        
                        ws.onopen = () => {
                            ws.close();
                            resolve(true);
                        };
                        
                        ws.onerror = () => {
                            resolve(false);
                        };
                        
                        setTimeout(() => resolve(false), 5000);
                    } catch (e) {
                        resolve(false);
                    }
                });
            """)
            
            return websocket_test
        except Exception:
            return False
    
    async def _test_responsive_behavior(self, driver: webdriver.Remote) -> bool:
        """Test responsive design"""
        try:
            driver.get(self.base_url)
            
            # Test different viewport sizes
            viewports = [
                (320, 568),   # iPhone SE
                (768, 1024),  # iPad
                (1920, 1080)  # Desktop
            ]
            
            for width, height in viewports:
                driver.set_window_size(width, height)
                await asyncio.sleep(1)
                
                # Check if content is accessible
                body = driver.find_element(By.TAG_NAME, "body")
                if not body.is_displayed():
                    return False
            
            return True
        except Exception:
            return False
    
    async def _test_performance_benchmarks(self, driver: webdriver.Remote) -> bool:
        """Test performance benchmarks"""
        try:
            start_time = time.time()
            driver.get(self.base_url)
            
            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            load_time = time.time() - start_time
            
            # Performance should be reasonable (under 5 seconds)
            return load_time < 5.0
        except Exception:
            return False
    
    def _create_test_document(self) -> str:
        """Create a test document for upload testing"""
        test_dir = Path("tests/ui/cross-browser/temp")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / "test_document.txt"
        test_file.write_text("This is a test document for upload testing.")
        
        return str(test_file.absolute())
    
    def _get_performance_metrics(self, driver: webdriver.Remote) -> Dict[str, float]:
        """Get performance metrics from browser"""
        try:
            return driver.execute_script("""
                const timing = performance.timing;
                const navigation = performance.getEntriesByType('navigation')[0];
                
                return {
                    page_load_time: (timing.loadEventEnd - timing.navigationStart) / 1000,
                    dom_content_loaded: (timing.domContentLoadedEventEnd - timing.navigationStart) / 1000,
                    first_byte: (timing.responseStart - timing.navigationStart) / 1000,
                    transfer_size: navigation ? navigation.transferSize : 0
                };
            """)
        except Exception:
            return {}
    
    def _capture_screenshot(self, driver: webdriver.Remote, browser: str, test_name: str) -> str:
        """Capture screenshot on test failure"""
        try:
            screenshot_dir = Path("tests/ui/cross-browser/screenshots")
            screenshot_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"{browser}_{test_name}_{timestamp}.png"
            filepath = screenshot_dir / filename
            
            driver.save_screenshot(str(filepath))
            return str(filepath)
        except Exception:
            return None
    
    def _test_feature_support(self, driver: webdriver.Remote) -> Dict[str, bool]:
        """Test browser feature support"""
        feature_tests = {
            'websockets': "return 'WebSocket' in window;",
            'local_storage': "return 'localStorage' in window;",
            'session_storage': "return 'sessionStorage' in window;",
            'geolocation': "return 'geolocation' in navigator;",
            'notifications': "return 'Notification' in window;",
            'service_workers': "return 'serviceWorker' in navigator;",
            'webgl': """
                try {
                    const canvas = document.createElement('canvas');
                    return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
                } catch (e) {
                    return false;
                }
            """,
            'web_assembly': "return 'WebAssembly' in window;",
            'fetch_api': "return 'fetch' in window;",
            'promises': "return 'Promise' in window;",
            'async_await': """
                try {
                    eval('(async function() {})');
                    return true;
                } catch (e) {
                    return false;
                }
            """,
            'css_grid': """
                return CSS.supports('display', 'grid');
            """,
            'css_flexbox': """
                return CSS.supports('display', 'flex');
            """
        }
        
        results = {}
        for feature, test_script in feature_tests.items():
            try:
                results[feature] = driver.execute_script(test_script)
            except Exception:
                results[feature] = False
        
        return results
    
    async def run_comprehensive_compatibility_test(self) -> CompatibilityReport:
        """Run comprehensive cross-browser compatibility test"""
        all_results = []
        browser_results = {}
        feature_matrix = {}
        performance_comparison = {}
        
        # Test each browser configuration
        for browser, config in self.browsers_config.items():
            print(f"Testing {browser}...")
            browser_results[browser] = []
            
            # Test desktop version
            try:
                desktop_results = await self.test_browser_compatibility(browser, mobile=False)
                browser_results[browser].extend(desktop_results)
                all_results.extend(desktop_results)
                
                # Test feature support
                if desktop_results:
                    driver = self._get_driver(browser)
                    if driver:
                        try:
                            driver.get(self.base_url)
                            feature_matrix[f"{browser}_desktop"] = self._test_feature_support(driver)
                            
                            # Collect performance metrics
                            perf_metrics = self._get_performance_metrics(driver)
                            if perf_metrics:
                                performance_comparison[f"{browser}_desktop"] = perf_metrics
                        finally:
                            driver.quit()
            except Exception as e:
                print(f"Error testing {browser} desktop: {e}")
            
            # Test mobile version if supported
            if config.get('mobile', False):
                try:
                    mobile_results = await self.test_browser_compatibility(browser, mobile=True)
                    browser_results[browser].extend(mobile_results)
                    all_results.extend(mobile_results)
                    
                    # Test mobile feature support
                    if mobile_results:
                        driver = self._get_driver(browser, mobile=True)
                        if driver:
                            try:
                                driver.get(self.base_url)
                                feature_matrix[f"{browser}_mobile"] = self._test_feature_support(driver)
                                
                                # Collect mobile performance metrics
                                perf_metrics = self._get_performance_metrics(driver)
                                if perf_metrics:
                                    performance_comparison[f"{browser}_mobile"] = perf_metrics
                            finally:
                                driver.quit()
                except Exception as e:
                    print(f"Error testing {browser} mobile: {e}")
        
        # Generate compatibility report
        total_tests = len(all_results)
        passed_tests = sum(1 for result in all_results if result.success)
        failed_tests = total_tests - passed_tests
        
        # Generate recommendations
        recommendations = self._generate_compatibility_recommendations(
            browser_results, feature_matrix, performance_comparison
        )
        
        return CompatibilityReport(
            timestamp=time.time(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            browser_results=browser_results,
            feature_matrix=feature_matrix,
            performance_comparison=performance_comparison,
            recommendations=recommendations
        )
    
    def _generate_compatibility_recommendations(
        self,
        browser_results: Dict[str, List[BrowserTestResult]],
        feature_matrix: Dict[str, Dict[str, bool]],
        performance_comparison: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate compatibility recommendations"""
        recommendations = []
        
        # Check for browser-specific failures
        for browser, results in browser_results.items():
            failed_tests = [r for r in results if not r.success]
            if failed_tests:
                critical_failures = [r for r in failed_tests if r.test_name in ['basic_functionality', 'document_processing']]
                if critical_failures:
                    recommendations.append(f"Critical functionality issues found in {browser}")
        
        # Check feature support gaps
        if feature_matrix:
            all_features = set()
            for browser_features in feature_matrix.values():
                all_features.update(browser_features.keys())
            
            for feature in all_features:
                support_count = sum(1 for browser_features in feature_matrix.values() 
                                  if browser_features.get(feature, False))
                total_browsers = len(feature_matrix)
                
                if support_count < total_browsers * 0.8:  # Less than 80% support
                    recommendations.append(f"Consider polyfill for {feature} (limited browser support)")
        
        # Check performance disparities
        if performance_comparison:
            load_times = {browser: metrics.get('page_load_time', 0) 
                         for browser, metrics in performance_comparison.items()}
            
            if load_times:
                max_load_time = max(load_times.values())
                min_load_time = min(load_times.values())
                
                if max_load_time > min_load_time * 2:  # 2x difference
                    recommendations.append("Significant performance differences detected across browsers")
        
        # General recommendations
        if not recommendations:
            recommendations.append("All browsers show good compatibility")
        else:
            recommendations.append("Consider implementing progressive enhancement for better compatibility")
            recommendations.append("Add feature detection for unsupported functionality")
        
        return recommendations


# Pytest fixtures and test cases
@pytest.fixture
def compatibility_tester():
    """Fixture for cross-browser compatibility tester"""
    return CrossBrowserTester()


@pytest.mark.asyncio
async def test_chrome_compatibility(compatibility_tester):
    """Test Chrome compatibility"""
    results = await compatibility_tester.test_browser_compatibility('chrome')
    
    # Check that basic functionality works
    basic_test = next((r for r in results if r.test_name == 'basic_functionality'), None)
    assert basic_test is not None, "Basic functionality test not found"
    assert basic_test.success, f"Basic functionality failed in Chrome: {basic_test.error_message}"


@pytest.mark.asyncio
async def test_firefox_compatibility(compatibility_tester):
    """Test Firefox compatibility"""
    results = await compatibility_tester.test_browser_compatibility('firefox')
    
    # Check that basic functionality works
    basic_test = next((r for r in results if r.test_name == 'basic_functionality'), None)
    assert basic_test is not None, "Basic functionality test not found"
    assert basic_test.success, f"Basic functionality failed in Firefox: {basic_test.error_message}"


@pytest.mark.asyncio
async def test_mobile_compatibility(compatibility_tester):
    """Test mobile browser compatibility"""
    results = await compatibility_tester.test_browser_compatibility('chrome', mobile=True)
    
    # Check responsive behavior
    responsive_test = next((r for r in results if r.test_name == 'responsive_behavior'), None)
    assert responsive_test is not None, "Responsive behavior test not found"
    assert responsive_test.success, f"Responsive behavior failed: {responsive_test.error_message}"


if __name__ == "__main__":
    # Run comprehensive compatibility test
    async def main():
        tester = CrossBrowserTester()
        report = await tester.run_comprehensive_compatibility_test()
        
        # Save report
        output_path = Path("tests/ui/cross-browser/reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"compatibility_report_{int(time.time())}.json", 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        print(f"Compatibility test completed.")
        print(f"Total tests: {report.total_tests}")
        print(f"Passed: {report.passed_tests}")
        print(f"Failed: {report.failed_tests}")
        print(f"Success rate: {(report.passed_tests / report.total_tests * 100):.1f}%")
        
        if report.recommendations:
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")
    
    asyncio.run(main())