"""
API Integration Testing Suite for RAG Chatbot UI

Tests API integration through the UI layer, validates error handling,
response processing, and data flow between frontend and FastAPI backend.
"""

import asyncio
import time
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, WebDriverException
import pytest
import numpy as np
from pathlib import Path
import uuid


@dataclass
class APITestResult:
    """API integration test result"""
    test_name: str
    endpoint: str
    method: str
    success: bool
    response_time: float
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    ui_response: Optional[str] = None
    data_validation: Optional[Dict[str, bool]] = None


class APIIntegrationTester:
    """API integration testing through UI layer"""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_base: str = "http://localhost:8000/api"):
        self.base_url = base_url
        self.api_base = api_base
        self.test_session = requests.Session()
        
        # API endpoints to test
        self.endpoints = {
            'health': {
                'path': '/health',
                'method': 'GET',
                'ui_trigger': None,
                'expected_status': 200
            },
            'chat': {
                'path': '/chat',
                'method': 'POST',
                'ui_trigger': 'chat_input',
                'expected_status': 200
            },
            'upload_document': {
                'path': '/documents/upload',
                'method': 'POST',
                'ui_trigger': 'file_upload',
                'expected_status': 200
            },
            'search_documents': {
                'path': '/documents/search',
                'method': 'POST',
                'ui_trigger': 'search_input',
                'expected_status': 200
            },
            'get_documents': {
                'path': '/documents',
                'method': 'GET',
                'ui_trigger': 'document_list',
                'expected_status': 200
            },
            'rag_query': {
                'path': '/rag/query',
                'method': 'POST',
                'ui_trigger': 'rag_query',
                'expected_status': 200
            },
            'conversation_history': {
                'path': '/conversations',
                'method': 'GET',
                'ui_trigger': 'conversation_list',
                'expected_status': 200
            }
        }
        
        # Test data for various scenarios
        self.test_data = {
            'simple_query': "What is the deployment process?",
            'complex_query': "Compare local LLM inference with OpenRouter API considering latency, cost, and privacy factors.",
            'invalid_query': "",
            'large_query': "x" * 5000,  # Very long query
            'special_chars_query': "Test with special chars: @#$%^&*()[]{}|\\:;\"'<>?,./`~",
            'multilingual_query': "Comment configurer le système? Wie konfiguriert man das System? システムの設定方法は？"
        }
    
    def _get_chrome_driver(self) -> webdriver.Chrome:
        """Configure Chrome driver for API testing"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Enable network logging to monitor API calls
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False,
            'enableTimeline': False
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    def _monitor_network_requests(self, driver: webdriver.Chrome, duration: float = 5.0) -> List[Dict[str, Any]]:
        """Monitor network requests during UI interaction"""
        # Wait for the specified duration to capture requests
        time.sleep(duration)
        
        # Get performance logs
        logs = driver.get_log('performance')
        network_requests = []
        
        for log in logs:
            message = json.loads(log['message'])
            
            if message['message']['method'] in ['Network.responseReceived', 'Network.requestWillBeSent']:
                try:
                    request_data = message['message']['params']
                    
                    if message['message']['method'] == 'Network.requestWillBeSent':
                        request_info = {
                            'type': 'request',
                            'url': request_data['request']['url'],
                            'method': request_data['request']['method'],
                            'timestamp': request_data['timestamp'],
                            'request_id': request_data['requestId']
                        }
                        
                        if 'postData' in request_data['request']:
                            request_info['post_data'] = request_data['request']['postData']
                        
                        network_requests.append(request_info)
                    
                    elif message['message']['method'] == 'Network.responseReceived':
                        response_info = {
                            'type': 'response',
                            'url': request_data['response']['url'],
                            'status': request_data['response']['status'],
                            'timestamp': request_data['timestamp'],
                            'request_id': request_data['requestId']
                        }
                        
                        network_requests.append(response_info)
                        
                except Exception:
                    continue
        
        return network_requests
    
    async def test_direct_api_endpoint(self, endpoint_name: str) -> APITestResult:
        """Test API endpoint directly"""
        endpoint_config = self.endpoints[endpoint_name]
        url = f"{self.api_base}{endpoint_config['path']}"
        
        start_time = time.time()
        
        try:
            if endpoint_config['method'] == 'GET':
                response = self.test_session.get(url, timeout=30)
            elif endpoint_config['method'] == 'POST':
                # Prepare test data based on endpoint
                test_data = self._get_test_data_for_endpoint(endpoint_name)
                response = self.test_session.post(url, json=test_data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {endpoint_config['method']}")
            
            response_time = time.time() - start_time
            
            # Validate response
            data_validation = self._validate_response_data(endpoint_name, response)
            
            return APITestResult(
                test_name=f"direct_api_{endpoint_name}",
                endpoint=url,
                method=endpoint_config['method'],
                success=response.status_code == endpoint_config['expected_status'],
                response_time=response_time,
                status_code=response.status_code,
                data_validation=data_validation
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return APITestResult(
                test_name=f"direct_api_{endpoint_name}",
                endpoint=url,
                method=endpoint_config['method'],
                success=False,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def test_api_through_ui(self, endpoint_name: str, driver: webdriver.Chrome) -> APITestResult:
        """Test API endpoint through UI interaction"""
        endpoint_config = self.endpoints[endpoint_name]
        
        if not endpoint_config['ui_trigger']:
            return APITestResult(
                test_name=f"ui_api_{endpoint_name}",
                endpoint=f"{self.api_base}{endpoint_config['path']}",
                method=endpoint_config['method'],
                success=False,
                response_time=0,
                error_message="No UI trigger defined"
            )
        
        start_time = time.time()
        
        try:
            # Navigate to the application
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Clear performance logs before test
            driver.get_log('performance')
            
            # Trigger UI interaction
            success = await self._trigger_ui_interaction(driver, endpoint_config['ui_trigger'], endpoint_name)
            
            if not success:
                return APITestResult(
                    test_name=f"ui_api_{endpoint_name}",
                    endpoint=f"{self.api_base}{endpoint_config['path']}",
                    method=endpoint_config['method'],
                    success=False,
                    response_time=time.time() - start_time,
                    error_message="UI interaction failed"
                )
            
            # Monitor network requests
            network_requests = self._monitor_network_requests(driver, 5.0)
            
            # Find the relevant API call
            api_request = None
            api_response = None
            
            for request in network_requests:
                if endpoint_config['path'] in request['url']:
                    if request['type'] == 'request':
                        api_request = request
                    elif request['type'] == 'response':
                        api_response = request
            
            response_time = time.time() - start_time
            
            if api_response:
                success = api_response['status'] == endpoint_config['expected_status']
                status_code = api_response['status']
            else:
                success = False
                status_code = None
            
            # Get UI response/feedback
            ui_response = await self._get_ui_response(driver, endpoint_name)
            
            return APITestResult(
                test_name=f"ui_api_{endpoint_name}",
                endpoint=f"{self.api_base}{endpoint_config['path']}",
                method=endpoint_config['method'],
                success=success,
                response_time=response_time,
                status_code=status_code,
                ui_response=ui_response
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return APITestResult(
                test_name=f"ui_api_{endpoint_name}",
                endpoint=f"{self.api_base}{endpoint_config['path']}",
                method=endpoint_config['method'],
                success=False,
                response_time=response_time,
                error_message=str(e)
            )
    
    async def _trigger_ui_interaction(self, driver: webdriver.Chrome, trigger_type: str, endpoint_name: str) -> bool:
        """Trigger specific UI interaction"""
        try:
            if trigger_type == 'chat_input':
                return await self._trigger_chat_input(driver, endpoint_name)
            elif trigger_type == 'file_upload':
                return await self._trigger_file_upload(driver)
            elif trigger_type == 'search_input':
                return await self._trigger_search_input(driver)
            elif trigger_type == 'document_list':
                return await self._trigger_document_list(driver)
            elif trigger_type == 'rag_query':
                return await self._trigger_rag_query(driver)
            elif trigger_type == 'conversation_list':
                return await self._trigger_conversation_list(driver)
            else:
                return False
        except Exception:
            return False
    
    async def _trigger_chat_input(self, driver: webdriver.Chrome, endpoint_name: str) -> bool:
        """Trigger chat input interaction"""
        try:
            chat_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea, .chat-input"))
            )
            
            # Use different test data based on test scenario
            test_query = self.test_data['simple_query']
            if 'complex' in endpoint_name:
                test_query = self.test_data['complex_query']
            elif 'invalid' in endpoint_name:
                test_query = self.test_data['invalid_query']
            
            chat_input.clear()
            chat_input.send_keys(test_query)
            chat_input.send_keys(Keys.RETURN)
            
            return True
        except Exception:
            return False
    
    async def _trigger_file_upload(self, driver: webdriver.Chrome) -> bool:
        """Trigger file upload interaction"""
        try:
            # Create a test file
            test_file_path = self._create_test_file()
            
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            file_input.send_keys(test_file_path)
            
            # Clean up
            Path(test_file_path).unlink(missing_ok=True)
            
            return True
        except Exception:
            return False
    
    async def _trigger_search_input(self, driver: webdriver.Chrome) -> bool:
        """Trigger search input interaction"""
        try:
            search_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='search'], .search-input"))
            )
            
            search_input.clear()
            search_input.send_keys("deployment")
            search_input.send_keys(Keys.RETURN)
            
            return True
        except Exception:
            return False
    
    async def _trigger_document_list(self, driver: webdriver.Chrome) -> bool:
        """Trigger document list loading"""
        try:
            # Navigate to documents page or trigger document list load
            driver.get(f"{self.base_url}/documents")
            await asyncio.sleep(2)
            return True
        except Exception:
            return False
    
    async def _trigger_rag_query(self, driver: webdriver.Chrome) -> bool:
        """Trigger RAG query interaction"""
        try:
            # Similar to chat input but specifically for RAG queries
            rag_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".rag-input, input[type='text'], textarea"))
            )
            
            rag_input.clear()
            rag_input.send_keys(self.test_data['simple_query'])
            rag_input.send_keys(Keys.RETURN)
            
            return True
        except Exception:
            return False
    
    async def _trigger_conversation_list(self, driver: webdriver.Chrome) -> bool:
        """Trigger conversation list loading"""
        try:
            # Navigate to conversations page or trigger conversation list load
            conversations_link = driver.find_element(By.CSS_SELECTOR, "a[href*='conversation'], .conversations-link")
            conversations_link.click()
            await asyncio.sleep(2)
            return True
        except Exception:
            return False
    
    async def _get_ui_response(self, driver: webdriver.Chrome, endpoint_name: str) -> Optional[str]:
        """Get UI response/feedback after API call"""
        try:
            # Wait for response elements to appear
            await asyncio.sleep(3)
            
            # Look for various response indicators
            response_selectors = [
                ".response",
                ".message",
                ".chat-response",
                ".result",
                ".feedback",
                ".status",
                ".notification"
            ]
            
            for selector in response_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        return elements[-1].text  # Get the latest response
                except Exception:
                    continue
            
            return None
        except Exception:
            return None
    
    def _create_test_file(self) -> str:
        """Create a test file for upload"""
        test_dir = Path("tests/ui/api-integration/temp")
        test_dir.mkdir(exist_ok=True)
        
        test_file = test_dir / f"test_document_{uuid.uuid4().hex[:8]}.txt"
        test_file.write_text("This is a test document for API integration testing.")
        
        return str(test_file.absolute())
    
    def _get_test_data_for_endpoint(self, endpoint_name: str) -> Dict[str, Any]:
        """Get appropriate test data for specific endpoint"""
        if endpoint_name == 'chat':
            return {
                "message": self.test_data['simple_query'],
                "conversation_id": None
            }
        elif endpoint_name == 'search_documents':
            return {
                "query": "deployment",
                "limit": 10
            }
        elif endpoint_name == 'rag_query':
            return {
                "query": self.test_data['simple_query'],
                "include_sources": True
            }
        elif endpoint_name == 'upload_document':
            return {
                "title": "Test Document",
                "content": "Test document content",
                "metadata": {
                    "app": "test",
                    "version": "1.0"
                }
            }
        else:
            return {}
    
    def _validate_response_data(self, endpoint_name: str, response: requests.Response) -> Dict[str, bool]:
        """Validate response data structure"""
        validation = {
            'status_code_valid': False,
            'content_type_valid': False,
            'data_structure_valid': False,
            'required_fields_present': False
        }
        
        # Validate status code
        expected_status = self.endpoints[endpoint_name]['expected_status']
        validation['status_code_valid'] = response.status_code == expected_status
        
        # Validate content type
        content_type = response.headers.get('content-type', '')
        validation['content_type_valid'] = 'application/json' in content_type
        
        if validation['content_type_valid']:
            try:
                data = response.json()
                validation['data_structure_valid'] = isinstance(data, dict)
                
                # Check required fields based on endpoint
                if endpoint_name == 'health':
                    validation['required_fields_present'] = 'status' in data
                elif endpoint_name == 'chat':
                    validation['required_fields_present'] = 'response' in data
                elif endpoint_name == 'search_documents':
                    validation['required_fields_present'] = 'results' in data
                elif endpoint_name == 'rag_query':
                    validation['required_fields_present'] = 'answer' in data and 'sources' in data
                else:
                    validation['required_fields_present'] = True
                    
            except json.JSONDecodeError:
                validation['data_structure_valid'] = False
        
        return validation
    
    async def test_error_handling(self, driver: webdriver.Chrome) -> Dict[str, APITestResult]:
        """Test error handling scenarios"""
        error_tests = {}
        
        # Test invalid input handling
        try:
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Test empty query
            chat_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
            )
            
            start_time = time.time()
            chat_input.clear()
            chat_input.send_keys("")
            chat_input.send_keys(Keys.RETURN)
            
            await asyncio.sleep(3)
            
            # Check for error message
            error_message = await self._get_ui_response(driver, 'error_handling')
            
            error_tests['empty_query'] = APITestResult(
                test_name="error_handling_empty_query",
                endpoint="/api/chat",
                method="POST",
                success=bool(error_message and ("error" in error_message.lower() or "required" in error_message.lower())),
                response_time=time.time() - start_time,
                ui_response=error_message
            )
            
        except Exception as e:
            error_tests['empty_query'] = APITestResult(
                test_name="error_handling_empty_query",
                endpoint="/api/chat",
                method="POST",
                success=False,
                response_time=0,
                error_message=str(e)
            )
        
        # Test oversized input
        try:
            start_time = time.time()
            chat_input.clear()
            chat_input.send_keys(self.test_data['large_query'])
            chat_input.send_keys(Keys.RETURN)
            
            await asyncio.sleep(3)
            
            error_message = await self._get_ui_response(driver, 'error_handling')
            
            error_tests['large_query'] = APITestResult(
                test_name="error_handling_large_query",
                endpoint="/api/chat",
                method="POST",
                success=True,  # Should handle gracefully, not necessarily error
                response_time=time.time() - start_time,
                ui_response=error_message
            )
            
        except Exception as e:
            error_tests['large_query'] = APITestResult(
                test_name="error_handling_large_query",
                endpoint="/api/chat",
                method="POST",
                success=False,
                response_time=0,
                error_message=str(e)
            )
        
        return error_tests
    
    async def run_comprehensive_api_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive API integration test suite"""
        driver = self._get_chrome_driver()
        
        try:
            results = {
                'timestamp': time.time(),
                'test_suite': 'api_integration',
                'direct_api_tests': {},
                'ui_api_tests': {},
                'error_handling_tests': {},
                'performance_summary': {},
                'recommendations': []
            }
            
            # Test direct API endpoints
            print("Testing direct API endpoints...")
            for endpoint_name in self.endpoints.keys():
                if endpoint_name != 'upload_document':  # Skip file upload for direct API test
                    result = await self.test_direct_api_endpoint(endpoint_name)
                    results['direct_api_tests'][endpoint_name] = asdict(result)
            
            # Test API through UI
            print("Testing API through UI interactions...")
            for endpoint_name in self.endpoints.keys():
                if self.endpoints[endpoint_name]['ui_trigger']:
                    result = await self.test_api_through_ui(endpoint_name, driver)
                    results['ui_api_tests'][endpoint_name] = asdict(result)
            
            # Test error handling
            print("Testing error handling scenarios...")
            error_results = await self.test_error_handling(driver)
            results['error_handling_tests'] = {name: asdict(result) for name, result in error_results.items()}
            
            # Generate performance summary
            results['performance_summary'] = self._generate_performance_summary(results)
            
            # Generate recommendations
            results['recommendations'] = self._generate_api_recommendations(results)
            
            return results
            
        finally:
            driver.quit()
    
    def _generate_performance_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary from test results"""
        summary = {
            'average_response_time': 0,
            'fastest_endpoint': None,
            'slowest_endpoint': None,
            'success_rate': 0,
            'ui_vs_direct_performance': {}
        }
        
        # Collect response times
        all_response_times = []
        endpoint_times = {}
        
        for test_type in ['direct_api_tests', 'ui_api_tests']:
            for endpoint, result in results[test_type].items():
                response_time = result['response_time']
                all_response_times.append(response_time)
                
                if endpoint not in endpoint_times:
                    endpoint_times[endpoint] = {}
                endpoint_times[endpoint][test_type] = response_time
        
        if all_response_times:
            summary['average_response_time'] = np.mean(all_response_times)
            
            # Find fastest and slowest
            min_time = min(all_response_times)
            max_time = max(all_response_times)
            
            for test_type in ['direct_api_tests', 'ui_api_tests']:
                for endpoint, result in results[test_type].items():
                    if result['response_time'] == min_time:
                        summary['fastest_endpoint'] = f"{endpoint} ({test_type})"
                    if result['response_time'] == max_time:
                        summary['slowest_endpoint'] = f"{endpoint} ({test_type})"
        
        # Calculate success rate
        total_tests = 0
        successful_tests = 0
        
        for test_type in ['direct_api_tests', 'ui_api_tests', 'error_handling_tests']:
            for result in results[test_type].values():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
        
        if total_tests > 0:
            summary['success_rate'] = (successful_tests / total_tests) * 100
        
        # Compare UI vs direct API performance
        for endpoint in endpoint_times:
            if 'direct_api_tests' in endpoint_times[endpoint] and 'ui_api_tests' in endpoint_times[endpoint]:
                direct_time = endpoint_times[endpoint]['direct_api_tests']
                ui_time = endpoint_times[endpoint]['ui_api_tests']
                overhead = ((ui_time - direct_time) / direct_time) * 100
                summary['ui_vs_direct_performance'][endpoint] = {
                    'direct_time': direct_time,
                    'ui_time': ui_time,
                    'overhead_percent': overhead
                }
        
        return summary
    
    def _generate_api_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate API integration recommendations"""
        recommendations = []
        
        # Check success rates
        perf_summary = results['performance_summary']
        success_rate = perf_summary.get('success_rate', 0)
        
        if success_rate < 90:
            recommendations.append(f"API success rate ({success_rate:.1f}%) below target - investigate failing endpoints")
        
        # Check response times
        avg_response_time = perf_summary.get('average_response_time', 0)
        if avg_response_time > 2.0:
            recommendations.append(f"Average response time ({avg_response_time:.2f}s) above 2s threshold")
        
        # Check UI overhead
        ui_performance = perf_summary.get('ui_vs_direct_performance', {})
        high_overhead_endpoints = []
        
        for endpoint, perf_data in ui_performance.items():
            overhead = perf_data.get('overhead_percent', 0)
            if overhead > 100:  # More than 100% overhead
                high_overhead_endpoints.append(endpoint)
        
        if high_overhead_endpoints:
            recommendations.append(f"High UI overhead detected for endpoints: {', '.join(high_overhead_endpoints)}")
        
        # Check error handling
        error_tests = results.get('error_handling_tests', {})
        failed_error_tests = [name for name, result in error_tests.items() if not result['success']]
        
        if failed_error_tests:
            recommendations.append(f"Error handling needs improvement for: {', '.join(failed_error_tests)}")
        
        # General recommendations
        if not recommendations:
            recommendations.append("API integration is performing well across all test scenarios")
        else:
            recommendations.append("Consider implementing request caching for frequently accessed endpoints")
            recommendations.append("Add request/response compression to improve performance")
            recommendations.append("Implement retry logic for failed requests")
        
        return recommendations


# Pytest fixtures and test cases
@pytest.fixture
def api_tester():
    """Fixture for API integration tester"""
    return APIIntegrationTester()


@pytest.mark.asyncio
async def test_health_endpoint_direct(api_tester):
    """Test health endpoint directly"""
    result = await api_tester.test_direct_api_endpoint('health')
    assert result.success, f"Health endpoint failed: {result.error_message}"
    assert result.status_code == 200, f"Health endpoint returned {result.status_code}"


@pytest.mark.asyncio
async def test_chat_endpoint_through_ui(api_tester):
    """Test chat endpoint through UI"""
    driver = api_tester._get_chrome_driver()
    try:
        result = await api_tester.test_api_through_ui('chat', driver)
        assert result.success, f"Chat through UI failed: {result.error_message}"
    finally:
        driver.quit()


@pytest.mark.asyncio
async def test_error_handling_scenarios(api_tester):
    """Test error handling scenarios"""
    driver = api_tester._get_chrome_driver()
    try:
        results = await api_tester.test_error_handling(driver)
        
        # At least one error handling test should succeed
        success_count = sum(1 for result in results.values() if result.success)
        assert success_count > 0, "No error handling tests succeeded"
    finally:
        driver.quit()


if __name__ == "__main__":
    # Run comprehensive API integration test
    async def main():
        tester = APIIntegrationTester()
        results = await tester.run_comprehensive_api_integration_test()
        
        # Save results
        output_path = Path("tests/ui/api-integration/reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"api_integration_report_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("API integration test completed.")
        print(f"Success rate: {results['performance_summary']['success_rate']:.1f}%")
        print(f"Average response time: {results['performance_summary']['average_response_time']:.2f}s")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())