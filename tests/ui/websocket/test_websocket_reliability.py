"""
WebSocket Reliability Testing Suite for RAG Chatbot UI

Tests WebSocket connection reliability, reconnection logic, message delivery,
and connection management under various network conditions.
"""

import asyncio
import time
import json
import websockets
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import socket


@dataclass
class WebSocketTestResult:
    """WebSocket test result container"""
    test_name: str
    success: bool
    duration: float
    messages_sent: int
    messages_received: int
    connection_time: float
    reconnection_attempts: int
    error_message: Optional[str] = None
    latency_stats: Optional[Dict[str, float]] = None
    connection_stability: Optional[float] = None


class WebSocketReliabilityTester:
    """WebSocket reliability and connection testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000", ws_url: str = "ws://localhost:8000/ws"):
        self.base_url = base_url
        self.ws_url = ws_url
        self.test_messages = [
            {"type": "ping", "data": "test_message_1"},
            {"type": "chat", "data": "Hello, this is a test message"},
            {"type": "query", "data": "What is the system status?"},
            {"type": "heartbeat", "data": "ping"},
            {"type": "complex", "data": {"nested": {"structure": "test", "array": [1, 2, 3]}}},
        ]
        
        self.connection_scenarios = [
            'normal_connection',
            'slow_network',
            'intermittent_connection',
            'connection_drops',
            'concurrent_connections',
            'message_flooding',
            'large_messages',
            'malformed_messages'
        ]
    
    def _get_chrome_driver(self) -> webdriver.Chrome:
        """Configure Chrome driver for WebSocket testing"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        # Enable WebSocket logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    async def test_basic_websocket_connection(self) -> WebSocketTestResult:
        """Test basic WebSocket connection and messaging"""
        start_time = time.time()
        messages_sent = 0
        messages_received = 0
        latencies = []
        
        try:
            # Connect to WebSocket
            connection_start = time.time()
            async with websockets.connect(self.ws_url) as websocket:
                connection_time = time.time() - connection_start
                
                # Send test messages and measure latency
                for message in self.test_messages:
                    message_start = time.time()
                    await websocket.send(json.dumps(message))
                    messages_sent += 1
                    
                    # Wait for response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        message_latency = time.time() - message_start
                        latencies.append(message_latency)
                        messages_received += 1
                    except asyncio.TimeoutError:
                        pass
                    
                    await asyncio.sleep(0.1)  # Small delay between messages
                
                duration = time.time() - start_time
                
                latency_stats = None
                if latencies:
                    latency_stats = {
                        'mean': np.mean(latencies),
                        'median': np.median(latencies),
                        'min': np.min(latencies),
                        'max': np.max(latencies),
                        'std': np.std(latencies)
                    }
                
                return WebSocketTestResult(
                    test_name="basic_websocket_connection",
                    success=messages_sent > 0 and messages_received > 0,
                    duration=duration,
                    messages_sent=messages_sent,
                    messages_received=messages_received,
                    connection_time=connection_time,
                    reconnection_attempts=0,
                    latency_stats=latency_stats
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="basic_websocket_connection",
                success=False,
                duration=duration,
                messages_sent=messages_sent,
                messages_received=messages_received,
                connection_time=0,
                reconnection_attempts=0,
                error_message=str(e)
            )
    
    async def test_reconnection_logic(self) -> WebSocketTestResult:
        """Test WebSocket reconnection logic"""
        start_time = time.time()
        reconnection_attempts = 0
        total_messages_sent = 0
        total_messages_received = 0
        
        try:
            # Initial connection
            connection_start = time.time()
            websocket = await websockets.connect(self.ws_url)
            initial_connection_time = time.time() - connection_start
            
            # Send initial message
            await websocket.send(json.dumps({"type": "test", "data": "initial_message"}))
            total_messages_sent += 1
            
            # Receive response
            try:
                await asyncio.wait_for(websocket.recv(), timeout=2.0)
                total_messages_received += 1
            except asyncio.TimeoutError:
                pass
            
            # Simulate connection drops and reconnections
            for i in range(3):
                # Close connection to simulate drop
                await websocket.close()
                await asyncio.sleep(1)
                
                # Attempt reconnection
                reconnection_start = time.time()
                try:
                    websocket = await websockets.connect(self.ws_url)
                    reconnection_attempts += 1
                    
                    # Test message after reconnection
                    await websocket.send(json.dumps({"type": "test", "data": f"reconnection_message_{i}"}))
                    total_messages_sent += 1
                    
                    try:
                        await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        total_messages_received += 1
                    except asyncio.TimeoutError:
                        pass
                        
                except Exception:
                    break
            
            await websocket.close()
            duration = time.time() - start_time
            
            return WebSocketTestResult(
                test_name="reconnection_logic",
                success=reconnection_attempts >= 2,  # At least 2 successful reconnections
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=initial_connection_time,
                reconnection_attempts=reconnection_attempts
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="reconnection_logic",
                success=False,
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=0,
                reconnection_attempts=reconnection_attempts,
                error_message=str(e)
            )
    
    async def test_concurrent_connections(self) -> WebSocketTestResult:
        """Test multiple concurrent WebSocket connections"""
        start_time = time.time()
        num_connections = 5
        successful_connections = 0
        total_messages_sent = 0
        total_messages_received = 0
        
        async def handle_connection(connection_id: int):
            nonlocal successful_connections, total_messages_sent, total_messages_received
            
            try:
                async with websockets.connect(self.ws_url) as websocket:
                    successful_connections += 1
                    
                    # Send messages from this connection
                    for i in range(3):
                        message = {
                            "type": "concurrent_test",
                            "connection_id": connection_id,
                            "message_id": i,
                            "data": f"Message {i} from connection {connection_id}"
                        }
                        
                        await websocket.send(json.dumps(message))
                        total_messages_sent += 1
                        
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                            total_messages_received += 1
                        except asyncio.TimeoutError:
                            pass
                        
                        await asyncio.sleep(0.1)
            except Exception:
                pass
        
        try:
            # Create concurrent connections
            tasks = [handle_connection(i) for i in range(num_connections)]
            await asyncio.gather(*tasks)
            
            duration = time.time() - start_time
            
            return WebSocketTestResult(
                test_name="concurrent_connections",
                success=successful_connections >= num_connections * 0.8,  # At least 80% success
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=0,
                reconnection_attempts=0,
                connection_stability=successful_connections / num_connections
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="concurrent_connections",
                success=False,
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=0,
                reconnection_attempts=0,
                error_message=str(e)
            )
    
    async def test_message_reliability(self) -> WebSocketTestResult:
        """Test message delivery reliability"""
        start_time = time.time()
        messages_sent = 0
        messages_received = 0
        message_ids_sent = set()
        message_ids_received = set()
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send numbered messages to track delivery
                for i in range(20):
                    message_id = f"msg_{i}_{int(time.time() * 1000)}"
                    message = {
                        "type": "reliability_test",
                        "message_id": message_id,
                        "sequence": i,
                        "data": f"Reliability test message {i}"
                    }
                    
                    await websocket.send(json.dumps(message))
                    messages_sent += 1
                    message_ids_sent.add(message_id)
                    
                    # Receive response
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        response_data = json.loads(response)
                        
                        if 'message_id' in response_data:
                            message_ids_received.add(response_data['message_id'])
                        
                        messages_received += 1
                    except (asyncio.TimeoutError, json.JSONDecodeError):
                        pass
                    
                    await asyncio.sleep(0.05)  # Small delay
                
                duration = time.time() - start_time
                
                # Calculate message delivery rate
                delivery_rate = len(message_ids_received) / len(message_ids_sent) if message_ids_sent else 0
                
                return WebSocketTestResult(
                    test_name="message_reliability",
                    success=delivery_rate >= 0.9,  # 90% delivery rate
                    duration=duration,
                    messages_sent=messages_sent,
                    messages_received=messages_received,
                    connection_time=0,
                    reconnection_attempts=0,
                    connection_stability=delivery_rate
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="message_reliability",
                success=False,
                duration=duration,
                messages_sent=messages_sent,
                messages_received=messages_received,
                connection_time=0,
                reconnection_attempts=0,
                error_message=str(e)
            )
    
    async def test_large_message_handling(self) -> WebSocketTestResult:
        """Test handling of large messages"""
        start_time = time.time()
        messages_sent = 0
        messages_received = 0
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Test messages of increasing size
                sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
                
                for size in sizes:
                    large_data = "x" * size
                    message = {
                        "type": "large_message_test",
                        "size": size,
                        "data": large_data
                    }
                    
                    message_start = time.time()
                    await websocket.send(json.dumps(message))
                    messages_sent += 1
                    
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        messages_received += 1
                        message_time = time.time() - message_start
                        
                        # Check if response time is reasonable for message size
                        if message_time > 5.0:  # More than 5 seconds is problematic
                            break
                            
                    except asyncio.TimeoutError:
                        break
                
                duration = time.time() - start_time
                
                return WebSocketTestResult(
                    test_name="large_message_handling",
                    success=messages_received >= messages_sent * 0.75,  # 75% success rate
                    duration=duration,
                    messages_sent=messages_sent,
                    messages_received=messages_received,
                    connection_time=0,
                    reconnection_attempts=0
                )
                
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="large_message_handling",
                success=False,
                duration=duration,
                messages_sent=messages_sent,
                messages_received=messages_received,
                connection_time=0,
                reconnection_attempts=0,
                error_message=str(e)
            )
    
    async def test_websocket_through_ui(self, driver: webdriver.Chrome) -> WebSocketTestResult:
        """Test WebSocket functionality through UI interactions"""
        start_time = time.time()
        
        try:
            # Navigate to application
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Inject WebSocket monitoring code
            driver.execute_script("""
                window.websocketTestResults = {
                    connected: false,
                    messagesSent: 0,
                    messagesReceived: 0,
                    connectionTime: 0,
                    errors: []
                };
                
                // Monitor existing WebSocket or create new one
                const originalWebSocket = window.WebSocket;
                window.WebSocket = function(url, protocols) {
                    const ws = new originalWebSocket(url, protocols);
                    const startTime = Date.now();
                    
                    ws.addEventListener('open', () => {
                        window.websocketTestResults.connected = true;
                        window.websocketTestResults.connectionTime = Date.now() - startTime;
                    });
                    
                    ws.addEventListener('message', () => {
                        window.websocketTestResults.messagesReceived++;
                    });
                    
                    ws.addEventListener('error', (error) => {
                        window.websocketTestResults.errors.push(error.toString());
                    });
                    
                    const originalSend = ws.send;
                    ws.send = function(data) {
                        window.websocketTestResults.messagesSent++;
                        return originalSend.call(this, data);
                    };
                    
                    return ws;
                };
            """)
            
            # Wait for WebSocket connection to be established
            await asyncio.sleep(3)
            
            # Trigger UI interactions that use WebSocket
            try:
                chat_input = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
                )
                
                # Send multiple messages through UI
                test_messages = [
                    "Test WebSocket message 1",
                    "Test WebSocket message 2",
                    "Test WebSocket message 3"
                ]
                
                for message in test_messages:
                    chat_input.clear()
                    chat_input.send_keys(message)
                    chat_input.send_keys("\n")
                    await asyncio.sleep(2)
                
            except Exception:
                pass
            
            # Get WebSocket test results
            ws_results = driver.execute_script("return window.websocketTestResults;")
            
            duration = time.time() - start_time
            
            return WebSocketTestResult(
                test_name="websocket_through_ui",
                success=ws_results['connected'] and ws_results['messagesSent'] > 0,
                duration=duration,
                messages_sent=ws_results['messagesSent'],
                messages_received=ws_results['messagesReceived'],
                connection_time=ws_results['connectionTime'] / 1000,  # Convert to seconds
                reconnection_attempts=0,
                error_message="; ".join(ws_results['errors']) if ws_results['errors'] else None
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="websocket_through_ui",
                success=False,
                duration=duration,
                messages_sent=0,
                messages_received=0,
                connection_time=0,
                reconnection_attempts=0,
                error_message=str(e)
            )
    
    async def test_connection_stability_under_load(self) -> WebSocketTestResult:
        """Test WebSocket connection stability under load"""
        start_time = time.time()
        total_messages_sent = 0
        total_messages_received = 0
        connection_drops = 0
        
        try:
            async with websockets.connect(self.ws_url) as websocket:
                # Send rapid messages to test stability
                for burst in range(5):  # 5 bursts of messages
                    for i in range(20):  # 20 messages per burst
                        try:
                            message = {
                                "type": "load_test",
                                "burst": burst,
                                "message": i,
                                "timestamp": time.time()
                            }
                            
                            await websocket.send(json.dumps(message))
                            total_messages_sent += 1
                            
                            # Try to receive response quickly
                            try:
                                await asyncio.wait_for(websocket.recv(), timeout=0.1)
                                total_messages_received += 1
                            except asyncio.TimeoutError:
                                pass
                                
                        except websockets.exceptions.ConnectionClosed:
                            connection_drops += 1
                            break
                    
                    await asyncio.sleep(0.5)  # Brief pause between bursts
            
            duration = time.time() - start_time
            stability = 1.0 - (connection_drops / max(total_messages_sent, 1))
            
            return WebSocketTestResult(
                test_name="connection_stability_under_load",
                success=stability >= 0.95 and connection_drops == 0,
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=0,
                reconnection_attempts=connection_drops,
                connection_stability=stability
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return WebSocketTestResult(
                test_name="connection_stability_under_load",
                success=False,
                duration=duration,
                messages_sent=total_messages_sent,
                messages_received=total_messages_received,
                connection_time=0,
                reconnection_attempts=connection_drops,
                error_message=str(e)
            )
    
    async def run_comprehensive_websocket_test(self) -> Dict[str, Any]:
        """Run comprehensive WebSocket reliability test suite"""
        results = {
            'timestamp': time.time(),
            'test_suite': 'websocket_reliability',
            'tests': {},
            'summary': {}
        }
        
        # Test basic connection
        print("Testing basic WebSocket connection...")
        basic_result = await self.test_basic_websocket_connection()
        results['tests']['basic_connection'] = asdict(basic_result)
        
        # Test reconnection logic
        print("Testing reconnection logic...")
        reconnection_result = await self.test_reconnection_logic()
        results['tests']['reconnection_logic'] = asdict(reconnection_result)
        
        # Test concurrent connections
        print("Testing concurrent connections...")
        concurrent_result = await self.test_concurrent_connections()
        results['tests']['concurrent_connections'] = asdict(concurrent_result)
        
        # Test message reliability
        print("Testing message reliability...")
        reliability_result = await self.test_message_reliability()
        results['tests']['message_reliability'] = asdict(reliability_result)
        
        # Test large message handling
        print("Testing large message handling...")
        large_message_result = await self.test_large_message_handling()
        results['tests']['large_message_handling'] = asdict(large_message_result)
        
        # Test connection stability under load
        print("Testing connection stability under load...")
        stability_result = await self.test_connection_stability_under_load()
        results['tests']['stability_under_load'] = asdict(stability_result)
        
        # Test WebSocket through UI
        print("Testing WebSocket through UI...")
        driver = self._get_chrome_driver()
        try:
            ui_result = await self.test_websocket_through_ui(driver)
            results['tests']['websocket_through_ui'] = asdict(ui_result)
        finally:
            driver.quit()
        
        # Generate summary
        results['summary'] = self._generate_websocket_summary(results)
        
        return results
    
    def _generate_websocket_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate WebSocket test summary"""
        tests = results['tests']
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test['success'])
        
        # Calculate average metrics
        total_messages_sent = sum(test['messages_sent'] for test in tests.values())
        total_messages_received = sum(test['messages_received'] for test in tests.values())
        
        delivery_rate = (total_messages_received / total_messages_sent * 100) if total_messages_sent > 0 else 0
        
        # Calculate average latency from tests that have latency stats
        latency_tests = [test for test in tests.values() if test.get('latency_stats')]
        avg_latency = 0
        if latency_tests:
            avg_latencies = [test['latency_stats']['mean'] for test in latency_tests]
            avg_latency = np.mean(avg_latencies)
        
        # Check for critical issues
        critical_issues = []
        if passed_tests / total_tests < 0.8:
            critical_issues.append("Low test success rate")
        
        if delivery_rate < 90:
            critical_issues.append("Poor message delivery rate")
        
        reconnection_test = tests.get('reconnection_logic', {})
        if not reconnection_test.get('success', False):
            critical_issues.append("Reconnection logic not working")
        
        # Generate recommendations
        recommendations = []
        if critical_issues:
            recommendations.extend([
                "Implement robust error handling for WebSocket connections",
                "Add exponential backoff for reconnection attempts",
                "Consider implementing message queuing for offline scenarios"
            ])
        else:
            recommendations.append("WebSocket implementation is reliable and performant")
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100),
            'message_delivery_rate': delivery_rate,
            'average_latency_ms': avg_latency * 1000,
            'critical_issues': critical_issues,
            'recommendations': recommendations
        }


# Pytest fixtures and test cases
@pytest.fixture
def websocket_tester():
    """Fixture for WebSocket reliability tester"""
    return WebSocketReliabilityTester()


@pytest.mark.asyncio
async def test_basic_websocket_functionality(websocket_tester):
    """Test basic WebSocket functionality"""
    result = await websocket_tester.test_basic_websocket_connection()
    assert result.success, f"Basic WebSocket test failed: {result.error_message}"
    assert result.messages_sent > 0, "No messages were sent"
    assert result.messages_received > 0, "No messages were received"


@pytest.mark.asyncio
async def test_websocket_reconnection(websocket_tester):
    """Test WebSocket reconnection logic"""
    result = await websocket_tester.test_reconnection_logic()
    assert result.success, f"Reconnection test failed: {result.error_message}"
    assert result.reconnection_attempts >= 2, "Insufficient reconnection attempts"


@pytest.mark.asyncio
async def test_websocket_message_reliability(websocket_tester):
    """Test WebSocket message delivery reliability"""
    result = await websocket_tester.test_message_reliability()
    assert result.success, f"Message reliability test failed: {result.error_message}"
    assert result.connection_stability >= 0.9, "Message delivery rate too low"


@pytest.mark.asyncio
async def test_websocket_ui_integration(websocket_tester):
    """Test WebSocket integration through UI"""
    driver = websocket_tester._get_chrome_driver()
    try:
        result = await websocket_tester.test_websocket_through_ui(driver)
        assert result.success, f"UI WebSocket test failed: {result.error_message}"
    finally:
        driver.quit()


if __name__ == "__main__":
    # Run comprehensive WebSocket reliability test
    async def main():
        tester = WebSocketReliabilityTester()
        results = await tester.run_comprehensive_websocket_test()
        
        # Save results
        output_path = Path("tests/ui/websocket/reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"websocket_reliability_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = results['summary']
        print("WebSocket reliability test completed.")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Message delivery rate: {summary['message_delivery_rate']:.1f}%")
        print(f"Average latency: {summary['average_latency_ms']:.1f}ms")
        
        if summary['critical_issues']:
            print("\nCritical issues:")
            for issue in summary['critical_issues']:
                print(f"  - {issue}")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())