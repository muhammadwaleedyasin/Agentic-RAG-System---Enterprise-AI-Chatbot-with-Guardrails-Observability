"""
Security Feature Validation Testing Suite for RAG Chatbot UI

Tests security features, authentication flows, data protection,
input validation, and vulnerability assessment through the UI layer.
"""

import asyncio
import time
import json
import base64
import hashlib
import hmac
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import pytest
import requests
from pathlib import Path
import re
import uuid


@dataclass
class SecurityTestResult:
    """Security test result container"""
    test_name: str
    security_feature: str
    success: bool
    vulnerability_detected: bool
    risk_level: str  # low, medium, high, critical
    details: str
    recommendations: List[str]
    error_message: Optional[str] = None


class SecurityFeatureTester:
    """Security feature validation and vulnerability testing"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_session = requests.Session()
        
        # Common security test patterns
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "';alert('XSS');//",
            "<svg onload=alert('XSS')>",
            "<%2Fscript%3E%3Cscript%3Ealert%28%22XSS%22%29%3C%2Fscript%3E"
        ]
        
        self.sql_injection_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT * FROM users --",
            "1' OR '1'='1' --",
            "admin'--",
            "' OR 1=1#"
        ]
        
        self.path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "....//....//....//etc/passwd",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]
        
        self.command_injection_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& dir",
            "`whoami`",
            "$(id)",
            "; ping -c 4 127.0.0.1"
        ]
        
        self.sensitive_data_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',  # IP address
            r'password\s*[=:]\s*[\'"]?([^\'"\s]+)',  # Passwords
        ]
    
    def _get_chrome_driver(self) -> webdriver.Chrome:
        """Configure Chrome driver for security testing"""
        options = ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        
        # Enable security-relevant logging
        options.add_experimental_option('perfLoggingPrefs', {
            'enableNetwork': True,
            'enablePage': False
        })
        
        caps = options.to_capabilities()
        caps['goog:loggingPrefs'] = {'performance': 'ALL', 'browser': 'ALL', 'driver': 'ALL'}
        
        return webdriver.Chrome(options=options)
    
    async def test_xss_protection(self, driver: webdriver.Chrome) -> SecurityTestResult:
        """Test Cross-Site Scripting (XSS) protection"""
        vulnerabilities_found = []
        
        try:
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Test XSS in input fields
            input_fields = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea, input[type='search']")
            
            for field in input_fields:
                for payload in self.xss_payloads[:3]:  # Test first 3 payloads
                    try:
                        field.clear()
                        field.send_keys(payload)
                        field.send_keys(Keys.RETURN)
                        
                        await asyncio.sleep(1)
                        
                        # Check if script executed (alert would be blocked in headless mode)
                        # Check for payload in page source instead
                        page_source = driver.page_source.lower()
                        
                        if payload.lower() in page_source and "<script>" in page_source:
                            vulnerabilities_found.append(f"XSS payload reflected: {payload}")
                        
                        # Check for JavaScript errors that might indicate blocked XSS
                        logs = driver.get_log('browser')
                        for log in logs:
                            if 'script' in log['message'].lower() and 'blocked' not in log['message'].lower():
                                vulnerabilities_found.append(f"Potential XSS execution: {log['message']}")
                        
                    except Exception:
                        continue
            
            # Test XSS in URL parameters
            xss_urls = [
                f"{self.base_url}?search={payload}"
                for payload in self.xss_payloads[:2]
            ]
            
            for url in xss_urls:
                try:
                    driver.get(url)
                    await asyncio.sleep(1)
                    
                    page_source = driver.page_source
                    for payload in self.xss_payloads:
                        if payload in page_source and not self._is_properly_escaped(payload, page_source):
                            vulnerabilities_found.append(f"URL XSS vulnerability: {payload}")
                
                except Exception:
                    continue
            
            vulnerability_detected = len(vulnerabilities_found) > 0
            risk_level = "high" if vulnerability_detected else "low"
            
            recommendations = []
            if vulnerability_detected:
                recommendations.extend([
                    "Implement proper input validation and output encoding",
                    "Use Content Security Policy (CSP) headers",
                    "Enable XSS protection headers",
                    "Sanitize user input before rendering"
                ])
            else:
                recommendations.append("XSS protection appears to be working correctly")
            
            return SecurityTestResult(
                test_name="xss_protection",
                security_feature="Cross-Site Scripting Protection",
                success=not vulnerability_detected,
                vulnerability_detected=vulnerability_detected,
                risk_level=risk_level,
                details=f"Tested {len(self.xss_payloads)} XSS payloads. Vulnerabilities: {vulnerabilities_found}",
                recommendations=recommendations
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name="xss_protection",
                security_feature="Cross-Site Scripting Protection",
                success=False,
                vulnerability_detected=False,
                risk_level="unknown",
                details="Test execution failed",
                recommendations=["Unable to complete XSS testing"],
                error_message=str(e)
            )
    
    def _is_properly_escaped(self, payload: str, page_source: str) -> bool:
        """Check if payload is properly escaped in page source"""
        escaped_versions = [
            payload.replace('<', '&lt;').replace('>', '&gt;'),
            payload.replace('<', '\\u003c').replace('>', '\\u003e'),
            payload.replace('"', '&quot;').replace("'", '&#x27;')
        ]
        
        return any(escaped in page_source for escaped in escaped_versions)
    
    async def test_input_validation(self, driver: webdriver.Chrome) -> SecurityTestResult:
        """Test input validation and sanitization"""
        validation_failures = []
        
        try:
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Find input fields
            input_fields = driver.find_elements(By.CSS_SELECTOR, "input, textarea, select")
            
            # Test various malicious inputs
            malicious_inputs = [
                "A" * 10000,  # Buffer overflow attempt
                "\x00\x01\x02\x03",  # Null bytes and control characters
                "{{7*7}}",  # Template injection
                "${7*7}",  # Expression injection
                "<!--#exec cmd='/bin/ls'-->",  # SSI injection
                "<%= 7*7 %>",  # ERB injection
            ]
            
            for field in input_fields:
                field_type = field.get_attribute('type')
                field_name = field.get_attribute('name') or field.get_attribute('id') or 'unknown'
                
                for malicious_input in malicious_inputs:
                    try:
                        field.clear()
                        field.send_keys(malicious_input)
                        
                        # Check if input was accepted without validation
                        current_value = field.get_attribute('value')
                        
                        if current_value == malicious_input:
                            # Check if this might be a vulnerability
                            if len(malicious_input) > 1000 and len(current_value) > 1000:
                                validation_failures.append(f"Field '{field_name}' accepts extremely long input")
                            elif any(char in malicious_input for char in ['\x00', '{{', '${', '<%=']):
                                validation_failures.append(f"Field '{field_name}' accepts potentially dangerous characters")
                        
                    except Exception:
                        continue
            
            # Test SQL injection patterns in search/query fields
            search_fields = driver.find_elements(By.CSS_SELECTOR, "input[type='search'], input[name*='search'], input[name*='query']")
            
            for field in search_fields:
                for sql_payload in self.sql_injection_payloads[:3]:
                    try:
                        field.clear()
                        field.send_keys(sql_payload)
                        field.send_keys(Keys.RETURN)
                        
                        await asyncio.sleep(1)
                        
                        # Check for SQL error messages
                        page_source = driver.page_source.lower()
                        sql_error_indicators = [
                            'sql syntax', 'mysql error', 'postgresql error',
                            'sqlite error', 'database error', 'syntax error'
                        ]
                        
                        if any(indicator in page_source for indicator in sql_error_indicators):
                            validation_failures.append(f"SQL injection vulnerability detected with payload: {sql_payload}")
                        
                    except Exception:
                        continue
            
            vulnerability_detected = len(validation_failures) > 0
            risk_level = "high" if vulnerability_detected else "low"
            
            recommendations = []
            if vulnerability_detected:
                recommendations.extend([
                    "Implement comprehensive input validation",
                    "Use parameterized queries to prevent SQL injection",
                    "Set maximum input lengths",
                    "Validate and sanitize all user inputs",
                    "Implement rate limiting for form submissions"
                ])
            else:
                recommendations.append("Input validation appears to be properly implemented")
            
            return SecurityTestResult(
                test_name="input_validation",
                security_feature="Input Validation and Sanitization",
                success=not vulnerability_detected,
                vulnerability_detected=vulnerability_detected,
                risk_level=risk_level,
                details=f"Tested input validation on {len(input_fields)} fields. Failures: {validation_failures}",
                recommendations=recommendations
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name="input_validation",
                security_feature="Input Validation and Sanitization",
                success=False,
                vulnerability_detected=False,
                risk_level="unknown",
                details="Test execution failed",
                recommendations=["Unable to complete input validation testing"],
                error_message=str(e)
            )
    
    async def test_authentication_security(self, driver: webdriver.Chrome) -> SecurityTestResult:
        """Test authentication and session security"""
        auth_issues = []
        
        try:
            # Test for authentication bypass attempts
            driver.get(self.base_url)
            
            # Check for authentication requirements
            current_url = driver.current_url
            page_source = driver.page_source.lower()
            
            # Look for authentication elements
            auth_elements = driver.find_elements(By.CSS_SELECTOR, "input[type='password'], form[action*='login'], .login, .auth")
            
            if not auth_elements:
                # No authentication found - check if this is intentional
                if 'admin' in page_source or 'dashboard' in page_source:
                    auth_issues.append("Administrative interface may lack authentication")
            
            # Test for session security
            cookies = driver.get_cookies()
            for cookie in cookies:
                cookie_name = cookie['name'].lower()
                
                # Check for secure cookie attributes
                if 'session' in cookie_name or 'auth' in cookie_name or 'token' in cookie_name:
                    if not cookie.get('secure', False):
                        auth_issues.append(f"Security cookie '{cookie['name']}' missing Secure flag")
                    
                    if not cookie.get('httpOnly', False):
                        auth_issues.append(f"Security cookie '{cookie['name']}' missing HttpOnly flag")
                    
                    # Check for weak session tokens
                    token_value = cookie['value']
                    if len(token_value) < 16:
                        auth_issues.append(f"Session token '{cookie['name']}' appears weak (too short)")
            
            # Test for common authentication bypasses
            bypass_attempts = [
                f"{self.base_url}/admin",
                f"{self.base_url}/dashboard",
                f"{self.base_url}/api/admin",
                f"{self.base_url}/../admin",
                f"{self.base_url}/admin../admin"
            ]
            
            for url in bypass_attempts:
                try:
                    driver.get(url)
                    await asyncio.sleep(1)
                    
                    if driver.current_url == url and "404" not in driver.page_source:
                        # Check if we accessed something we shouldn't
                        source = driver.page_source.lower()
                        if any(term in source for term in ['admin', 'dashboard', 'users', 'configuration']):
                            auth_issues.append(f"Potential authentication bypass at: {url}")
                
                except Exception:
                    continue
            
            # Test for information disclosure
            info_disclosure_urls = [
                f"{self.base_url}/.env",
                f"{self.base_url}/config.json",
                f"{self.base_url}/.git/config",
                f"{self.base_url}/robots.txt",
                f"{self.base_url}/sitemap.xml"
            ]
            
            for url in info_disclosure_urls:
                try:
                    response = self.test_session.get(url, timeout=5)
                    if response.status_code == 200 and len(response.text) > 0:
                        content = response.text.lower()
                        if any(sensitive in content for sensitive in ['password', 'secret', 'key', 'token']):
                            auth_issues.append(f"Sensitive information disclosed at: {url}")
                
                except Exception:
                    continue
            
            vulnerability_detected = len(auth_issues) > 0
            risk_level = "critical" if any("bypass" in issue for issue in auth_issues) else "medium" if vulnerability_detected else "low"
            
            recommendations = []
            if vulnerability_detected:
                recommendations.extend([
                    "Implement proper authentication for all sensitive areas",
                    "Use secure session management with proper cookie attributes",
                    "Implement proper access controls and authorization",
                    "Remove or secure sensitive configuration files",
                    "Use strong session tokens with sufficient entropy"
                ])
            else:
                recommendations.append("Authentication security appears to be properly implemented")
            
            return SecurityTestResult(
                test_name="authentication_security",
                security_feature="Authentication and Session Security",
                success=not vulnerability_detected,
                vulnerability_detected=vulnerability_detected,
                risk_level=risk_level,
                details=f"Authentication security assessment. Issues found: {auth_issues}",
                recommendations=recommendations
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name="authentication_security",
                security_feature="Authentication and Session Security",
                success=False,
                vulnerability_detected=False,
                risk_level="unknown",
                details="Test execution failed",
                recommendations=["Unable to complete authentication security testing"],
                error_message=str(e)
            )
    
    async def test_data_protection(self, driver: webdriver.Chrome) -> SecurityTestResult:
        """Test data protection and privacy features"""
        data_issues = []
        
        try:
            driver.get(self.base_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Test for sensitive data exposure in client-side code
            page_source = driver.page_source
            
            # Check for sensitive data patterns
            for pattern in self.sensitive_data_patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                if matches:
                    data_issues.append(f"Potential sensitive data exposed: {pattern}")
            
            # Check JavaScript for sensitive data
            js_content = driver.execute_script("return document.documentElement.innerHTML;")
            
            # Look for API keys, tokens, passwords in JavaScript
            sensitive_js_patterns = [
                r'api[_-]?key[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{10,})',
                r'secret[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{10,})',
                r'token[\'"\s]*[:=][\'"\s]*([a-zA-Z0-9]{10,})',
                r'password[\'"\s]*[:=][\'"\s]*([^\'"\\s]{6,})',
            ]
            
            for pattern in sensitive_js_patterns:
                matches = re.findall(pattern, js_content, re.IGNORECASE)
                if matches:
                    data_issues.append(f"Sensitive data in JavaScript: {pattern}")
            
            # Test form data handling
            forms = driver.find_elements(By.TAG_NAME, "form")
            for form in forms:
                action = form.get_attribute('action')
                method = form.get_attribute('method')
                
                # Check for forms submitting over HTTP
                if action and action.startswith('http://'):
                    data_issues.append(f"Form submits data over insecure HTTP: {action}")
                
                # Check for password fields without proper attributes
                password_fields = form.find_elements(By.CSS_SELECTOR, "input[type='password']")
                for field in password_fields:
                    autocomplete = field.get_attribute('autocomplete')
                    if autocomplete != 'new-password' and autocomplete != 'current-password':
                        data_issues.append("Password field missing proper autocomplete attribute")
            
            # Test for data leakage in network requests
            driver.execute_script("console.clear();")
            
            # Trigger some interactions to generate network traffic
            input_fields = driver.find_elements(By.CSS_SELECTOR, "input[type='text'], textarea")
            if input_fields:
                input_fields[0].send_keys("test data for network analysis")
                input_fields[0].send_keys(Keys.RETURN)
                await asyncio.sleep(2)
            
            # Check performance logs for sensitive data in requests
            logs = driver.get_log('performance')
            for log in logs:
                try:
                    message = json.loads(log['message'])
                    if message['message']['method'] == 'Network.requestWillBeSent':
                        request_data = message['message']['params']
                        url = request_data['request']['url']
                        
                        # Check for sensitive data in URLs
                        for pattern in self.sensitive_data_patterns:
                            if re.search(pattern, url, re.IGNORECASE):
                                data_issues.append(f"Sensitive data in URL: {url}")
                        
                        # Check POST data
                        if 'postData' in request_data['request']:
                            post_data = request_data['request']['postData']
                            for pattern in self.sensitive_data_patterns:
                                if re.search(pattern, post_data, re.IGNORECASE):
                                    data_issues.append("Sensitive data in POST request")
                
                except Exception:
                    continue
            
            # Test local storage and session storage for sensitive data
            local_storage = driver.execute_script("return JSON.stringify(localStorage);")
            session_storage = driver.execute_script("return JSON.stringify(sessionStorage);")
            
            for storage_name, storage_data in [("localStorage", local_storage), ("sessionStorage", session_storage)]:
                if storage_data:
                    for pattern in self.sensitive_data_patterns:
                        if re.search(pattern, storage_data, re.IGNORECASE):
                            data_issues.append(f"Sensitive data in {storage_name}")
            
            vulnerability_detected = len(data_issues) > 0
            risk_level = "high" if vulnerability_detected else "low"
            
            recommendations = []
            if vulnerability_detected:
                recommendations.extend([
                    "Remove sensitive data from client-side code",
                    "Use HTTPS for all data transmission",
                    "Implement proper data encryption for stored data",
                    "Avoid storing sensitive data in browser storage",
                    "Use secure coding practices to prevent data leakage"
                ])
            else:
                recommendations.append("Data protection measures appear to be properly implemented")
            
            return SecurityTestResult(
                test_name="data_protection",
                security_feature="Data Protection and Privacy",
                success=not vulnerability_detected,
                vulnerability_detected=vulnerability_detected,
                risk_level=risk_level,
                details=f"Data protection assessment. Issues found: {data_issues}",
                recommendations=recommendations
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name="data_protection",
                security_feature="Data Protection and Privacy",
                success=False,
                vulnerability_detected=False,
                risk_level="unknown",
                details="Test execution failed",
                recommendations=["Unable to complete data protection testing"],
                error_message=str(e)
            )
    
    async def test_security_headers(self) -> SecurityTestResult:
        """Test security headers configuration"""
        header_issues = []
        
        try:
            response = self.test_session.get(self.base_url, timeout=10)
            headers = response.headers
            
            # Required security headers
            required_headers = {
                'Content-Security-Policy': 'CSP header missing',
                'X-Frame-Options': 'Clickjacking protection missing',
                'X-Content-Type-Options': 'MIME type sniffing protection missing',
                'Strict-Transport-Security': 'HSTS header missing',
                'X-XSS-Protection': 'XSS protection header missing',
                'Referrer-Policy': 'Referrer policy not set'
            }
            
            for header, issue in required_headers.items():
                if header not in headers:
                    header_issues.append(issue)
                else:
                    # Validate header values
                    header_value = headers[header].lower()
                    
                    if header == 'X-Frame-Options':
                        if header_value not in ['deny', 'sameorigin']:
                            header_issues.append("X-Frame-Options header has weak value")
                    
                    elif header == 'X-Content-Type-Options':
                        if 'nosniff' not in header_value:
                            header_issues.append("X-Content-Type-Options should be 'nosniff'")
                    
                    elif header == 'Content-Security-Policy':
                        if 'unsafe-inline' in header_value or 'unsafe-eval' in header_value:
                            header_issues.append("CSP contains unsafe directives")
                    
                    elif header == 'Strict-Transport-Security':
                        if 'max-age' not in header_value:
                            header_issues.append("HSTS header missing max-age")
            
            # Check for information disclosure headers
            disclosure_headers = ['Server', 'X-Powered-By', 'X-AspNet-Version']
            for header in disclosure_headers:
                if header in headers:
                    header_issues.append(f"Information disclosure header present: {header}")
            
            vulnerability_detected = len(header_issues) > 0
            risk_level = "medium" if vulnerability_detected else "low"
            
            recommendations = []
            if vulnerability_detected:
                recommendations.extend([
                    "Implement all recommended security headers",
                    "Configure Content Security Policy (CSP) properly",
                    "Enable HSTS for HTTPS sites",
                    "Remove information disclosure headers",
                    "Review and strengthen security header configurations"
                ])
            else:
                recommendations.append("Security headers are properly configured")
            
            return SecurityTestResult(
                test_name="security_headers",
                security_feature="HTTP Security Headers",
                success=not vulnerability_detected,
                vulnerability_detected=vulnerability_detected,
                risk_level=risk_level,
                details=f"Security headers assessment. Issues found: {header_issues}",
                recommendations=recommendations
            )
            
        except Exception as e:
            return SecurityTestResult(
                test_name="security_headers",
                security_feature="HTTP Security Headers",
                success=False,
                vulnerability_detected=False,
                risk_level="unknown",
                details="Test execution failed",
                recommendations=["Unable to complete security headers testing"],
                error_message=str(e)
            )
    
    async def run_comprehensive_security_test(self) -> Dict[str, Any]:
        """Run comprehensive security feature validation test"""
        driver = self._get_chrome_driver()
        
        try:
            results = {
                'timestamp': time.time(),
                'test_suite': 'security_features',
                'tests': {},
                'summary': {}
            }
            
            # Test XSS protection
            print("Testing XSS protection...")
            xss_result = await self.test_xss_protection(driver)
            results['tests']['xss_protection'] = asdict(xss_result)
            
            # Test input validation
            print("Testing input validation...")
            validation_result = await self.test_input_validation(driver)
            results['tests']['input_validation'] = asdict(validation_result)
            
            # Test authentication security
            print("Testing authentication security...")
            auth_result = await self.test_authentication_security(driver)
            results['tests']['authentication_security'] = asdict(auth_result)
            
            # Test data protection
            print("Testing data protection...")
            data_result = await self.test_data_protection(driver)
            results['tests']['data_protection'] = asdict(data_result)
            
            # Test security headers (without driver)
            print("Testing security headers...")
            headers_result = await self.test_security_headers()
            results['tests']['security_headers'] = asdict(headers_result)
            
            # Generate security summary
            results['summary'] = self._generate_security_summary(results)
            
            return results
            
        finally:
            driver.quit()
    
    def _generate_security_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive security summary"""
        tests = results['tests']
        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test['success'])
        
        # Count vulnerabilities by risk level
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        vulnerabilities = []
        
        for test in tests.values():
            if test['vulnerability_detected']:
                risk_level = test['risk_level']
                risk_counts[risk_level] += 1
                vulnerabilities.append({
                    'test': test['test_name'],
                    'feature': test['security_feature'],
                    'risk': risk_level,
                    'details': test['details']
                })
        
        # Calculate overall security score
        max_score = 100
        deductions = {
            'critical': 30,
            'high': 20,
            'medium': 10,
            'low': 5
        }
        
        security_score = max_score
        for risk, count in risk_counts.items():
            security_score -= (deductions[risk] * count)
        
        security_score = max(0, security_score)
        
        # Determine overall security status
        if security_score >= 90:
            security_status = "excellent"
        elif security_score >= 75:
            security_status = "good"
        elif security_score >= 60:
            security_status = "fair"
        else:
            security_status = "poor"
        
        # Generate priority recommendations
        priority_recommendations = []
        if risk_counts['critical'] > 0:
            priority_recommendations.append("CRITICAL: Address critical security vulnerabilities immediately")
        if risk_counts['high'] > 0:
            priority_recommendations.append("HIGH: Fix high-risk security issues as soon as possible")
        if risk_counts['medium'] > 0:
            priority_recommendations.append("MEDIUM: Address medium-risk security issues in next release")
        
        # Collect all recommendations
        all_recommendations = []
        for test in tests.values():
            all_recommendations.extend(test['recommendations'])
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'security_score': security_score,
            'security_status': security_status,
            'vulnerabilities_by_risk': risk_counts,
            'total_vulnerabilities': sum(risk_counts.values()),
            'vulnerabilities': vulnerabilities,
            'priority_recommendations': priority_recommendations,
            'all_recommendations': list(set(all_recommendations))  # Remove duplicates
        }


# Pytest fixtures and test cases
@pytest.fixture
def security_tester():
    """Fixture for security feature tester"""
    return SecurityFeatureTester()


@pytest.mark.asyncio
async def test_xss_protection_validation(security_tester):
    """Test XSS protection validation"""
    driver = security_tester._get_chrome_driver()
    try:
        result = await security_tester.test_xss_protection(driver)
        assert result.success, f"XSS protection test failed: {result.details}"
        assert not result.vulnerability_detected, "XSS vulnerabilities detected"
    finally:
        driver.quit()


@pytest.mark.asyncio
async def test_input_validation_security(security_tester):
    """Test input validation security"""
    driver = security_tester._get_chrome_driver()
    try:
        result = await security_tester.test_input_validation(driver)
        assert result.success, f"Input validation test failed: {result.details}"
        if result.vulnerability_detected and result.risk_level in ['high', 'critical']:
            pytest.fail(f"High-risk input validation vulnerabilities detected: {result.details}")
    finally:
        driver.quit()


@pytest.mark.asyncio
async def test_security_headers_configuration(security_tester):
    """Test security headers configuration"""
    result = await security_tester.test_security_headers()
    assert result.success, f"Security headers test failed: {result.details}"
    
    # Allow medium risk but fail on high/critical
    if result.vulnerability_detected and result.risk_level in ['high', 'critical']:
        pytest.fail(f"Critical security header issues detected: {result.details}")


if __name__ == "__main__":
    # Run comprehensive security test
    async def main():
        tester = SecurityFeatureTester()
        results = await tester.run_comprehensive_security_test()
        
        # Save results
        output_path = Path("tests/ui/security/reports")
        output_path.mkdir(exist_ok=True)
        
        with open(output_path / f"security_report_{int(time.time())}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        summary = results['summary']
        print("Security feature validation completed.")
        print(f"Security score: {summary['security_score']}/100 ({summary['security_status']})")
        print(f"Total vulnerabilities: {summary['total_vulnerabilities']}")
        
        if summary['vulnerabilities']:
            print("\nVulnerabilities found:")
            for vuln in summary['vulnerabilities']:
                print(f"  - {vuln['feature']} ({vuln['risk']}): {vuln['details']}")
        
        if summary['priority_recommendations']:
            print("\nPriority recommendations:")
            for rec in summary['priority_recommendations']:
                print(f"  - {rec}")
    
    asyncio.run(main())