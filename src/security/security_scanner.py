"""
Security Scanner and Vulnerability Assessment

Performs security scans, vulnerability assessments, and security monitoring
for enterprise RAG systems with threat detection capabilities.
"""
import re
import logging
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import ipaddress
import json


class ThreatLevel(Enum):
    """Security threat severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTH_BYPASS = "auth_bypass"
    DATA_EXPOSURE = "data_exposure"
    INSECURE_CONFIG = "insecure_config"
    MALICIOUS_INPUT = "malicious_input"
    RATE_LIMIT_BYPASS = "rate_limit_bypass"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityFinding:
    """Represents a security finding from scanning"""
    finding_id: str
    vulnerability_type: VulnerabilityType
    threat_level: ThreatLevel
    title: str
    description: str
    affected_component: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    cve_references: List[str] = field(default_factory=list)
    discovered_at: datetime = field(default_factory=datetime.now)
    remediated: bool = False
    remediated_at: Optional[datetime] = None


@dataclass
class VulnerabilityAssessment:
    """Comprehensive vulnerability assessment report"""
    assessment_id: str
    scan_timestamp: datetime
    total_findings: int
    findings_by_severity: Dict[str, int]
    security_score: float
    compliance_status: str
    findings: List[SecurityFinding]
    recommendations: List[str]
    next_scan_due: datetime


class SecurityScanner:
    """Comprehensive security scanner for RAG systems"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Malicious patterns to detect
        self.injection_patterns = [
            # SQL Injection
            r"(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)",
            r"(?i)(\'\s*or\s*\'\s*1\s*=\s*1|admin\'\s*--)",
            
            # NoSQL Injection
            r"(?i)(\$where|\$ne|\$gt|\$lt|\$regex)",
            
            # Command Injection
            r"(?i)(;\s*rm\s+-rf|;\s*cat\s+/etc/passwd|;\s*wget|;\s*curl)",
            r"(?i)(\|\s*nc\s+|\|\s*netcat|\&\&\s*rm)",
            
            # Path Traversal
            r"(\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e%5c)",
            
            # Script Injection
            r"(?i)(<script|javascript:|vbscript:|onload=|onerror=)",
            r"(?i)(eval\(|setTimeout\(|setInterval\()"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(?i)(<script[^>]*>.*?</script>)",
            r"(?i)(javascript:|vbscript:|data:text/html)",
            r"(?i)(onload|onerror|onclick|onmouseover)=",
            r"(?i)(<iframe|<object|<embed|<link)"
        ]
        
        # Suspicious file patterns
        self.suspicious_file_patterns = [
            r"\.(?:exe|bat|cmd|scr|pif|com|dll|vbs|js|jar)$",
            r"\.(?:php|asp|jsp|py|rb|pl)$"
        ]
        
        # Malicious IP ranges (example)
        self.suspicious_ip_ranges = [
            "10.0.0.0/8",      # Private networks used maliciously
            "192.168.0.0/16",  # Private networks
            "172.16.0.0/12"    # Private networks
        ]
        
        # Initialize compiled patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self.compiled_injection_patterns = [
            re.compile(pattern) for pattern in self.injection_patterns
        ]
        
        self.compiled_xss_patterns = [
            re.compile(pattern) for pattern in self.xss_patterns
        ]
        
        self.compiled_file_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_file_patterns
        ]
    
    def scan_input_content(self, content: str, source: str = "user_input") -> List[SecurityFinding]:
        """Scan input content for security threats"""
        findings = []
        
        # Check for injection attacks
        injection_findings = self._scan_injection_attacks(content, source)
        findings.extend(injection_findings)
        
        # Check for XSS attacks
        xss_findings = self._scan_xss_attacks(content, source)
        findings.extend(xss_findings)
        
        # Check for malicious patterns
        malicious_findings = self._scan_malicious_patterns(content, source)
        findings.extend(malicious_findings)
        
        # Check for data leakage attempts
        data_findings = self._scan_data_leakage(content, source)
        findings.extend(data_findings)
        
        return findings
    
    def _scan_injection_attacks(self, content: str, source: str) -> List[SecurityFinding]:
        """Scan for injection attack patterns"""
        findings = []
        
        for i, pattern in enumerate(self.compiled_injection_patterns):
            matches = pattern.findall(content)
            if matches:
                finding = SecurityFinding(
                    finding_id=f"injection_{source}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.INJECTION,
                    threat_level=ThreatLevel.HIGH,
                    title="Potential Injection Attack Detected",
                    description=f"Detected potential injection attack pattern in {source}",
                    affected_component=source,
                    evidence={
                        "pattern_index": i,
                        "matches": matches[:5],  # Limit evidence
                        "pattern": self.injection_patterns[i]
                    },
                    recommendations=[
                        "Sanitize and validate all user inputs",
                        "Use parameterized queries for database operations",
                        "Implement input length restrictions",
                        "Apply principle of least privilege"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    def _scan_xss_attacks(self, content: str, source: str) -> List[SecurityFinding]:
        """Scan for XSS attack patterns"""
        findings = []
        
        for i, pattern in enumerate(self.compiled_xss_patterns):
            matches = pattern.findall(content)
            if matches:
                finding = SecurityFinding(
                    finding_id=f"xss_{source}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.XSS,
                    threat_level=ThreatLevel.MEDIUM,
                    title="Potential XSS Attack Detected",
                    description=f"Detected potential cross-site scripting pattern in {source}",
                    affected_component=source,
                    evidence={
                        "pattern_index": i,
                        "matches": matches[:3],
                        "pattern": self.xss_patterns[i]
                    },
                    recommendations=[
                        "Encode output data properly",
                        "Implement Content Security Policy (CSP)",
                        "Validate and sanitize HTML inputs",
                        "Use secure templating engines"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    def _scan_malicious_patterns(self, content: str, source: str) -> List[SecurityFinding]:
        """Scan for various malicious patterns"""
        findings = []
        
        # Check for suspicious file references
        for pattern in self.compiled_file_patterns:
            matches = pattern.findall(content)
            if matches:
                finding = SecurityFinding(
                    finding_id=f"malicious_file_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.MALICIOUS_INPUT,
                    threat_level=ThreatLevel.MEDIUM,
                    title="Suspicious File Reference Detected",
                    description=f"Detected reference to potentially malicious file types in {source}",
                    affected_component=source,
                    evidence={
                        "file_references": matches[:5]
                    },
                    recommendations=[
                        "Restrict file upload types",
                        "Implement file scanning",
                        "Use sandboxed file processing"
                    ]
                )
                findings.append(finding)
        
        # Check for suspicious encoding
        if self._has_suspicious_encoding(content):
            finding = SecurityFinding(
                finding_id=f"encoding_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                vulnerability_type=VulnerabilityType.MALICIOUS_INPUT,
                threat_level=ThreatLevel.LOW,
                title="Suspicious Encoding Detected",
                description="Detected potentially obfuscated or encoded content",
                affected_component=source,
                evidence={
                    "content_length": len(content),
                    "suspicious_chars": True
                },
                recommendations=[
                    "Decode and analyze suspicious content",
                    "Implement encoding validation",
                    "Log suspicious activity"
                ]
            )
            findings.append(finding)
        
        return findings
    
    def _scan_data_leakage(self, content: str, source: str) -> List[SecurityFinding]:
        """Scan for potential data leakage attempts"""
        findings = []
        
        # Common data exfiltration patterns
        exfiltration_patterns = [
            r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+",
            r"(?i)(api[_-]?key|apikey|auth[_-]?token)\s*[:=]\s*\S+",
            r"(?i)(secret|private[_-]?key)\s*[:=]\s*\S+",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP addresses
            r"(?i)file\s*:\s*///?\w+"  # File protocol
        ]
        
        for i, pattern in enumerate(exfiltration_patterns):
            matches = re.findall(pattern, content)
            if matches:
                finding = SecurityFinding(
                    finding_id=f"data_leak_{source}_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.DATA_EXPOSURE,
                    threat_level=ThreatLevel.HIGH,
                    title="Potential Data Leakage Detected",
                    description=f"Detected potential sensitive data exposure in {source}",
                    affected_component=source,
                    evidence={
                        "pattern_type": ["credentials", "api_keys", "secrets", "ip_addresses", "file_access"][i],
                        "match_count": len(matches)
                    },
                    recommendations=[
                        "Remove sensitive data from inputs",
                        "Implement data loss prevention (DLP)",
                        "Monitor for credential exposure",
                        "Use secure credential storage"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    def _has_suspicious_encoding(self, content: str) -> bool:
        """Check if content has suspicious encoding patterns"""
        # Check for high percentage of encoded characters
        encoded_chars = len(re.findall(r'%[0-9a-fA-F]{2}', content))
        unicode_escapes = len(re.findall(r'\\u[0-9a-fA-F]{4}', content))
        html_entities = len(re.findall(r'&[a-zA-Z]+;|&#\d+;', content))
        
        total_suspicious = encoded_chars + unicode_escapes + html_entities
        
        if len(content) > 0:
            suspicious_ratio = total_suspicious / len(content)
            return suspicious_ratio > 0.1  # More than 10% suspicious characters
        
        return False
    
    def scan_configuration(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Scan system configuration for security issues"""
        findings = []
        
        # Check for insecure configurations
        insecure_configs = [
            ("debug", True, "Debug mode enabled in production"),
            ("ssl_verify", False, "SSL verification disabled"),
            ("encryption", False, "Encryption disabled"),
            ("logging_level", "DEBUG", "Verbose logging enabled"),
            ("cors_origins", "*", "CORS allows all origins"),
            ("rate_limiting", False, "Rate limiting disabled")
        ]
        
        for config_key, dangerous_value, description in insecure_configs:
            if config.get(config_key) == dangerous_value:
                finding = SecurityFinding(
                    finding_id=f"config_{config_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.INSECURE_CONFIG,
                    threat_level=ThreatLevel.MEDIUM,
                    title=f"Insecure Configuration: {config_key}",
                    description=description,
                    affected_component="system_configuration",
                    evidence={
                        "config_key": config_key,
                        "current_value": config.get(config_key),
                        "recommended_value": "secure_setting"
                    },
                    recommendations=[
                        f"Change {config_key} to a secure value",
                        "Review security configuration guidelines",
                        "Implement configuration validation"
                    ]
                )
                findings.append(finding)
        
        # Check for missing security headers
        security_headers = [
            "Content-Security-Policy",
            "X-Frame-Options",
            "X-Content-Type-Options",
            "Strict-Transport-Security"
        ]
        
        response_headers = config.get("response_headers", {})
        for header in security_headers:
            if header not in response_headers:
                finding = SecurityFinding(
                    finding_id=f"header_{header}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    vulnerability_type=VulnerabilityType.INSECURE_CONFIG,
                    threat_level=ThreatLevel.LOW,
                    title=f"Missing Security Header: {header}",
                    description=f"Security header {header} is not configured",
                    affected_component="http_headers",
                    evidence={
                        "missing_header": header,
                        "current_headers": list(response_headers.keys())
                    },
                    recommendations=[
                        f"Implement {header} security header",
                        "Review OWASP security header guidelines"
                    ]
                )
                findings.append(finding)
        
        return findings
    
    def scan_network_traffic(self, 
                           source_ip: str, 
                           destination: str, 
                           payload: str) -> List[SecurityFinding]:
        """Scan network traffic for security threats"""
        findings = []
        
        # Check source IP against threat lists
        if self._is_suspicious_ip(source_ip):
            finding = SecurityFinding(
                finding_id=f"suspicious_ip_{source_ip}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                vulnerability_type=VulnerabilityType.MALICIOUS_INPUT,
                threat_level=ThreatLevel.HIGH,
                title="Suspicious Source IP Detected",
                description=f"Traffic from potentially malicious IP: {source_ip}",
                affected_component="network_traffic",
                evidence={
                    "source_ip": source_ip,
                    "destination": destination
                },
                recommendations=[
                    "Block suspicious IP address",
                    "Investigate traffic patterns",
                    "Update threat intelligence feeds"
                ]
            )
            findings.append(finding)
        
        # Scan payload for threats
        payload_findings = self.scan_input_content(payload, "network_payload")
        findings.extend(payload_findings)
        
        return findings
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # Check against known suspicious ranges
            for range_str in self.suspicious_ip_ranges:
                network = ipaddress.ip_network(range_str)
                if ip in network:
                    return True
            
            # Check for localhost/loopback from external
            if ip.is_loopback or ip.is_private:
                return True
            
        except ValueError:
            # Invalid IP format is suspicious
            return True
        
        return False
    
    def perform_vulnerability_assessment(self, 
                                       system_config: Dict[str, Any],
                                       recent_inputs: List[str],
                                       network_logs: List[Dict]) -> VulnerabilityAssessment:
        """Perform comprehensive vulnerability assessment"""
        assessment_id = f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        all_findings = []
        
        # Scan configuration
        config_findings = self.scan_configuration(system_config)
        all_findings.extend(config_findings)
        
        # Scan recent inputs
        for i, input_content in enumerate(recent_inputs[:100]):  # Limit to recent 100
            input_findings = self.scan_input_content(input_content, f"input_{i}")
            all_findings.extend(input_findings)
        
        # Scan network traffic
        for log_entry in network_logs[-50:]:  # Recent 50 network logs
            if all(key in log_entry for key in ["source_ip", "destination", "payload"]):
                network_findings = self.scan_network_traffic(
                    log_entry["source_ip"],
                    log_entry["destination"],
                    log_entry["payload"]
                )
                all_findings.extend(network_findings)
        
        # Calculate severity distribution
        findings_by_severity = {}
        for finding in all_findings:
            severity = finding.threat_level.value
            findings_by_severity[severity] = findings_by_severity.get(severity, 0) + 1
        
        # Calculate security score (0-100)
        security_score = self._calculate_security_score(all_findings)
        
        # Determine compliance status
        compliance_status = self._determine_compliance_status(all_findings)
        
        # Generate recommendations
        recommendations = self._generate_assessment_recommendations(all_findings)
        
        # Schedule next scan
        next_scan_due = datetime.now() + timedelta(days=7)  # Weekly scans
        
        assessment = VulnerabilityAssessment(
            assessment_id=assessment_id,
            scan_timestamp=datetime.now(),
            total_findings=len(all_findings),
            findings_by_severity=findings_by_severity,
            security_score=security_score,
            compliance_status=compliance_status,
            findings=all_findings,
            recommendations=recommendations,
            next_scan_due=next_scan_due
        )
        
        self.logger.info(f"Vulnerability assessment completed: {assessment_id}")
        return assessment
    
    def _calculate_security_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall security score based on findings"""
        if not findings:
            return 100.0
        
        # Severity weights
        severity_weights = {
            ThreatLevel.CRITICAL: 20,
            ThreatLevel.HIGH: 10,
            ThreatLevel.MEDIUM: 5,
            ThreatLevel.LOW: 1,
            ThreatLevel.INFO: 0.5
        }
        
        total_penalty = sum(
            severity_weights.get(finding.threat_level, 1) 
            for finding in findings
        )
        
        # Score calculation (higher penalties = lower score)
        base_score = 100.0
        penalty_factor = min(total_penalty / 10.0, 95.0)  # Cap at 95% penalty
        
        return max(base_score - penalty_factor, 5.0)  # Minimum score of 5
    
    def _determine_compliance_status(self, findings: List[SecurityFinding]) -> str:
        """Determine compliance status based on findings"""
        critical_findings = [f for f in findings if f.threat_level == ThreatLevel.CRITICAL]
        high_findings = [f for f in findings if f.threat_level == ThreatLevel.HIGH]
        
        if critical_findings:
            return "NON_COMPLIANT"
        elif len(high_findings) > 5:
            return "NON_COMPLIANT"
        elif high_findings:
            return "PARTIALLY_COMPLIANT"
        else:
            return "COMPLIANT"
    
    def _generate_assessment_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate overall recommendations based on findings"""
        recommendations = set()
        
        # Add general recommendations based on finding types
        vulnerability_types = {f.vulnerability_type for f in findings}
        
        if VulnerabilityType.INJECTION in vulnerability_types:
            recommendations.add("Implement comprehensive input validation and sanitization")
            recommendations.add("Use parameterized queries and prepared statements")
        
        if VulnerabilityType.XSS in vulnerability_types:
            recommendations.add("Implement Content Security Policy (CSP)")
            recommendations.add("Encode all output data properly")
        
        if VulnerabilityType.INSECURE_CONFIG in vulnerability_types:
            recommendations.add("Review and harden system configuration")
            recommendations.add("Implement security configuration baselines")
        
        if VulnerabilityType.DATA_EXPOSURE in vulnerability_types:
            recommendations.add("Implement data loss prevention (DLP) controls")
            recommendations.add("Regular security awareness training")
        
        # Add priority recommendations
        recommendations.add("Conduct regular security assessments")
        recommendations.add("Implement continuous security monitoring")
        recommendations.add("Maintain up-to-date threat intelligence")
        
        return list(recommendations)


# Example usage and testing
if __name__ == "__main__":
    # Test security scanner
    scanner = SecurityScanner()
    
    # Test malicious input detection
    test_inputs = [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM admin",
        "Normal user query about company policies",
        "cat /etc/passwd && rm -rf /",
        "http://example.com/malware.exe"
    ]
    
    for i, test_input in enumerate(test_inputs):
        findings = scanner.scan_input_content(test_input, f"test_input_{i}")
        print(f"Input {i+1}: {len(findings)} security findings")
        for finding in findings:
            print(f"  - {finding.threat_level.value.upper()}: {finding.title}")
    
    # Test configuration scan
    test_config = {
        "debug": True,
        "ssl_verify": False,
        "cors_origins": "*",
        "response_headers": {
            "X-Frame-Options": "DENY"
        }
    }
    
    config_findings = scanner.scan_configuration(test_config)
    print(f"\nConfiguration scan: {len(config_findings)} findings")
    for finding in config_findings:
        print(f"  - {finding.title}")
    
    # Test full vulnerability assessment
    assessment = scanner.perform_vulnerability_assessment(
        system_config=test_config,
        recent_inputs=test_inputs,
        network_logs=[]
    )
    
    print(f"\nVulnerability Assessment:")
    print(f"  Security Score: {assessment.security_score:.1f}/100")
    print(f"  Compliance Status: {assessment.compliance_status}")
    print(f"  Total Findings: {assessment.total_findings}")
    print(f"  By Severity: {assessment.findings_by_severity}")