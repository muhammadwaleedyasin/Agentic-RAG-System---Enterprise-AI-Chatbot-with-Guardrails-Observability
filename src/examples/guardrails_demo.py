"""
Enterprise Guardrails System Demo

Comprehensive demonstration of the enterprise guardrails and compliance
system for RAG applications with real-world examples.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List

# Import all guardrails components
from ..guardrails import (
    GuardrailsOrchestrator, 
    CitationEnforcer, 
    PIIDetector, 
    TopicFilter,
    ComplianceLogger
)
from ..security import (
    AccessController, 
    EncryptionManager, 
    PrivacyManager,
    SecurityScanner
)
from ..config.compliance_config import ComplianceConfig, ComplianceLevel


class GuardrailsDemo:
    """Demonstration of enterprise guardrails system"""
    
    def __init__(self):
        # Initialize all components
        self.orchestrator = GuardrailsOrchestrator()
        self.access_controller = AccessController()
        self.encryption_manager = EncryptionManager()
        self.privacy_manager = PrivacyManager(self.encryption_manager)
        self.security_scanner = SecurityScanner()
        
        # Demo data
        self.demo_users = self._create_demo_users()
        self.demo_documents = self._create_demo_documents()
        self.demo_queries = self._create_demo_queries()
    
    def _create_demo_users(self) -> Dict[str, str]:
        """Create demo users with different roles"""
        users = {
            "admin": self.access_controller.create_user(
                "admin1", "admin_user", "admin@company.com", "admin", "secure_password"
            ),
            "manager": self.access_controller.create_user(
                "mgr1", "jane_manager", "jane@company.com", "manager", "manager_password"
            ),
            "employee": self.access_controller.create_user(
                "emp1", "john_employee", "john@company.com", "employee", "employee_password"
            ),
            "contractor": self.access_controller.create_user(
                "con1", "contractor", "contractor@external.com", "contractor", "contractor_password"
            )
        }
        
        # Authenticate users and get tokens
        tokens = {}
        for role, user in users.items():
            token = self.access_controller.authenticate_user(user.username, f"{role}_password")
            tokens[role] = token
        
        return tokens
    
    def _create_demo_documents(self) -> List[Dict]:
        """Create demo documents with various access levels"""
        return [
            {
                "id": "doc_1",
                "title": "Employee Handbook",
                "content": "General company policies and procedures...",
                "metadata": {
                    "access_level": "internal",
                    "department": "hr",
                    "classification": "internal"
                }
            },
            {
                "id": "doc_2", 
                "title": "Remote Work Policy",
                "content": "Remote work is permitted with manager approval. Contact HR at hr@company.com or call (555) 123-4567 for questions.",
                "metadata": {
                    "access_level": "internal",
                    "department": "hr",
                    "classification": "internal"
                }
            },
            {
                "id": "doc_3",
                "title": "Financial Confidential Report",
                "content": "Q4 revenue was $2.5M. Strategic acquisition plans are confidential.",
                "metadata": {
                    "access_level": "confidential",
                    "department": "finance",
                    "classification": "confidential"
                }
            },
            {
                "id": "doc_4",
                "title": "Technical API Documentation",
                "content": "API endpoint: GET /api/users returns user data. Authentication required.",
                "metadata": {
                    "access_level": "internal",
                    "department": "engineering",
                    "classification": "internal"
                }
            }
        ]
    
    def _create_demo_queries(self) -> List[Dict]:
        """Create demo queries with various compliance scenarios"""
        return [
            {
                "query": "What is the company remote work policy?",
                "expected_compliance": True,
                "scenario": "Normal business query"
            },
            {
                "query": "Can you tell me John Smith's salary and SSN 123-45-6789?",
                "expected_compliance": False,
                "scenario": "PII exposure attempt"
            },
            {
                "query": "'; DROP TABLE users; -- What are the company policies?",
                "expected_compliance": False,
                "scenario": "SQL injection attempt"
            },
            {
                "query": "What medical treatment should I get for my condition?",
                "expected_compliance": False,
                "scenario": "Blocked topic (medical advice)"
            },
            {
                "query": "How do I access the confidential financial reports?",
                "expected_compliance": False,
                "scenario": "Unauthorized access attempt"
            },
            {
                "query": "What is our API documentation for user management?",
                "expected_compliance": True,
                "scenario": "Technical documentation query"
            }
        ]
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all guardrails features"""
        print("🔒 Enterprise Guardrails System Demo")
        print("=" * 50)
        
        # 1. Access Control Demo
        await self._demo_access_control()
        
        # 2. PII Detection Demo
        await self._demo_pii_detection()
        
        # 3. Citation Enforcement Demo
        await self._demo_citation_enforcement()
        
        # 4. Topic Filtering Demo
        await self._demo_topic_filtering()
        
        # 5. Security Scanning Demo
        await self._demo_security_scanning()
        
        # 6. Full Orchestrator Demo
        await self._demo_full_orchestration()
        
        # 7. Privacy Management Demo
        await self._demo_privacy_management()
        
        # 8. Compliance Reporting Demo
        await self._demo_compliance_reporting()
        
        print("\n✅ Enterprise Guardrails Demo Complete!")
    
    async def _demo_access_control(self):
        """Demonstrate access control features"""
        print("\n🛡️  Access Control Demo")
        print("-" * 30)
        
        # Test different user access levels
        for role, token in self.demo_users.items():
            print(f"\n👤 Testing {role.upper()} access:")
            
            # Filter documents by access level
            accessible_docs = self.access_controller.filter_documents_by_access(
                token, self.demo_documents
            )
            
            print(f"   Can access {len(accessible_docs)} of {len(self.demo_documents)} documents")
            
            # Test specific permissions
            can_delete = self.access_controller.check_permission(
                token, self.access_controller.roles[role].permissions.__iter__().__next__()
            )
            print(f"   Has admin permissions: {role == 'admin'}")
    
    async def _demo_pii_detection(self):
        """Demonstrate PII detection and redaction"""
        print("\n🔍 PII Detection Demo")
        print("-" * 30)
        
        detector = PIIDetector()
        
        test_texts = [
            "Contact John Doe at john.doe@company.com or (555) 123-4567",
            "My SSN is 123-45-6789 and credit card is 4532-1234-5678-9012",
            "Send the report to IP address 192.168.1.100",
            "This text contains no PII information"
        ]
        
        for i, text in enumerate(test_texts):
            result = detector.detect_pii(text)
            print(f"\n📝 Text {i+1}: '{text[:50]}...'")
            print(f"   PII Detected: {result.has_pii}")
            print(f"   Risk Level: {result.risk_level}")
            print(f"   Entities: {[e.entity_type for e in result.entities_found]}")
            if result.has_pii:
                print(f"   Redacted: '{result.redacted_text[:50]}...'")
    
    async def _demo_citation_enforcement(self):
        """Demonstrate citation enforcement"""
        print("\n📚 Citation Enforcement Demo")
        print("-" * 30)
        
        enforcer = CitationEnforcer()
        
        test_cases = [
            {
                "response": "The company allows remote work with approval [1]. Contact HR for questions [2].",
                "sources": [
                    {"id": "1", "title": "Remote Work Policy"},
                    {"id": "2", "title": "HR Contact Info"}
                ]
            },
            {
                "response": "Remote work is allowed. No manager approval needed.",
                "sources": [
                    {"id": "1", "title": "Remote Work Policy"}
                ]
            }
        ]
        
        for i, case in enumerate(test_cases):
            enforced_response, result = enforcer.enforce_citations(
                case["response"], case["sources"]
            )
            
            print(f"\n📄 Case {i+1}:")
            print(f"   Original: '{case['response']}'")
            print(f"   Citations Valid: {result.is_valid}")
            print(f"   Citations Found: {len(result.citations_found)}")
            print(f"   Violations: {result.violations}")
            if result.suggestions:
                print(f"   Suggestions: {result.suggestions}")
    
    async def _demo_topic_filtering(self):
        """Demonstrate topic filtering"""
        print("\n🚫 Topic Filtering Demo")
        print("-" * 30)
        
        topic_filter = TopicFilter()
        
        test_queries = [
            "What is the company remote work policy?",
            "What medication should I take for my headache?",
            "How should I invest my personal savings?",
            "What are our API authentication methods?"
        ]
        
        for query in test_queries:
            result = topic_filter.filter_content(query)
            
            print(f"\n❓ Query: '{query}'")
            print(f"   Allowed: {result.is_allowed}")
            if result.primary_topic:
                print(f"   Primary Topic: {result.primary_topic.topic} ({result.primary_topic.confidence:.2f})")
            if result.blocked_topics:
                print(f"   Blocked Topics: {result.blocked_topics}")
            print(f"   Action: {result.action_taken}")
    
    async def _demo_security_scanning(self):
        """Demonstrate security scanning"""
        print("\n🔍 Security Scanning Demo")
        print("-" * 30)
        
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "Normal user query about policies",
            "cat /etc/passwd && rm -rf /",
            "What is the remote work policy?"
        ]
        
        for i, input_text in enumerate(malicious_inputs):
            findings = self.security_scanner.scan_input_content(input_text, f"input_{i}")
            
            print(f"\n🔍 Input {i+1}: '{input_text[:50]}...'")
            print(f"   Security Findings: {len(findings)}")
            
            for finding in findings:
                print(f"   - {finding.threat_level.value.upper()}: {finding.title}")
    
    async def _demo_full_orchestration(self):
        """Demonstrate full guardrails orchestration"""
        print("\n🎭 Full Orchestration Demo")
        print("-" * 30)
        
        for query_case in self.demo_queries:
            print(f"\n🤖 Scenario: {query_case['scenario']}")
            print(f"   Query: '{query_case['query']}'")
            
            # Simulate RAG response
            mock_response = f"Based on company policy, {query_case['query'].lower()}"
            mock_sources = [{"id": "1", "title": "Company Policy"}]
            
            # Process through guardrails
            result = self.orchestrator.process_response(
                response_text=mock_response,
                retrieved_sources=mock_sources,
                user_id="demo_user",
                session_id="demo_session",
                query_text=query_case['query']
            )
            
            print(f"   Compliant: {result.is_compliant}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Violations: {len(result.violations)}")
            print(f"   Actions Taken: {result.actions_taken}")
            
            if result.violations:
                print(f"   Violation Details: {result.violations[:2]}")
    
    async def _demo_privacy_management(self):
        """Demonstrate privacy management features"""
        print("\n🔐 Privacy Management Demo")
        print("-" * 30)
        
        # Register data subject
        subject = self.privacy_manager.register_data_subject(
            "demo_user_123",
            email="demo.user@company.com",
            name="Demo User"
        )
        
        print(f"📋 Registered data subject: {subject.subject_id}")
        
        # Record data processing
        record = self.privacy_manager.record_data_processing(
            subject_id="demo_user_123",
            data_type="query_logs",
            purpose="service_improvement",
            legal_basis="legitimate_interest"
        )
        
        print(f"📝 Recorded data processing: {record.record_id}")
        
        # Update consent
        self.privacy_manager.update_consent("demo_user_123", "service_improvement", True)
        print("✅ Updated consent status")
        
        # Handle access request
        user_data = self.privacy_manager.handle_access_request("demo_user_123")
        print(f"📊 Access request data: {len(user_data['processing_records'])} records")
        
        # Generate privacy report
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        report = self.privacy_manager.generate_privacy_report(start_date, end_date)
        
        print(f"📈 Privacy report: {report['data_processing']['total_records']} total records")
    
    async def _demo_compliance_reporting(self):
        """Demonstrate compliance reporting"""
        print("\n📊 Compliance Reporting Demo")
        print("-" * 30)
        
        # Get system status
        status = self.orchestrator.get_system_status()
        print("🔧 System Status:")
        print(f"   Guardrails Enabled: {status['guardrails_enabled']}")
        print(f"   Compliance Level: {status['compliance_level']}")
        print(f"   Total Requests: {status['statistics']['total_requests']}")
        
        # Export audit report
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now()
        
        audit_report = self.orchestrator.compliance_logger.export_audit_report(
            start_date, end_date
        )
        
        print(f"\n📋 Audit Report:")
        print(f"   Events: {audit_report['summary']['total_events']}")
        print(f"   Compliance Rate: {audit_report['summary'].get('compliance_rate', 'N/A')}")
        
        # Security assessment
        assessment = self.security_scanner.perform_vulnerability_assessment(
            system_config={"debug": False, "ssl_verify": True},
            recent_inputs=["normal query", "'; DROP TABLE users; --"],
            network_logs=[]
        )
        
        print(f"\n🔒 Security Assessment:")
        print(f"   Security Score: {assessment.security_score:.1f}/100")
        print(f"   Compliance Status: {assessment.compliance_status}")
        print(f"   Total Findings: {assessment.total_findings}")


async def main():
    """Run the enterprise guardrails demo"""
    try:
        demo = GuardrailsDemo()
        await demo.run_comprehensive_demo()
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())