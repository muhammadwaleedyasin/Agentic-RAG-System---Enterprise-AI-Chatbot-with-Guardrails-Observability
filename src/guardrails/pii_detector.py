"""
PII Detection and Redaction System

Detects personally identifiable information (PII) in text and provides
configurable redaction capabilities for enterprise compliance.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib

from ..config.compliance_config import get_compliance_config, PIIDetectionConfig, ActionType


@dataclass
class PIIEntity:
    """Represents a detected PII entity"""
    entity_type: str
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    pattern_name: Optional[str] = None
    redacted_text: Optional[str] = None


@dataclass
class PIIDetectionResult:
    """Result of PII detection scan"""
    has_pii: bool
    entities_found: List[PIIEntity]
    redacted_text: str
    confidence_score: float
    risk_level: str
    recommendations: List[str]


class PIIDetector:
    """Detects various types of PII using pattern matching and ML models"""
    
    def __init__(self, config: Optional[PIIDetectionConfig] = None):
        self.config = config or get_compliance_config().pii_detection
        self.logger = logging.getLogger(__name__)
        
        # Built-in PII patterns
        self.pii_patterns = {
            "SSN": [
                r'\b\d{3}-\d{2}-\d{4}\b',  # XXX-XX-XXXX
                r'\b\d{3}\s\d{2}\s\d{4}\b',  # XXX XX XXXX
                r'\b\d{9}\b'  # XXXXXXXXX
            ],
            "CREDIT_CARD": [
                r'\b4\d{3}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Visa
                r'\b5[1-5]\d{2}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Mastercard
                r'\b3[47]\d{2}[\s-]?\d{6}[\s-]?\d{5}\b',  # Amex
            ],
            "EMAIL_ADDRESS": [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            "PHONE_NUMBER": [
                r'\b\(\d{3}\)\s?\d{3}-\d{4}\b',  # (XXX) XXX-XXXX
                r'\b\d{3}-\d{3}-\d{4}\b',  # XXX-XXX-XXXX
                r'\b\d{3}\.\d{3}\.\d{4}\b',  # XXX.XXX.XXXX
                r'\b\+1[\s-]?\d{3}[\s-]?\d{3}[\s-]?\d{4}\b'  # +1 XXX XXX XXXX
            ],
            "IP_ADDRESS": [
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ],
            "US_PASSPORT": [
                r'\b[A-Z]{1,2}\d{6,9}\b'
            ],
            "US_DRIVER_LICENSE": [
                r'\b[A-Z]{1,2}\d{6,8}\b',
                r'\b\d{8,9}\b'
            ],
            "PERSON": [
                # Simple name patterns - could be enhanced with NER
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+, [A-Z][a-z]+\b',  # Last, First
            ]
        }
        
        # Add custom patterns from config
        self.pii_patterns.update(self.config.custom_patterns)
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for entity_type, patterns in self.pii_patterns.items():
            self.compiled_patterns[entity_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def detect_pii(self, text: str) -> PIIDetectionResult:
        """Detect PII in the given text"""
        entities_found = []
        
        # Scan for each configured entity type
        for entity_type in self.config.entities_to_detect:
            if entity_type in self.compiled_patterns:
                entities = self._detect_entity_type(text, entity_type)
                entities_found.extend(entities)
        
        # Sort entities by position
        entities_found.sort(key=lambda e: e.start_pos)
        
        # Calculate confidence and risk
        confidence_score = self._calculate_detection_confidence(entities_found)
        risk_level = self._assess_risk_level(entities_found)
        
        # Generate redacted text
        redacted_text = self._redact_entities(text, entities_found)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(entities_found)
        
        return PIIDetectionResult(
            has_pii=len(entities_found) > 0,
            entities_found=entities_found,
            redacted_text=redacted_text,
            confidence_score=confidence_score,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def _detect_entity_type(self, text: str, entity_type: str) -> List[PIIEntity]:
        """Detect specific entity type in text"""
        entities = []
        
        if entity_type not in self.compiled_patterns:
            return entities
        
        for i, pattern in enumerate(self.compiled_patterns[entity_type]):
            for match in pattern.finditer(text):
                # Additional validation for certain entity types
                if self._validate_entity(match.group(), entity_type):
                    entity = PIIEntity(
                        entity_type=entity_type,
                        text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=self._calculate_pattern_confidence(entity_type, i),
                        pattern_name=f"{entity_type}_pattern_{i}"
                    )
                    entities.append(entity)
        
        return entities
    
    def _validate_entity(self, text: str, entity_type: str) -> bool:
        """Additional validation for detected entities"""
        if entity_type == "CREDIT_CARD":
            return self._validate_credit_card(text)
        elif entity_type == "SSN":
            return self._validate_ssn(text)
        elif entity_type == "EMAIL_ADDRESS":
            return self._validate_email(text)
        return True
    
    def _validate_credit_card(self, number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        # Remove non-digits
        number = re.sub(r'\D', '', number)
        
        if len(number) < 13 or len(number) > 19:
            return False
        
        # Luhn algorithm
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d * 2))
            return checksum % 10 == 0
        
        return luhn_check(number)
    
    def _validate_ssn(self, ssn: str) -> bool:
        """Basic SSN validation"""
        # Remove non-digits
        ssn = re.sub(r'\D', '', ssn)
        
        if len(ssn) != 9:
            return False
        
        # Check for invalid patterns
        if ssn == "000000000" or ssn.startswith("000") or ssn[3:5] == "00" or ssn[5:] == "0000":
            return False
        
        return True
    
    def _validate_email(self, email: str) -> bool:
        """Basic email validation"""
        return "@" in email and "." in email.split("@")[-1]
    
    def _calculate_pattern_confidence(self, entity_type: str, pattern_index: int) -> float:
        """Calculate confidence score for pattern match"""
        # Different patterns have different confidence levels
        confidence_map = {
            "SSN": [0.95, 0.90, 0.80],
            "CREDIT_CARD": [0.95, 0.95, 0.95],
            "EMAIL_ADDRESS": [0.98],
            "PHONE_NUMBER": [0.90, 0.90, 0.85, 0.85],
            "IP_ADDRESS": [0.85],
            "PERSON": [0.60, 0.65]  # Names are harder to detect accurately
        }
        
        if entity_type in confidence_map and pattern_index < len(confidence_map[entity_type]):
            return confidence_map[entity_type][pattern_index]
        
        return 0.75  # Default confidence
    
    def _calculate_detection_confidence(self, entities: List[PIIEntity]) -> float:
        """Calculate overall detection confidence"""
        if not entities:
            return 1.0  # High confidence in no PII detected
        
        # Average confidence of all detected entities, weighted by entity type sensitivity
        sensitivity_weights = {
            "SSN": 1.0,
            "CREDIT_CARD": 1.0,
            "EMAIL_ADDRESS": 0.7,
            "PHONE_NUMBER": 0.8,
            "IP_ADDRESS": 0.6,
            "PERSON": 0.5,
            "US_PASSPORT": 0.9,
            "US_DRIVER_LICENSE": 0.8
        }
        
        total_weight = 0
        weighted_confidence = 0
        
        for entity in entities:
            weight = sensitivity_weights.get(entity.entity_type, 0.7)
            weighted_confidence += entity.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.5
    
    def _assess_risk_level(self, entities: List[PIIEntity]) -> str:
        """Assess risk level based on detected entities"""
        if not entities:
            return "LOW"
        
        high_risk_types = {"SSN", "CREDIT_CARD", "US_PASSPORT"}
        medium_risk_types = {"PHONE_NUMBER", "US_DRIVER_LICENSE"}
        
        high_risk_count = sum(1 for e in entities if e.entity_type in high_risk_types)
        medium_risk_count = sum(1 for e in entities if e.entity_type in medium_risk_types)
        
        if high_risk_count > 0:
            return "HIGH"
        elif medium_risk_count > 2 or len(entities) > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _redact_entities(self, text: str, entities: List[PIIEntity]) -> str:
        """Redact detected PII entities from text"""
        if not entities:
            return text
        
        # Sort entities by position (reverse order to maintain positions)
        sorted_entities = sorted(entities, key=lambda e: e.start_pos, reverse=True)
        
        redacted_text = text
        for entity in sorted_entities:
            if self.config.preserve_format:
                # Preserve format by replacing with same length of redaction chars
                redacted_value = self._format_preserving_redaction(entity.text)
            else:
                redacted_value = f"[{entity.entity_type}]"
            
            entity.redacted_text = redacted_value
            redacted_text = (
                redacted_text[:entity.start_pos] + 
                redacted_value + 
                redacted_text[entity.end_pos:]
            )
        
        return redacted_text
    
    def _format_preserving_redaction(self, text: str) -> str:
        """Create format-preserving redaction"""
        redacted = ""
        for char in text:
            if char.isalnum():
                redacted += self.config.redaction_char
            else:
                redacted += char
        return redacted
    
    def _generate_recommendations(self, entities: List[PIIEntity]) -> List[str]:
        """Generate recommendations based on detected PII"""
        recommendations = []
        
        if not entities:
            return ["No PII detected - text appears safe for sharing"]
        
        entity_types = {e.entity_type for e in entities}
        
        if "SSN" in entity_types:
            recommendations.append("SSN detected - ensure this information is properly secured")
        
        if "CREDIT_CARD" in entity_types:
            recommendations.append("Credit card information detected - remove before sharing")
        
        if "EMAIL_ADDRESS" in entity_types:
            recommendations.append("Email addresses detected - consider if sharing is necessary")
        
        if len(entities) > 3:
            recommendations.append("Multiple PII entities detected - review content carefully")
        
        return recommendations


class PIIRedactor:
    """Handles PII redaction with various strategies"""
    
    def __init__(self, config: Optional[PIIDetectionConfig] = None):
        self.config = config or get_compliance_config().pii_detection
        self.detector = PIIDetector(config)
        self.logger = logging.getLogger(__name__)
    
    def process_text(self, text: str) -> Tuple[str, PIIDetectionResult]:
        """Process text according to PII detection configuration"""
        detection_result = self.detector.detect_pii(text)
        
        if not detection_result.has_pii:
            return text, detection_result
        
        # Handle based on action configuration
        action = self.config.action_on_detection
        
        if action == ActionType.BLOCK:
            raise ValueError("Text contains PII and is blocked by policy")
        elif action == ActionType.REDACT:
            return detection_result.redacted_text, detection_result
        elif action == ActionType.WARN:
            self.logger.warning(f"PII detected in text: {len(detection_result.entities_found)} entities")
            return text, detection_result
        else:  # LOG_ONLY
            self.logger.info(f"PII detected: {[e.entity_type for e in detection_result.entities_found]}")
            return text, detection_result
    
    def create_hash_replacement(self, text: str, entity_type: str) -> str:
        """Create a consistent hash-based replacement for PII"""
        # Create a hash that's consistent but not reversible
        hash_obj = hashlib.sha256(f"{entity_type}:{text}".encode())
        hash_hex = hash_obj.hexdigest()[:8]
        return f"[{entity_type}_{hash_hex}]"


# Example usage and testing
if __name__ == "__main__":
    # Test the PII detection system
    detector = PIIDetector()
    
    test_text = """
    Please send the report to john.doe@company.com and call me at (555) 123-4567.
    My SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
    """
    
    result = detector.detect_pii(test_text)
    print(f"PII Detected: {result.has_pii}")
    print(f"Entities: {[(e.entity_type, e.text) for e in result.entities_found]}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Redacted Text: {result.redacted_text}")