"""
Topic-Based Content Filtering System

Classifies content by topic and enforces allow/block lists based on
enterprise content policies and compliance requirements.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import json

from ..config.compliance_config import get_compliance_config, TopicFilterConfig, ActionType


@dataclass
class TopicClassification:
    """Represents a topic classification result"""
    topic: str
    confidence: float
    keywords_matched: List[str]
    category: str
    risk_level: str


@dataclass
class TopicFilterResult:
    """Result of topic filtering"""
    is_allowed: bool
    primary_topic: Optional[TopicClassification]
    all_topics: List[TopicClassification]
    blocked_topics: List[str]
    violations: List[str]
    recommendations: List[str]
    action_taken: str


class TopicClassifier:
    """Classifies text content by topic using keyword-based and rule-based approaches"""
    
    def __init__(self, config: Optional[TopicFilterConfig] = None):
        self.config = config or get_compliance_config().topic_filter
        self.logger = logging.getLogger(__name__)
        
        # Topic definitions with keywords and patterns
        self.topic_definitions = {
            "personal_finances": {
                "keywords": [
                    "salary", "income", "tax", "investment", "401k", "retirement",
                    "debt", "loan", "mortgage", "credit score", "bankruptcy",
                    "financial advisor", "portfolio", "stocks", "bonds"
                ],
                "patterns": [
                    r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Dollar amounts
                    r'\b\d+%\s*(?:interest|return|yield)\b',  # Interest rates
                ],
                "category": "financial",
                "risk_level": "high"
            },
            "medical_advice": {
                "keywords": [
                    "diagnosis", "treatment", "medication", "prescription", "dosage",
                    "symptoms", "disease", "illness", "medical condition", "doctor",
                    "hospital", "surgery", "therapy", "medicine", "drug"
                ],
                "patterns": [
                    r'\b\d+\s*mg\b',  # Medication dosages
                    r'\bdr\.?\s+[a-z]+\b',  # Doctor names
                ],
                "category": "medical",
                "risk_level": "high"
            },
            "legal_advice": {
                "keywords": [
                    "lawsuit", "legal action", "attorney", "lawyer", "court",
                    "contract", "liability", "damages", "settlement", "litigation",
                    "compliance", "regulation", "violation", "penalty"
                ],
                "patterns": [
                    r'\bcase\s+no\.?\s+\d+\b',  # Case numbers
                    r'\b\w+\s+v\.?\s+\w+\b',  # Legal case format
                ],
                "category": "legal",
                "risk_level": "high"
            },
            "political_opinions": {
                "keywords": [
                    "political party", "election", "candidate", "vote", "campaign",
                    "democrat", "republican", "liberal", "conservative", "policy",
                    "government", "president", "congress", "senate"
                ],
                "patterns": [
                    r'\b(?:trump|biden|harris|obama)\b',  # Political figures
                ],
                "category": "political",
                "risk_level": "medium"
            },
            "confidential_strategy": {
                "keywords": [
                    "confidential", "proprietary", "trade secret", "competitive advantage",
                    "strategy", "roadmap", "acquisition", "merger", "partnership",
                    "internal only", "restricted", "classified"
                ],
                "patterns": [
                    r'\b(?:confidential|restricted|internal)\s+(?:only|use)\b',
                ],
                "category": "business",
                "risk_level": "high"
            },
            "hr_sensitive": {
                "keywords": [
                    "performance review", "disciplinary action", "termination",
                    "harassment", "discrimination", "grievance", "complaint",
                    "salary negotiation", "promotion", "demotion"
                ],
                "patterns": [],
                "category": "hr",
                "risk_level": "high"
            },
            "technical_documentation": {
                "keywords": [
                    "api", "documentation", "technical", "implementation", "code",
                    "system", "architecture", "database", "server", "configuration"
                ],
                "patterns": [
                    r'\bapi\s+endpoint\b',
                    r'\b(?:get|post|put|delete)\s+/\w+\b',
                ],
                "category": "technical",
                "risk_level": "low"
            },
            "general_business": {
                "keywords": [
                    "meeting", "project", "timeline", "deliverable", "milestone",
                    "team", "department", "budget", "resource", "planning"
                ],
                "patterns": [],
                "category": "business",
                "risk_level": "low"
            }
        }
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for each topic"""
        for topic_data in self.topic_definitions.values():
            topic_data["compiled_patterns"] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in topic_data["patterns"]
            ]
    
    def classify_text(self, text: str) -> List[TopicClassification]:
        """Classify text into one or more topics"""
        text_lower = text.lower()
        classifications = []
        
        for topic, definition in self.topic_definitions.items():
            # Count keyword matches
            keywords_matched = []
            keyword_score = 0
            
            for keyword in definition["keywords"]:
                if keyword.lower() in text_lower:
                    keywords_matched.append(keyword)
                    keyword_score += 1
            
            # Count pattern matches
            pattern_score = 0
            for pattern in definition.get("compiled_patterns", []):
                matches = pattern.findall(text)
                pattern_score += len(matches)
                if matches:
                    keywords_matched.extend([f"pattern:{pattern.pattern}" for _ in matches])
            
            # Calculate confidence score
            total_score = keyword_score + (pattern_score * 2)  # Patterns weighted higher
            text_words = len(text.split())
            confidence = min(total_score / max(text_words * 0.1, 1), 1.0)  # Normalize by text length
            
            # Only include topics above threshold
            if confidence >= self.config.topic_detection_threshold * 0.5:  # Lower threshold for detection
                classification = TopicClassification(
                    topic=topic,
                    confidence=confidence,
                    keywords_matched=keywords_matched,
                    category=definition["category"],
                    risk_level=definition["risk_level"]
                )
                classifications.append(classification)
        
        # Sort by confidence
        classifications.sort(key=lambda c: c.confidence, reverse=True)
        return classifications
    
    def get_primary_topic(self, classifications: List[TopicClassification]) -> Optional[TopicClassification]:
        """Get the primary (highest confidence) topic"""
        if not classifications:
            return None
        
        # Filter by detection threshold
        valid_classifications = [
            c for c in classifications 
            if c.confidence >= self.config.topic_detection_threshold
        ]
        
        return valid_classifications[0] if valid_classifications else None


class TopicFilter:
    """Filters content based on topic allow/block lists"""
    
    def __init__(self, config: Optional[TopicFilterConfig] = None):
        self.config = config or get_compliance_config().topic_filter
        self.classifier = TopicClassifier(config)
        self.logger = logging.getLogger(__name__)
    
    def filter_content(self, text: str, context: Optional[Dict] = None) -> TopicFilterResult:
        """Filter content based on topic classification"""
        if not self.config.enabled:
            return TopicFilterResult(
                is_allowed=True,
                primary_topic=None,
                all_topics=[],
                blocked_topics=[],
                violations=[],
                recommendations=[],
                action_taken="none"
            )
        
        # Classify the text
        classifications = self.classifier.classify_text(text)
        primary_topic = self.classifier.get_primary_topic(classifications)
        
        # Check against block and allow lists
        blocked_topics = []
        violations = []
        recommendations = []
        
        for classification in classifications:
            if classification.topic in self.config.blocked_topics:
                blocked_topics.append(classification.topic)
                violations.append(
                    f"Content classified as '{classification.topic}' "
                    f"(confidence: {classification.confidence:.2f}) is blocked by policy"
                )
        
        # If allow list is specified, check that content matches
        if self.config.allowed_topics:
            allowed_classifications = [
                c for c in classifications 
                if c.topic in self.config.allowed_topics
            ]
            if not allowed_classifications and classifications:
                violations.append("Content does not match any allowed topics")
        
        # Determine if content is allowed
        is_allowed = len(violations) == 0
        
        # Determine action to take
        action_taken = self._determine_action(is_allowed, blocked_topics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(classifications, blocked_topics)
        
        return TopicFilterResult(
            is_allowed=is_allowed,
            primary_topic=primary_topic,
            all_topics=classifications,
            blocked_topics=blocked_topics,
            violations=violations,
            recommendations=recommendations,
            action_taken=action_taken
        )
    
    def _determine_action(self, is_allowed: bool, blocked_topics: List[str]) -> str:
        """Determine what action to take based on filter result"""
        if is_allowed:
            return "allow"
        
        action = self.config.action_on_blocked_topic
        
        if action == ActionType.BLOCK:
            return "block"
        elif action == ActionType.REDACT:
            return "redact"
        elif action == ActionType.WARN:
            return "warn"
        else:  # LOG_ONLY
            return "log"
    
    def _generate_recommendations(self, 
                                 classifications: List[TopicClassification],
                                 blocked_topics: List[str]) -> List[str]:
        """Generate recommendations based on topic analysis"""
        recommendations = []
        
        if not classifications:
            recommendations.append("Content appears to be general - no specific topic detected")
            return recommendations
        
        if blocked_topics:
            recommendations.append(
                f"Content contains blocked topics: {', '.join(blocked_topics)}. "
                "Consider rephrasing or removing sensitive content."
            )
        
        high_risk_topics = [c.topic for c in classifications if c.risk_level == "high"]
        if high_risk_topics:
            recommendations.append(
                f"High-risk topics detected: {', '.join(high_risk_topics)}. "
                "Review content carefully before sharing."
            )
        
        # Topic-specific recommendations
        for classification in classifications:
            if classification.topic == "medical_advice":
                recommendations.append(
                    "Medical content detected. Ensure you're not providing specific medical advice."
                )
            elif classification.topic == "legal_advice":
                recommendations.append(
                    "Legal content detected. Consider adding disclaimer about not providing legal advice."
                )
            elif classification.topic == "personal_finances":
                recommendations.append(
                    "Financial content detected. Verify no personal financial information is exposed."
                )
        
        return recommendations
    
    def add_blocked_topic(self, topic: str):
        """Add a topic to the blocked list"""
        if topic not in self.config.blocked_topics:
            self.config.blocked_topics.append(topic)
            self.logger.info(f"Added '{topic}' to blocked topics list")
    
    def remove_blocked_topic(self, topic: str):
        """Remove a topic from the blocked list"""
        if topic in self.config.blocked_topics:
            self.config.blocked_topics.remove(topic)
            self.logger.info(f"Removed '{topic}' from blocked topics list")
    
    def add_allowed_topic(self, topic: str):
        """Add a topic to the allowed list"""
        if topic not in self.config.allowed_topics:
            self.config.allowed_topics.append(topic)
            self.logger.info(f"Added '{topic}' to allowed topics list")
    
    def get_topic_definitions(self) -> Dict:
        """Get all available topic definitions"""
        return {
            topic: {
                "keywords": definition["keywords"],
                "category": definition["category"],
                "risk_level": definition["risk_level"]
            }
            for topic, definition in self.classifier.topic_definitions.items()
        }
    
    def update_topic_keywords(self, topic: str, keywords: List[str]):
        """Update keywords for a specific topic"""
        if topic in self.classifier.topic_definitions:
            self.classifier.topic_definitions[topic]["keywords"] = keywords
            self.logger.info(f"Updated keywords for topic '{topic}'")


# Example usage and testing
if __name__ == "__main__":
    # Test the topic filtering system
    topic_filter = TopicFilter()
    
    test_texts = [
        "Can you help me with my tax return and investment portfolio?",
        "What medication should I take for my headache?",
        "Our API documentation shows the GET /users endpoint returns user data.",
        "The company's acquisition strategy is highly confidential.",
        "Let's schedule a team meeting to discuss the project timeline."
    ]
    
    for text in test_texts:
        result = topic_filter.filter_content(text)
        print(f"\nText: {text[:50]}...")
        print(f"Allowed: {result.is_allowed}")
        if result.primary_topic:
            print(f"Primary Topic: {result.primary_topic.topic} ({result.primary_topic.confidence:.2f})")
        print(f"Violations: {result.violations}")
        print(f"Action: {result.action_taken}")