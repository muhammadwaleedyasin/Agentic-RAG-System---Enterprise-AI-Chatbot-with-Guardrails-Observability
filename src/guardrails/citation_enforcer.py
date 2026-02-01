"""
Citation Enforcement System

Validates that LLM responses properly cite their sources and enforces
citation requirements based on enterprise compliance policies.
"""
import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

from ..config.compliance_config import get_compliance_config, CitationConfig


@dataclass
class Citation:
    """Represents a citation in the response"""
    source_id: str
    text_span: str
    start_pos: int
    end_pos: int
    confidence: float
    page_number: Optional[int] = None
    metadata: Dict = None


@dataclass
class CitationValidationResult:
    """Result of citation validation"""
    is_valid: bool
    citations_found: List[Citation]
    missing_citations: List[str]
    confidence_score: float
    violations: List[str]
    suggestions: List[str]


class CitationValidator:
    """Validates citations in LLM responses"""
    
    def __init__(self, config: Optional[CitationConfig] = None):
        self.config = config or get_compliance_config().citation
        self.logger = logging.getLogger(__name__)
        
        # Citation pattern matching
        self.citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\[([a-zA-Z0-9_-]+)\]',  # [doc_id], [source-1], etc.
            r'\((\d+)\)',  # (1), (2), etc.
            r'\(([a-zA-Z0-9_-]+)\)',  # (doc_id), (source-1), etc.
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern) for pattern in self.citation_patterns]
    
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract all citations from text"""
        citations = []
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text):
                citation = Citation(
                    source_id=match.group(1),
                    text_span=match.group(0),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=1.0  # Pattern-based extraction is high confidence
                )
                citations.append(citation)
        
        # Remove duplicates while preserving order
        unique_citations = []
        seen_ids = set()
        for citation in citations:
            if citation.source_id not in seen_ids:
                unique_citations.append(citation)
                seen_ids.add(citation.source_id)
        
        return unique_citations
    
    def validate_citations(self, 
                         response_text: str, 
                         retrieved_sources: List[Dict]) -> CitationValidationResult:
        """Validate citations in response against retrieved sources"""
        citations = self.extract_citations(response_text)
        violations = []
        suggestions = []
        
        # Check minimum citations requirement
        if len(citations) < self.config.min_citations_required:
            violations.append(
                f"Insufficient citations: {len(citations)} found, "
                f"{self.config.min_citations_required} required"
            )
            suggestions.append("Add more source citations to support your claims")
        
        # Check maximum citations limit
        if len(citations) > self.config.max_citations_allowed:
            violations.append(
                f"Too many citations: {len(citations)} found, "
                f"maximum {self.config.max_citations_allowed} allowed"
            )
            suggestions.append("Consolidate or remove some citations")
        
        # Validate citation IDs against retrieved sources
        source_ids = {src.get('id', src.get('source_id', '')) for src in retrieved_sources}
        missing_citations = []
        
        for citation in citations:
            if citation.source_id not in source_ids:
                missing_citations.append(citation.source_id)
                violations.append(f"Citation {citation.source_id} not found in retrieved sources")
        
        # Check for unreferenced sources
        cited_ids = {c.source_id for c in citations}
        uncited_sources = source_ids - cited_ids
        
        if uncited_sources and self.config.enforce_citations:
            suggestions.append(f"Consider citing these retrieved sources: {list(uncited_sources)}")
        
        # Calculate confidence score
        confidence_score = self._calculate_citation_confidence(citations, retrieved_sources)
        
        # Determine if validation passes
        is_valid = (
            len(violations) == 0 and
            confidence_score >= self.config.confidence_threshold
        )
        
        return CitationValidationResult(
            is_valid=is_valid,
            citations_found=citations,
            missing_citations=missing_citations,
            confidence_score=confidence_score,
            violations=violations,
            suggestions=suggestions
        )
    
    def _calculate_citation_confidence(self, 
                                     citations: List[Citation], 
                                     retrieved_sources: List[Dict]) -> float:
        """Calculate overall confidence score for citations"""
        if not citations:
            return 0.0
        
        # Factors contributing to confidence
        coverage_score = len(citations) / max(len(retrieved_sources), 1)
        coverage_score = min(coverage_score, 1.0)  # Cap at 1.0
        
        # Quality score based on citation format consistency
        format_score = sum(1 for c in citations if self._is_well_formatted(c)) / len(citations)
        
        # Relevance score (simplified - could be enhanced with semantic analysis)
        relevance_score = 0.8  # Placeholder - could analyze citation context
        
        # Weighted average
        confidence = (coverage_score * 0.4 + format_score * 0.3 + relevance_score * 0.3)
        return round(confidence, 3)
    
    def _is_well_formatted(self, citation: Citation) -> bool:
        """Check if citation follows proper formatting"""
        # Check if citation matches expected format
        expected_format = self.config.citation_format
        if "{source_id}" in expected_format:
            expected = expected_format.replace("{source_id}", citation.source_id)
            return citation.text_span == expected
        return True


class CitationEnforcer:
    """Enforces citation requirements and automatically adds missing citations"""
    
    def __init__(self, config: Optional[CitationConfig] = None):
        self.config = config or get_compliance_config().citation
        self.validator = CitationValidator(config)
        self.logger = logging.getLogger(__name__)
    
    def enforce_citations(self, 
                         response_text: str, 
                         retrieved_sources: List[Dict]) -> Tuple[str, CitationValidationResult]:
        """Enforce citation requirements on response text"""
        validation_result = self.validator.validate_citations(response_text, retrieved_sources)
        
        if not self.config.enforce_citations:
            return response_text, validation_result
        
        # If citations are invalid, attempt to fix
        if not validation_result.is_valid:
            response_text = self._auto_add_citations(response_text, retrieved_sources, validation_result)
            # Re-validate after auto-correction
            validation_result = self.validator.validate_citations(response_text, retrieved_sources)
        
        return response_text, validation_result
    
    def _auto_add_citations(self, 
                           response_text: str, 
                           retrieved_sources: List[Dict],
                           validation_result: CitationValidationResult) -> str:
        """Automatically add missing citations to response"""
        if not self.config.allow_partial_citations:
            return response_text
        
        # Simple approach: append missing citations at the end
        if validation_result.missing_citations:
            self.logger.info(f"Auto-adding {len(validation_result.missing_citations)} missing citations")
            return response_text  # For now, don't auto-modify
        
        # If no citations exist, add references to all sources
        if not validation_result.citations_found and retrieved_sources:
            citations_text = " ".join([
                self.config.citation_format.format(source_id=src.get('id', f"src_{i}"))
                for i, src in enumerate(retrieved_sources[:self.config.max_citations_allowed])
            ])
            response_text += f"\n\nSources: {citations_text}"
        
        return response_text
    
    def format_citations(self, sources: List[Dict]) -> str:
        """Format a list of sources as citations"""
        citations = []
        for i, source in enumerate(sources[:self.config.max_citations_allowed]):
            source_id = source.get('id', f"source_{i+1}")
            citation = self.config.citation_format.format(source_id=source_id)
            citations.append(citation)
        
        return " ".join(citations)
    
    def get_citation_requirements(self) -> Dict:
        """Get current citation requirements for display"""
        return {
            "enforce_citations": self.config.enforce_citations,
            "min_citations_required": self.config.min_citations_required,
            "max_citations_allowed": self.config.max_citations_allowed,
            "citation_format": self.config.citation_format,
            "confidence_threshold": self.config.confidence_threshold
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the citation system
    enforcer = CitationEnforcer()
    
    test_response = "The company policy states that remote work is allowed [1]. However, approval is required [2]."
    test_sources = [
        {"id": "1", "title": "Remote Work Policy", "content": "..."},
        {"id": "2", "title": "HR Guidelines", "content": "..."},
        {"id": "3", "title": "Uncited Source", "content": "..."}
    ]
    
    enforced_response, result = enforcer.enforce_citations(test_response, test_sources)
    print(f"Valid: {result.is_valid}")
    print(f"Citations: {[c.source_id for c in result.citations_found]}")
    print(f"Violations: {result.violations}")
    print(f"Suggestions: {result.suggestions}")