#!/usr/bin/env python3
"""
Source Verification & Hallucination Detection System

Multi-layer verification to detect when LLM answers don't match provided sources.

Layers:
1. Citation Validation - Check cited sources exist
2. Content Verification - Verify answer matches sources
3. Hallucination Detection - Detect fabricated information
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import re
from difflib import SequenceMatcher


@dataclass
class VerificationResult:
    """Result of source verification"""
    is_verified: bool
    confidence: float
    issues: List[str]
    source_support: Dict[int, float]  # source_num -> support_score
    hallucination_risk: float  # 0.0 = low, 1.0 = high
    recommendation: str

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        return {
            'is_verified': self.is_verified,
            'confidence': self.confidence,
            'issues': self.issues,
            'source_support': {str(k): v for k, v in self.source_support.items()},
            'hallucination_risk': self.hallucination_risk,
            'recommendation': self.recommendation
        }


class CitationValidator:
    """
    Validate that citations in answer match provided sources

    Checks:
    1. All cited source numbers exist
    2. Cited sources actually support the claims
    3. No phantom citations
    """

    def __init__(self):
        # Patterns to detect citations: [1], [2], source [3], etc.
        self.citation_patterns = [
            r'\[(\d+)\]',           # [1], [2]
            r'source\s+\[(\d+)\]',  # source [1]
            r'sources?\s+(\d+)',    # source 1, sources 1
        ]

    def validate(self, answer: str, num_sources: int) -> Dict:
        """
        Validate citations in answer

        Returns:
            {
                'valid': bool,
                'cited_sources': List[int],
                'invalid_citations': List[int],
                'missing_sources': List[int],
                'issues': List[str]
            }
        """
        # Extract all citation numbers
        cited = set()
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, answer, re.IGNORECASE)
            cited.update(int(m) for m in matches)

        # Check validity
        valid_sources = set(range(1, num_sources + 1))
        invalid = cited - valid_sources
        missing = valid_sources - cited  # Sources not cited

        issues = []

        # Flag invalid citations
        if invalid:
            issues.append(
                f"Answer cites non-existent sources: {sorted(invalid)}"
            )

        # Flag if answer cites very few sources
        if len(cited) < max(1, num_sources // 3):
            issues.append(
                f"Answer only cites {len(cited)}/{num_sources} sources"
            )

        return {
            'valid': len(invalid) == 0,
            'cited_sources': sorted(cited),
            'invalid_citations': sorted(invalid),
            'missing_sources': sorted(missing),
            'issues': issues
        }


class ContentVerifier:
    """
    Check if answer content is supported by provided sources

    Methods:
    1. Extract claims from answer
    2. Check each claim against sources
    3. Calculate support score
    """

    def __init__(self, min_similarity: float = 0.6):
        self.min_similarity = min_similarity

    def verify(
        self,
        answer: str,
        sources: List[Dict],
        cited_source_nums: List[int]
    ) -> Dict:
        """
        Verify answer content against sources

        Returns:
            {
                'supported': bool,
                'support_score': float,  # 0.0-1.0
                'unsupported_claims': List[str],
                'source_alignment': Dict[int, float]  # source_num -> relevance
            }
        """
        # Split answer into claims (sentences)
        claims = self._extract_claims(answer)

        # Check each claim against sources
        claim_support = []
        unsupported = []

        for claim in claims:
            # Check if claim is supported
            support = self._check_claim_support(
                claim,
                sources,
                cited_source_nums
            )

            claim_support.append(support)

            if support < self.min_similarity:
                unsupported.append(claim)

        # Calculate overall support score
        avg_support = sum(claim_support) / len(claim_support) if claim_support else 0

        # Calculate per-source alignment
        source_alignment = {}
        for source_num in cited_source_nums:
            if 1 <= source_num <= len(sources):
                source = sources[source_num - 1]
                content = source.get('text_preview', source.get('content_preview', ''))
                alignment = self._calculate_alignment(answer, content)
                source_alignment[source_num] = alignment

        return {
            'supported': avg_support >= self.min_similarity,
            'support_score': avg_support,
            'unsupported_claims': unsupported,
            'source_alignment': source_alignment
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract individual claims from answer

        Simple approach: split by sentences
        """
        # Remove citations from claims
        clean_answer = re.sub(r'\[?\d+\]?', '', answer)

        # Split by sentence
        sentences = re.split(r'[.!?]+', clean_answer)

        # Filter out very short sentences
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]

        return claims

    def _check_claim_support(
        self,
        claim: str,
        sources: List[Dict],
        cited_sources: List[int]
    ) -> float:
        """
        Check if claim is supported by any cited source

        Returns: max similarity score across cited sources
        """
        max_support = 0.0

        for source_num in cited_sources:
            if 1 <= source_num <= len(sources):
                source = sources[source_num - 1]
                content = source.get('text_preview', source.get('content_preview', ''))
                similarity = self._text_similarity(claim, content)
                max_support = max(max_support, similarity)

        return max_support

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts

        Uses sequence matching
        """
        # Normalize texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        # Quick check: substring match
        if text1 in text2 or text2 in text1:
            return 1.0

        # Sequence matcher
        matcher = SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _calculate_alignment(self, answer: str, source_content: str) -> float:
        """
        Calculate how well answer aligns with a specific source

        Higher score = answer draws more from this source
        """
        # Split into words
        answer_words = set(answer.lower().split())
        source_words = set(source_content.lower().split())

        # Calculate word overlap
        overlap = answer_words & source_words

        if not answer_words:
            return 0.0

        return len(overlap) / len(answer_words)


class HallucinationDetector:
    """
    Detect various types of hallucinations

    Types:
    1. Factual errors (inventing information)
    2. Reasoning errors (wrong conclusions)
    3. Source misattribution (wrong citation)
    4. Confidence mismatch (uncertain sources, confident answer)
    """

    def __init__(self):
        # Patterns that indicate potential hallucination
        self.red_flags = [
            r'according to (?:source )?\[(\d+)\]',  # Check these carefully
            r'(?:clearly|obviously|definitely)',    # Overconfidence
            r'(?:all|every|none|never)',           # Absolute claims
            r'\d+%',                                # Specific numbers
            r'\d{4}',                               # Years/dates
        ]

        # Uncertainty phrases (good - shows LLM is careful)
        self.uncertainty_phrases = [
            'might', 'may', 'possibly', 'likely', 'suggests',
            'appears', 'seems', 'could be', 'based on', 'indicates'
        ]

    def detect(
        self,
        answer: str,
        sources: List[Dict],
        verification_result: Dict
    ) -> Dict:
        """
        Detect hallucination risk

        Returns:
            {
                'risk_level': str,  # 'low', 'medium', 'high'
                'risk_score': float,  # 0.0-1.0
                'red_flags': List[str],
                'issues': List[str]
            }
        """
        risk_score = 0.0
        issues = []
        red_flags_found = []

        # Check 1: Invalid citations (major red flag)
        if verification_result.get('invalid_citations'):
            risk_score += 0.4
            issues.append("Answer cites sources that don't exist")
            red_flags_found.append("phantom_citations")

        # Check 2: Low source support
        support_score = verification_result.get('support_score', 1.0)
        if support_score < 0.6:
            risk_score += 0.3
            issues.append("Answer content not well-supported by sources")
            red_flags_found.append("weak_support")

        # Check 3: Overconfident language with weak sources
        has_overconfidence = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in self.red_flags[1:3]  # Confidence patterns
        )

        if has_overconfidence and support_score < 0.7:
            risk_score += 0.2
            issues.append("Overconfident language with weak source support")
            red_flags_found.append("overconfidence")

        # Check 4: Specific numbers without strong citations
        has_numbers = bool(re.search(r'\d+(?:\.\d+)?%?', answer))
        if has_numbers and len(verification_result.get('cited_sources', [])) < 2:
            risk_score += 0.1
            issues.append("Specific numbers cited with limited sources")
            red_flags_found.append("unsupported_numbers")

        # Check 5: No uncertainty phrases (LLM might be making things up)
        has_uncertainty = any(phrase in answer.lower() for phrase in self.uncertainty_phrases)
        if not has_uncertainty and support_score < 0.8:
            risk_score += 0.1
            red_flags_found.append("no_uncertainty")

        # Determine risk level
        if risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'

        return {
            'risk_level': risk_level,
            'risk_score': min(risk_score, 1.0),
            'red_flags': red_flags_found,
            'issues': issues
        }


class SourceVerificationSystem:
    """
    Complete source verification system

    Integrates all verification layers to produce final judgment
    """

    def __init__(self):
        self.citation_validator = CitationValidator()
        self.content_verifier = ContentVerifier()
        self.hallucination_detector = HallucinationDetector()

    def verify(
        self,
        answer: str,
        sources: List[Dict],
        original_query: str = None
    ) -> VerificationResult:
        """
        Complete verification pipeline

        Returns: VerificationResult with all verification details
        """
        num_sources = len(sources)

        # Layer 1: Validate citations
        citation_result = self.citation_validator.validate(answer, num_sources)

        # Layer 2: Verify content
        content_result = self.content_verifier.verify(
            answer,
            sources,
            citation_result['cited_sources']
        )

        # Layer 3: Detect hallucinations
        verification_data = {
            **citation_result,
            **content_result
        }

        hallucination_result = self.hallucination_detector.detect(
            answer,
            sources,
            verification_data
        )

        # Combine results
        is_verified = (
            citation_result['valid'] and
            content_result['supported'] and
            hallucination_result['risk_level'] != 'high'
        )

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            citation_result,
            content_result,
            hallucination_result
        )

        # Collect all issues
        issues = []
        issues.extend(citation_result.get('issues', []))
        issues.extend(hallucination_result.get('issues', []))

        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_verified,
            hallucination_result['risk_level'],
            content_result['support_score']
        )

        return VerificationResult(
            is_verified=is_verified,
            confidence=confidence,
            issues=issues,
            source_support=content_result.get('source_alignment', {}),
            hallucination_risk=hallucination_result['risk_score'],
            recommendation=recommendation
        )

    def _calculate_overall_confidence(
        self,
        citation_result: Dict,
        content_result: Dict,
        hallucination_result: Dict
    ) -> float:
        """Calculate final confidence score (0.0-1.0)"""
        # Start with content support score
        confidence = content_result['support_score']

        # Penalize for citation issues
        if citation_result['invalid_citations']:
            confidence *= 0.5

        # Penalize for hallucination risk
        confidence *= (1.0 - hallucination_result['risk_score'])

        return max(0.0, min(1.0, confidence))

    def _generate_recommendation(
        self,
        is_verified: bool,
        risk_level: str,
        support_score: float
    ) -> str:
        """Generate user-facing recommendation"""
        if is_verified and support_score > 0.8:
            return "✅ Answer is well-supported by sources"

        elif is_verified:
            return "⚠️ Answer is acceptable but consider reviewing sources"

        elif risk_level == 'high':
            return "❌ High risk of hallucination - answer may be unreliable"

        elif risk_level == 'medium':
            return "⚠️ Medium verification risk - verify important claims"

        else:
            return "⚠️ Answer needs stronger source support"
