#!/usr/bin/env python3
"""
Metadata-Driven Retrieval

Provides intelligent filtering and boosting based on chunk metadata:
- Query analysis to extract language, domain, topic hints
- Pre-filtering using metadata
- Score boosting based on metadata matches
- Context-aware retrieval strategies
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class RetrievalStrategy(Enum):
    """Retrieval strategy based on query context"""
    GENERAL = "general"  # No specific filtering
    CODE_SEARCH = "code_search"  # Filter by language, boost complete functions
    DOMAIN_SPECIFIC = "domain_specific"  # Filter by domain/topic
    EXAMPLE_SEARCH = "example_search"  # Prioritize chunks with examples
    DOCUMENTATION = "documentation"  # Prioritize documented code
    SECURITY_RESEARCH = "security_research"  # Focus on security patterns


@dataclass
class QueryMetadata:
    """Extracted metadata from user query"""

    # Language hints
    languages: List[str] = field(default_factory=list)
    language_confidence: float = 0.0

    # Domain hints
    domains: List[str] = field(default_factory=list)
    domain_confidence: float = 0.0

    # Topic hints
    topics: List[str] = field(default_factory=list)
    topic_confidence: float = 0.0

    # Code patterns
    code_patterns: List[str] = field(default_factory=list)

    # Quality preferences
    prefer_complete: bool = False  # Prefer complete functions
    prefer_examples: bool = False  # Prefer chunks with examples
    prefer_documented: bool = False  # Prefer documented code
    prefer_simple: bool = False  # Prefer simple complexity
    prefer_complex: bool = False  # Prefer complex code

    # Retrieval strategy
    strategy: RetrievalStrategy = RetrievalStrategy.GENERAL


class QueryAnalyzer:
    """
    Analyze user queries to extract metadata hints

    Examples:
    - "C buffer overflow" → language=c, domains=[security], topics=[buffer_overflow]
    - "Python neural network example" → language=python, domains=[ai_ml], prefer_examples=True
    - "Rust concurrency with documentation" → language=rust, topics=[concurrency], prefer_documented=True
    """

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        'python': [r'\bpython\b', r'\.py\b', r'\bpyth\b'],
        'c': [r'\bc\b', r'\bc code\b', r'\.c\b'],
        'cpp': [r'\bc\+\+\b', r'\bcpp\b', r'\.cpp\b'],
        'rust': [r'\brust\b', r'\.rs\b'],
        'ruby': [r'\bruby\b', r'\.rb\b'],
        'gdscript': [r'\bgdscript\b', r'\bgodot\b', r'\.gd\b'],
        'javascript': [r'\bjavascript\b', r'\bjs\b', r'\.js\b'],
        'bash': [r'\bbash\b', r'\bshell\b', r'\.sh\b'],
        'go': [r'\bgolang\b', r'\bgo\b', r'\.go\b'],
        'java': [r'\bjava\b', r'\.java\b'],
    }

    # Domain detection patterns
    DOMAIN_PATTERNS = {
        'security': [
            r'\bsecurity\b', r'\bexploit\b', r'\bvulnerability\b',
            r'\bbuffer overflow\b', r'\boverflow\b', r'\binjection\b',
            r'\bcrypto\b', r'\bencrypt\b', r'\bhash\b', r'\bauth\b',
        ],
        'systems': [
            r'\bmemory\b', r'\bkernel\b', r'\bprocess\b', r'\bthread\b',
            r'\ballocation\b', r'\bsyscall\b', r'\bos\b',
        ],
        'networking': [
            r'\bnetwork\b', r'\bsocket\b', r'\btcp\b', r'\budp\b',
            r'\bhttp\b', r'\bapi\b', r'\bprotocol\b',
        ],
        'physics': [
            r'\bphysics\b', r'\bsimulation\b', r'\bcollision\b',
            r'\bvelocity\b', r'\bforce\b', r'\bgravity\b',
        ],
        'neuroscience': [
            r'\bneuron\b', r'\bsynapse\b', r'\bbrain\b', r'\bspike\b',
            r'\bmembrane\b', r'\bplasticity\b',
        ],
        'ai_ml': [
            r'\bneural network\b', r'\bdeep learning\b', r'\bmachine learning\b',
            r'\btraining\b', r'\bmodel\b', r'\bgradient\b', r'\bai\b', r'\bml\b',
        ],
        'graphics': [
            r'\bgraphics\b', r'\brender\b', r'\bshader\b', r'\bopengl\b',
            r'\bvulkan\b', r'\btexture\b', r'\bgpu\b',
        ],
        'web': [
            r'\bweb\b', r'\bhttp\b', r'\brest\b', r'\bapi\b',
            r'\bjson\b', r'\bbackend\b', r'\bfrontend\b',
        ],
    }

    # Topic detection patterns
    TOPIC_PATTERNS = {
        'buffer_overflow': [r'\bbuffer overflow\b', r'\boverflow\b'],
        'cryptography': [r'\bcrypto\b', r'\bencrypt\b', r'\bcipher\b', r'\bhash\b'],
        'memory_management': [r'\bmemory\b', r'\balloc\b', r'\bfree\b', r'\bmalloc\b'],
        'concurrency': [r'\bconcurrency\b', r'\bthread\b', r'\bmutex\b', r'\bparallel\b'],
        'algorithms': [r'\balgorithm\b', r'\bsort\b', r'\bsearch\b', r'\bgraph\b'],
    }

    # Code pattern hints
    CODE_PATTERN_KEYWORDS = {
        'unsafe_string_operation': [r'\bstrcpy\b', r'\bstrcat\b', r'\bunsafe string\b'],
        'manual_memory_management': [r'\bmalloc\b', r'\bfree\b', r'\bmemory\b'],
        'concurrent_code': [r'\bthread\b', r'\bmutex\b', r'\bconcurrent\b'],
        'error_handling': [r'\berror\b', r'\bexception\b', r'\btry\b', r'\bcatch\b'],
        'optimization': [r'\boptimize\b', r'\bfast\b', r'\bperformance\b'],
    }

    # Quality preference keywords
    QUALITY_KEYWORDS = {
        'examples': [r'\bexample\b', r'\bdemo\b', r'\bsample\b', r'\bshowcase\b'],
        'documentation': [r'\bdocumented\b', r'\bdoc\b', r'\bcomment\b', r'\bexplain\b'],
        'complete': [r'\bcomplete\b', r'\bfull\b', r'\bentire\b', r'\bwhole\b'],
        'simple': [r'\bsimple\b', r'\bbasic\b', r'\beasy\b', r'\bstraightforward\b'],
        'complex': [r'\bcomplex\b', r'\badvanced\b', r'\bsophisticated\b'],
    }

    def __init__(self):
        self.logger = logging.getLogger("QueryAnalyzer")

    def analyze(self, query: str) -> QueryMetadata:
        """
        Analyze query and extract metadata hints

        Args:
            query: User's search query

        Returns:
            QueryMetadata with extracted hints
        """
        query_lower = query.lower()
        metadata = QueryMetadata()

        # 1. Detect languages
        metadata.languages, metadata.language_confidence = self._detect_languages(query_lower)

        # 2. Detect domains
        metadata.domains, metadata.domain_confidence = self._detect_domains(query_lower)

        # 3. Detect topics
        metadata.topics, metadata.topic_confidence = self._detect_topics(query_lower)

        # 4. Detect code patterns
        metadata.code_patterns = self._detect_code_patterns(query_lower)

        # 5. Detect quality preferences
        self._detect_quality_preferences(query_lower, metadata)

        # 6. Determine retrieval strategy
        metadata.strategy = self._determine_strategy(metadata, query_lower)

        self.logger.debug(
            f"Query analysis: languages={metadata.languages}, "
            f"domains={metadata.domains}, topics={metadata.topics}, "
            f"strategy={metadata.strategy.value}"
        )

        return metadata

    def _detect_languages(self, query: str) -> Tuple[List[str], float]:
        """Detect programming languages mentioned in query"""
        detected = []
        scores = {}

        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[lang] = score

        # Return languages sorted by score
        if scores:
            detected = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            confidence = min(max(scores.values()) / 2.0, 1.0)
            return detected, confidence

        return [], 0.0

    def _detect_domains(self, query: str) -> Tuple[List[str], float]:
        """Detect domains mentioned in query"""
        detected = []
        scores = {}

        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[domain] = score

        if scores:
            detected = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            confidence = min(max(scores.values()) / 3.0, 1.0)
            return detected, confidence

        return [], 0.0

    def _detect_topics(self, query: str) -> Tuple[List[str], float]:
        """Detect specific topics in query"""
        detected = []
        scores = {}

        for topic, patterns in self.TOPIC_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 2  # Topics are more specific, weight higher
            if score > 0:
                scores[topic] = score

        if scores:
            detected = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            confidence = min(max(scores.values()) / 4.0, 1.0)
            return detected, confidence

        return [], 0.0

    def _detect_code_patterns(self, query: str) -> List[str]:
        """Detect code patterns mentioned in query"""
        detected = []

        for pattern, keywords in self.CODE_PATTERN_KEYWORDS.items():
            for keyword in keywords:
                if re.search(keyword, query, re.IGNORECASE):
                    detected.append(pattern)
                    break

        return detected

    def _detect_quality_preferences(self, query: str, metadata: QueryMetadata):
        """Detect quality preferences in query"""

        # Check for example preference
        for keyword in self.QUALITY_KEYWORDS['examples']:
            if re.search(keyword, query, re.IGNORECASE):
                metadata.prefer_examples = True
                break

        # Check for documentation preference
        for keyword in self.QUALITY_KEYWORDS['documentation']:
            if re.search(keyword, query, re.IGNORECASE):
                metadata.prefer_documented = True
                break

        # Check for completeness preference
        for keyword in self.QUALITY_KEYWORDS['complete']:
            if re.search(keyword, query, re.IGNORECASE):
                metadata.prefer_complete = True
                break

        # Check for simplicity preference
        for keyword in self.QUALITY_KEYWORDS['simple']:
            if re.search(keyword, query, re.IGNORECASE):
                metadata.prefer_simple = True
                break

        # Check for complexity preference
        for keyword in self.QUALITY_KEYWORDS['complex']:
            if re.search(keyword, query, re.IGNORECASE):
                metadata.prefer_complex = True
                break

    def _determine_strategy(self, metadata: QueryMetadata, query: str) -> RetrievalStrategy:
        """Determine best retrieval strategy based on query"""

        # Security research
        if 'security' in metadata.domains or any(t in metadata.topics for t in ['buffer_overflow', 'cryptography']):
            return RetrievalStrategy.SECURITY_RESEARCH

        # Example search
        if metadata.prefer_examples:
            return RetrievalStrategy.EXAMPLE_SEARCH

        # Documentation search
        if metadata.prefer_documented:
            return RetrievalStrategy.DOCUMENTATION

        # Domain-specific search
        if len(metadata.domains) > 0 and metadata.domain_confidence > 0.5:
            return RetrievalStrategy.DOMAIN_SPECIFIC

        # Code search (has language hint)
        if len(metadata.languages) > 0:
            return RetrievalStrategy.CODE_SEARCH

        # General search
        return RetrievalStrategy.GENERAL


class MetadataFilter:
    """
    Build vector store filters based on query metadata

    Converts QueryMetadata into LanceDB filter expressions
    """

    def __init__(self):
        self.logger = logging.getLogger("MetadataFilter")

    def build_filter(self, query_metadata: QueryMetadata,
                    strict: bool = False) -> Optional[Dict]:
        """
        Build LanceDB filter from query metadata

        Args:
            query_metadata: Extracted query metadata
            strict: If True, require ALL filters to match (AND)
                   If False, allow partial matches (more lenient)

        Returns:
            Filter dict, or None if no filters
        """
        filters = []

        # Language filter (high priority)
        if query_metadata.languages and query_metadata.language_confidence > 0.6:
            lang = query_metadata.languages[0]  # Use top language
            filters.append({'language': lang})
            self.logger.debug(f"Adding language filter: {lang}")

        # Domain filter (medium priority)
        if query_metadata.domains and query_metadata.domain_confidence > 0.5:
            # For domains, use $in operator to match any domain
            domain_filter = {'domains': {'$in': query_metadata.domains}}
            filters.append(domain_filter)
            self.logger.debug(f"Adding domain filter: {query_metadata.domains}")

        # Topic filter (high priority - very specific)
        if query_metadata.topics and query_metadata.topic_confidence > 0.7:
            topic_filter = {'topics': {'$in': query_metadata.topics}}
            filters.append(topic_filter)
            self.logger.debug(f"Adding topic filter: {query_metadata.topics}")

        # Quality filters
        if query_metadata.prefer_complete:
            filters.append({'is_complete': True})
            self.logger.debug("Adding filter: is_complete=True")

        if query_metadata.prefer_examples:
            filters.append({'has_examples': True})
            self.logger.debug("Adding filter: has_examples=True")

        if query_metadata.prefer_documented:
            filters.append({'has_documentation': True})
            self.logger.debug("Adding filter: has_documentation=True")

        if query_metadata.prefer_simple:
            filters.append({'complexity': 'simple'})
            self.logger.debug("Adding filter: complexity=simple")

        if query_metadata.prefer_complex:
            filters.append({'complexity': 'complex'})
            self.logger.debug("Adding filter: complexity=complex")

        # Code pattern filters
        if query_metadata.code_patterns:
            pattern_filter = {'code_patterns': {'$in': query_metadata.code_patterns}}
            filters.append(pattern_filter)
            self.logger.debug(f"Adding code pattern filter: {query_metadata.code_patterns}")

        # Build final filter
        if not filters:
            return None

        if len(filters) == 1:
            return filters[0]

        # Combine filters with AND or OR
        if strict:
            return {'$and': filters}
        else:
            # For lenient mode, use OR for domain/topic filters, AND for others
            # This is more complex, for now just use AND
            return {'$and': filters}


class ScoreBooster:
    """
    Boost chunk scores based on metadata matches

    Increases relevance scores when chunk metadata matches query hints
    """

    # Boost multipliers
    BOOST_LANGUAGE_MATCH = 1.20  # 20% boost for language match
    BOOST_DOMAIN_MATCH = 1.30  # 30% boost for domain match
    BOOST_TOPIC_MATCH = 1.50  # 50% boost for topic match
    BOOST_COMPLETE_FUNCTION = 1.15  # 15% boost for complete functions
    BOOST_HAS_EXAMPLES = 1.25  # 25% boost if has examples
    BOOST_HAS_DOCUMENTATION = 1.10  # 10% boost if documented
    BOOST_CODE_PATTERN_MATCH = 1.35  # 35% boost for code pattern match
    BOOST_COMPLEXITY_MATCH = 1.05  # 5% boost for complexity match

    def __init__(self):
        self.logger = logging.getLogger("ScoreBooster")

    def boost_scores(self, chunks: List, query_metadata: QueryMetadata) -> List:
        """
        Boost chunk scores based on metadata matches

        Args:
            chunks: List of RetrievedChunk objects
            query_metadata: Query metadata with hints

        Returns:
            Chunks with boosted final_score
        """
        for chunk in chunks:
            boost = 1.0
            reasons = []

            # Language match
            if query_metadata.languages and hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_lang = chunk.metadata.get('language')
                if chunk_lang in query_metadata.languages:
                    boost *= self.BOOST_LANGUAGE_MATCH
                    reasons.append(f"language:{chunk_lang}")

            # Domain match
            if query_metadata.domains and hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_domains = chunk.metadata.get('domains', [])
                if isinstance(chunk_domains, list):
                    matching_domains = set(chunk_domains) & set(query_metadata.domains)
                    if matching_domains:
                        boost *= self.BOOST_DOMAIN_MATCH
                        reasons.append(f"domains:{','.join(matching_domains)}")

            # Topic match (highest boost)
            if query_metadata.topics and hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_topics = chunk.metadata.get('topics', [])
                if isinstance(chunk_topics, list):
                    matching_topics = set(chunk_topics) & set(query_metadata.topics)
                    if matching_topics:
                        boost *= self.BOOST_TOPIC_MATCH
                        reasons.append(f"topics:{','.join(matching_topics)}")

            # Complete function preference
            if query_metadata.prefer_complete and hasattr(chunk, 'metadata') and chunk.metadata:
                if chunk.metadata.get('is_complete'):
                    boost *= self.BOOST_COMPLETE_FUNCTION
                    reasons.append("complete")

            # Examples preference
            if query_metadata.prefer_examples and hasattr(chunk, 'metadata') and chunk.metadata:
                if chunk.metadata.get('has_examples'):
                    boost *= self.BOOST_HAS_EXAMPLES
                    reasons.append("examples")

            # Documentation preference
            if query_metadata.prefer_documented and hasattr(chunk, 'metadata') and chunk.metadata:
                if chunk.metadata.get('has_documentation'):
                    boost *= self.BOOST_HAS_DOCUMENTATION
                    reasons.append("documented")

            # Code pattern match
            if query_metadata.code_patterns and hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_patterns = chunk.metadata.get('code_patterns', [])
                if isinstance(chunk_patterns, list):
                    matching_patterns = set(chunk_patterns) & set(query_metadata.code_patterns)
                    if matching_patterns:
                        boost *= self.BOOST_CODE_PATTERN_MATCH
                        reasons.append(f"patterns:{','.join(matching_patterns)}")

            # Complexity match
            if hasattr(chunk, 'metadata') and chunk.metadata:
                chunk_complexity = chunk.metadata.get('complexity')
                if query_metadata.prefer_simple and chunk_complexity == 'simple':
                    boost *= self.BOOST_COMPLEXITY_MATCH
                    reasons.append("simple")
                elif query_metadata.prefer_complex and chunk_complexity == 'complex':
                    boost *= self.BOOST_COMPLEXITY_MATCH
                    reasons.append("complex")

            # Apply boost
            if boost > 1.0:
                original_score = chunk.final_score
                chunk.final_score *= boost
                self.logger.debug(
                    f"Boosted chunk {chunk.chunk_id[:20]}... "
                    f"from {original_score:.3f} to {chunk.final_score:.3f} "
                    f"({boost:.2f}x) - {', '.join(reasons)}"
                )

        # Re-sort by boosted scores
        chunks.sort(key=lambda c: c.final_score, reverse=True)

        return chunks
