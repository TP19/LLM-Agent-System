#!/usr/bin/env python3
"""
Query Parser - Natural Language Query Understanding

Parses natural language queries into structured intents for intelligent retrieval.

Example:
    Input: "What did Feynman say about quantum physics in chapter 3?"
    Output: QueryIntent(
        author='Feynman',
        topic='quantum physics',
        chapter=3,
        intent_type='content_search'
    )
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class IntentType(Enum):
    """Types of query intents"""
    CONTENT_SEARCH = "content_search"  # "What does X say about Y?"
    NAVIGATION = "navigation"           # "Find section Z"
    SUMMARY = "summary"                 # "Summarize chapter X"
    COMPARISON = "comparison"           # "Compare X and Y"
    CODE_SEARCH = "code_search"         # "Find implementation of X"
    LOG_SEARCH = "log_search"           # "Show logs from yesterday"
    LISTING = "listing"                 # "List all documents about X"


@dataclass
class QueryIntent:
    """Structured representation of query intent"""
    
    # Original query
    raw_query: str
    
    # Extracted entities
    author: Optional[str] = None
    title: Optional[str] = None
    chapter: Optional[int] = None
    section: Optional[int] = None
    topic: Optional[str] = None
    
    # Temporal filters
    time_range: Optional[str] = None  # "yesterday", "last week", etc.
    
    # Document type filters
    doc_type: Optional[str] = None  # "book", "code", "log", etc.
    language: Optional[str] = None  # "python", "javascript", etc.
    
    # Intent classification
    intent_type: IntentType = IntentType.CONTENT_SEARCH
    
    # Metadata
    confidence: float = 0.0
    entities_found: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


class QueryParser:
    """
    Parse natural language queries into structured intents
    
    Features:
    - Entity extraction (author, title, chapter, section)
    - Topic identification
    - Intent classification
    - Temporal parsing
    - Type detection (code, book, log)
    """
    
    def __init__(self):
        self.logger = logging.getLogger("QueryParser")
        
        # Common author patterns
        self.author_patterns = [
            r"(?:by|from|according to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:book|paper|article|lecture)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|wrote|explains|discusses)",
        ]
        
        # Chapter/section patterns
        self.chapter_patterns = [
            r"chapter\s+(\d+)",
            r"ch(?:ap)?\.?\s+(\d+)",
            r"(\d+)(?:st|nd|rd|th)\s+chapter",
        ]
        
        self.section_patterns = [
            r"section\s+(\d+)",
            r"sec\.?\s+(\d+)",
            r"(\d+)\.(\d+)",  # Like "3.2"
        ]
        
        # Title patterns
        self.title_patterns = [
            r'"([^"]+)"',  # Quoted title
            r"'([^']+)'",  # Single quoted
            r"book\s+(?:titled|called|named)\s+([A-Z][^\s,\.]+(?:\s+[A-Z][^\s,\.]+)*)",
            r"(?:in|from)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",  # Like "Feynman Lectures"
        ]
        
        # Temporal patterns
        self.temporal_patterns = {
            'today': r'\btoday\b',
            'yesterday': r'\byesterday\b',
            'last_week': r'\blast\s+week\b',
            'last_month': r'\blast\s+month\b',
            'this_week': r'\bthis\s+week\b',
        }
        
        # Code language patterns
        self.language_patterns = {
            'python': r'\b(?:python|\.py)\b',
            'javascript': r'\b(?:javascript|js|\.js)\b',
            'java': r'\b(?:java|\.java)\b',
            'cpp': r'\b(?:c\+\+|cpp|\.cpp)\b',
            'go': r'\b(?:golang|go|\.go)\b',
        }
        
        # Intent keywords
        self.intent_keywords = {
            IntentType.CONTENT_SEARCH: [
                'what', 'explain', 'tell me', 'describe', 'how does', 'why does',
                'information about', 'details about'
            ],
            IntentType.NAVIGATION: [
                'find', 'show', 'locate', 'where is', 'get', 'retrieve'
            ],
            IntentType.SUMMARY: [
                'summarize', 'summary', 'overview', 'brief', 'outline'
            ],
            IntentType.COMPARISON: [
                'compare', 'difference', 'versus', 'vs', 'contrast', 'similarity'
            ],
            IntentType.CODE_SEARCH: [
                'implementation', 'function', 'class', 'method', 'code for',
                'how to implement'
            ],
            IntentType.LOG_SEARCH: [
                'logs', 'log file', 'error log', 'debugging', 'trace'
            ],
            IntentType.LISTING: [
                'list', 'show all', 'all documents', 'everything about'
            ],
        }
    
    def parse(self, query: str) -> QueryIntent:
        """
        Parse a natural language query into structured intent
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryIntent with extracted entities and classification
        """
        
        self.logger.debug(f"Parsing query: {query}")
        
        # Initialize intent
        intent = QueryIntent(raw_query=query)
        
        # Extract entities
        intent.author = self._extract_author(query)
        intent.title = self._extract_title(query)
        intent.chapter = self._extract_chapter(query)
        intent.section = self._extract_section(query)
        intent.topic = self._extract_topic(query)
        
        # Extract temporal filters
        intent.time_range = self._extract_temporal(query)
        
        # Extract type filters
        intent.doc_type = self._detect_doc_type(query)
        intent.language = self._detect_language(query)
        
        # Classify intent
        intent.intent_type = self._classify_intent(query)
        
        # Extract keywords
        intent.keywords = self._extract_keywords(query)
        
        # Track what we found
        if intent.author:
            intent.entities_found.append('author')
        if intent.title:
            intent.entities_found.append('title')
        if intent.chapter:
            intent.entities_found.append('chapter')
        if intent.section:
            intent.entities_found.append('section')
        if intent.topic:
            intent.entities_found.append('topic')
        
        # Calculate confidence
        intent.confidence = self._calculate_confidence(intent)
        
        self.logger.info(
            f"Parsed intent: {intent.intent_type.value}, "
            f"entities: {intent.entities_found}, "
            f"confidence: {intent.confidence:.2f}"
        )
        
        return intent
    
    def _extract_author(self, query: str) -> Optional[str]:
        """Extract author name from query"""
        
        for pattern in self.author_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                self.logger.debug(f"Found author: {author}")
                return author
        
        return None
    
    def _extract_title(self, query: str) -> Optional[str]:
        """Extract document title from query"""
        
        for pattern in self.title_patterns:
            match = re.search(pattern, query)
            if match:
                title = match.group(1).strip()
                self.logger.debug(f"Found title: {title}")
                return title
        
        return None
    
    def _extract_chapter(self, query: str) -> Optional[int]:
        """Extract chapter number from query"""
        
        for pattern in self.chapter_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    chapter = int(match.group(1))
                    self.logger.debug(f"Found chapter: {chapter}")
                    return chapter
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_section(self, query: str) -> Optional[int]:
        """Extract section number from query"""
        
        for pattern in self.section_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    # Handle both "section 2" and "3.2" formats
                    if len(match.groups()) == 2:
                        section = int(match.group(2))
                    else:
                        section = int(match.group(1))
                    self.logger.debug(f"Found section: {section}")
                    return section
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract main topic from query"""
        
        # Remove stop words and common query words
        stop_words = {
            'what', 'does', 'say', 'about', 'the', 'in', 'chapter', 'section',
            'find', 'show', 'me', 'tell', 'explain', 'how', 'why', 'when',
            'where', 'which', 'who', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'can', 'could', 'should', 'would',
            'my', 'documents', 'document', 'book', 'article', 'paper'
        }
        
        # Extract meaningful words
        words = re.findall(r'\b[a-z]{3,}\b', query.lower())
        
        # Filter out stop words
        meaningful_words = [w for w in words if w not in stop_words]
        
        if not meaningful_words:
            return None
        
        # Look for quoted phrases first (highest priority)
        quoted = re.findall(r'"([^"]+)"', query)
        if quoted:
            topic = quoted[0].strip()
            self.logger.debug(f"Found topic (quoted): {topic}")
            return topic
        
        # Look for "about X" pattern
        about_match = re.search(r'\babout\s+([a-z\s]+?)(?:\s+in|\s+from|$)', query.lower())
        if about_match:
            topic = about_match.group(1).strip()
            self.logger.debug(f"Found topic (about): {topic}")
            return topic
        
        # Look for multi-word technical terms
        bigrams = [f"{meaningful_words[i]} {meaningful_words[i+1]}" 
                   for i in range(len(meaningful_words)-1)]
        
        # Check for known technical terms
        technical_terms = ['quantum', 'machine learning', 'neural network', 
                          'data structure', 'algorithm', 'authentication']
        for term in technical_terms:
            if term in query.lower():
                self.logger.debug(f"Found topic (technical): {term}")
                return term
        
        # Use first meaningful bigram or word
        if bigrams:
            topic = bigrams[0]
        elif meaningful_words:
            topic = meaningful_words[0]
        else:
            return None
        
        self.logger.debug(f"Found topic (extracted): {topic}")
        return topic
    
    def _extract_temporal(self, query: str) -> Optional[str]:
        """Extract temporal reference from query"""
        
        query_lower = query.lower()
        
        for time_key, pattern in self.temporal_patterns.items():
            if re.search(pattern, query_lower):
                self.logger.debug(f"Found temporal: {time_key}")
                return time_key
        
        return None
    
    def _detect_doc_type(self, query: str) -> Optional[str]:
        """Detect document type from query"""
        
        query_lower = query.lower()
        
        # Explicit type mentions
        if 'book' in query_lower or 'chapter' in query_lower:
            return 'book'
        elif 'code' in query_lower or 'implementation' in query_lower:
            return 'code'
        elif 'log' in query_lower or 'error' in query_lower:
            return 'log'
        elif 'article' in query_lower or 'paper' in query_lower:
            return 'article'
        
        return None
    
    def _detect_language(self, query: str) -> Optional[str]:
        """Detect programming language from query"""
        
        query_lower = query.lower()
        
        for lang, pattern in self.language_patterns.items():
            if re.search(pattern, query_lower):
                self.logger.debug(f"Found language: {lang}")
                return lang
        
        return None
    
    def _classify_intent(self, query: str) -> IntentType:
        """Classify the intent type of the query"""
        
        query_lower = query.lower()
        
        # Score each intent type
        scores = {}
        for intent_type, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[intent_type] = score
        
        # Get highest scoring intent
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] > 0:
                self.logger.debug(f"Classified as: {best_intent.value}")
                return best_intent
        
        # Default to content search
        return IntentType.CONTENT_SEARCH
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        
        # Common stop words
        stop_words = {
            'what', 'does', 'say', 'about', 'the', 'and', 'for', 'are',
            'with', 'from', 'this', 'that', 'have', 'find', 'show'
        }
        
        # Filter keywords
        keywords = [w for w in words if w not in stop_words]
        
        return keywords[:10]  # Limit to top 10
    
    def _calculate_confidence(self, intent: QueryIntent) -> float:
        """Calculate confidence score for parsed intent"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for each entity found
        if intent.author:
            confidence += 0.1
        if intent.title:
            confidence += 0.1
        if intent.chapter is not None:
            confidence += 0.1
        if intent.topic:
            confidence += 0.1
        if intent.doc_type:
            confidence += 0.05
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def format_intent(self, intent: QueryIntent) -> str:
        """Format intent for display"""
        
        lines = []
        lines.append(f"Query: {intent.raw_query}")
        lines.append(f"Intent: {intent.intent_type.value}")
        
        if intent.author:
            lines.append(f"  Author: {intent.author}")
        if intent.title:
            lines.append(f"  Title: {intent.title}")
        if intent.chapter:
            lines.append(f"  Chapter: {intent.chapter}")
        if intent.section:
            lines.append(f"  Section: {intent.section}")
        if intent.topic:
            lines.append(f"  Topic: {intent.topic}")
        if intent.doc_type:
            lines.append(f"  Type: {intent.doc_type}")
        if intent.time_range:
            lines.append(f"  Time: {intent.time_range}")
        
        lines.append(f"Confidence: {intent.confidence:.2f}")
        
        return "\n".join(lines)