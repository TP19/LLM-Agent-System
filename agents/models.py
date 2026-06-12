#!/usr/bin/env python3
"""
Data models for Intelligent Learning Agents
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorType(Enum):
    """Common error types"""
    SYNTAX_ERROR = "SyntaxError"
    RUNTIME_ERROR = "RuntimeError"
    ASSERTION_ERROR = "AssertionError"
    IMPORT_ERROR = "ImportError"
    TYPE_ERROR = "TypeError"
    VALUE_ERROR = "ValueError"
    ATTRIBUTE_ERROR = "AttributeError"
    NAME_ERROR = "NameError"
    TEST_FAILURE = "TestFailure"
    OTHER = "Other"


class LogStatus(Enum):
    """Overall log analysis status"""
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class Error:
    """Individual error extracted from logs"""
    type: str  # "SyntaxError", "AssertionError", etc.
    file: Optional[str] = None
    line: Optional[int] = None
    message: str = ""
    severity: str = "medium"
    stack_trace: Optional[str] = None
    project: str = ""

    @property
    def signature(self) -> str:
        """Generate unique signature for this error"""
        return f"{self.type}:{self.file}:{self.line}:{self.message[:50]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "type": self.type,
            "file": self.file,
            "line": self.line,
            "message": self.message,
            "severity": self.severity,
            "stack_trace": self.stack_trace,
            "project": self.project
        }


@dataclass
class Warning:
    """Warning extracted from logs"""
    category: str
    message: str
    file: Optional[str] = None
    line: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "category": self.category,
            "message": self.message,
            "file": self.file,
            "line": self.line
        }


@dataclass
class LogAnalysis:
    """Complete log analysis result"""
    status: str  # "success" | "failed" | "degraded"
    errors: List[Error] = field(default_factory=list)
    warnings: List[Warning] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_output: str = ""
    project: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "metrics": self.metrics,
            "project": self.project,
            "timestamp": self.timestamp.isoformat(),
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "critical_errors": len([e for e in self.errors if e.severity == "critical"]),
                "has_metrics": bool(self.metrics)
            }
        }


@dataclass
class ErrorCase:
    """Error case stored in knowledge base"""
    error_signature: str
    error_type: str
    file: str
    line: int
    project: str
    compressed_output: str
    original_tokens: int
    compressed_tokens: int
    fix: str
    outcome: str  # "success" | "failed" | "pending"
    timestamp: datetime
    cycle_number: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_signature": self.error_signature,
            "error_type": self.error_type,
            "file": self.file,
            "line": self.line,
            "project": self.project,
            "compressed_output": self.compressed_output,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": self.original_tokens / max(1, self.compressed_tokens),
            "fix": self.fix,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "cycle_number": self.cycle_number
        }


@dataclass
class CriteriaCheck:
    """Individual success criteria check"""
    criterion: str
    actual: str
    status: str  # "met" | "not_met" | "unknown"
    explanation: str
    gap: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "criterion": self.criterion,
            "actual": self.actual,
            "status": self.status,
            "explanation": self.explanation,
            "gap": self.gap
        }


@dataclass
class CriteriaResult:
    """Complete success criteria evaluation"""
    criteria_met: bool
    details: List[CriteriaCheck] = field(default_factory=list)
    overall_progress: float = 0.0  # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "criteria_met": self.criteria_met,
            "details": [d.to_dict() for d in self.details],
            "overall_progress": self.overall_progress,
            "summary": {
                "total_criteria": len(self.details),
                "met": len([d for d in self.details if d.status == "met"]),
                "not_met": len([d for d in self.details if d.status == "not_met"])
            }
        }
