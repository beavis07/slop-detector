"""Data models for slop detection results."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ContentType(Enum):
    """Classification of file content types."""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DATA = "data"
    UNKNOWN = "unknown"


class Confidence(Enum):
    """Confidence levels for detection."""
    VERY_LOW = "very_low"      # < 20%
    LOW = "low"                # 20-40%
    MEDIUM = "medium"          # 40-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # > 80%

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        """Convert a 0-1 score to confidence level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MEDIUM
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


@dataclass
class DetectionSignal:
    """A single detection signal from an analyzer."""
    name: str
    score: float  # 0-1 where 1 = definitely AI
    weight: float  # How much to trust this signal
    description: str
    evidence: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class FileAnalysis:
    """Analysis result for a single file."""
    path: Path
    content_type: ContentType
    ai_probability: float  # 0-1
    confidence: Confidence
    signals: list[DetectionSignal] = field(default_factory=list)
    line_count: int = 0
    ai_line_estimate: int = 0

    @property
    def human_probability(self) -> float:
        return 1.0 - self.ai_probability


@dataclass
class CommitAnalysis:
    """Analysis of commit patterns."""
    sha: str
    message: str
    ai_probability: float
    signals: list[DetectionSignal] = field(default_factory=list)
    files_changed: int = 0


@dataclass
class CategorySummary:
    """Summary for a content category."""
    content_type: ContentType
    total_files: int
    total_lines: int
    ai_probability: float
    ai_line_estimate: int
    confidence: Confidence
    top_signals: list[str] = field(default_factory=list)


@dataclass
class RepositoryAnalysis:
    """Complete analysis of a repository."""
    repo_url: str
    repo_name: str

    # Overall metrics
    overall_ai_probability: float
    overall_confidence: Confidence

    # Breakdown by content type
    code_analysis: Optional[CategorySummary] = None
    docs_analysis: Optional[CategorySummary] = None
    config_analysis: Optional[CategorySummary] = None

    # Detailed results
    file_analyses: list[FileAnalysis] = field(default_factory=list)
    commit_analyses: list[CommitAnalysis] = field(default_factory=list)

    # Metadata
    total_files: int = 0
    total_lines: int = 0
    analysis_duration_seconds: float = 0.0

    # Top evidence
    top_ai_signals: list[str] = field(default_factory=list)
    top_human_signals: list[str] = field(default_factory=list)

    # Definitive signals (smoking gun evidence)
    definitive_signals: list[DetectionSignal] = field(default_factory=list)

    # Commit analysis summary
    total_commits_analyzed: int = 0
    ai_attributed_commits: int = 0
    commit_signals: list[DetectionSignal] = field(default_factory=list)

    def get_summary(self) -> dict:
        """Get a summary dictionary of the analysis."""
        return {
            "repository": self.repo_name,
            "overall_ai_probability": f"{self.overall_ai_probability:.1%}",
            "confidence": self.overall_confidence.value,
            "breakdown": {
                "code": f"{self.code_analysis.ai_probability:.1%}" if self.code_analysis else "N/A",
                "documentation": f"{self.docs_analysis.ai_probability:.1%}" if self.docs_analysis else "N/A",
                "configuration": f"{self.config_analysis.ai_probability:.1%}" if self.config_analysis else "N/A",
            },
            "total_files": self.total_files,
            "total_lines": self.total_lines,
        }
