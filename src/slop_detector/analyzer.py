"""
Main analyzer orchestrating all detection modules.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .aggregator import SignalAggregator, AggregatorConfig
from .detectors.heuristic_detector import HeuristicDetector, HeuristicConfig
from .detectors.statistical_detector import StatisticalDetector, StatisticalConfig
from .detectors.ml_detector import MLDetector, MLConfig
from .detectors.commit_analyzer import CommitAnalyzer, CommitConfig
from .detectors.definitive_detector import DefinitiveDetector
from .file_classifier import FileClassifier
from .models import (
    ContentType,
    DetectionSignal,
    FileAnalysis,
    RepositoryAnalysis,
)
from .repo_analyzer import RepoAnalyzer


@dataclass
class AnalyzerConfig:
    """Configuration for the main analyzer."""
    # Clone settings
    clone_depth: Optional[int] = 100  # Shallow clone for speed

    # Analysis toggles
    enable_heuristic: bool = True
    enable_statistical: bool = True
    enable_ml: bool = False  # Disabled by default (requires model download)
    enable_commit_analysis: bool = True

    # Performance
    max_files: int = 500
    max_file_size: int = 1024 * 1024  # 1MB

    # Sub-configs
    heuristic_config: Optional[HeuristicConfig] = None
    statistical_config: Optional[StatisticalConfig] = None
    ml_config: Optional[MLConfig] = None
    commit_config: Optional[CommitConfig] = None
    aggregator_config: Optional[AggregatorConfig] = None


class SlopDetector:
    """Main slop detection orchestrator."""

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        console: Optional[Console] = None
    ):
        self.config = config or AnalyzerConfig()
        self.console = console or Console()

        # Initialize components
        self.file_classifier = FileClassifier()

        self.heuristic_detector = (
            HeuristicDetector(self.config.heuristic_config)
            if self.config.enable_heuristic else None
        )

        self.statistical_detector = (
            StatisticalDetector(self.config.statistical_config)
            if self.config.enable_statistical else None
        )

        self.ml_detector = (
            MLDetector(self.config.ml_config)
            if self.config.enable_ml else None
        )

        self.commit_analyzer = (
            CommitAnalyzer(self.config.commit_config)
            if self.config.enable_commit_analysis else None
        )

        # Definitive detector is ALWAYS enabled - these are smoking gun signals
        self.definitive_detector = DefinitiveDetector()

        self.aggregator = SignalAggregator(self.config.aggregator_config)

    def analyze_repository(
        self,
        repo_url: str,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> RepositoryAnalysis:
        """
        Analyze a GitHub repository for AI-generated content.

        Args:
            repo_url: GitHub repository URL or local path
            progress_callback: Optional callback for progress updates

        Returns:
            RepositoryAnalysis with detection results
        """
        start_time = time.time()

        with RepoAnalyzer(max_file_size=self.config.max_file_size) as repo:
            # Clone or open repository
            is_local = not repo_url.startswith(("http://", "https://", "git@"))

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:

                if is_local:
                    task = progress.add_task("Opening local repository...", total=1)
                    repo.open_local_repo(repo_url)
                    repo_name = Path(repo_url).name
                else:
                    task = progress.add_task("Cloning repository...", total=1)
                    repo.clone_repo(repo_url, depth=self.config.clone_depth)
                    repo_name = repo.get_repo_name(repo_url)

                progress.update(task, completed=1)

                # Get files to analyze
                files = list(repo.iter_files())[:self.config.max_files]

                # FIRST: Scan for definitive AI marker files (quick scan)
                task = progress.add_task("Scanning for AI markers...", total=1)
                definitive_file_signals = self.definitive_detector.scan_repository_markers(files)
                progress.update(task, completed=1)

                # Analyze files
                task = progress.add_task(
                    f"Analyzing {len(files)} files...",
                    total=len(files)
                )

                file_analyses = []
                for rel_path in files:
                    content = repo.read_file(rel_path)
                    if content:
                        analysis = self._analyze_file(rel_path, content)
                        if analysis:
                            file_analyses.append(analysis)

                    progress.update(task, advance=1)

                # Analyze commits
                commit_signals = []
                if self.commit_analyzer:
                    task = progress.add_task("Analyzing commit history...", total=1)
                    commits = repo.get_commit_history(max_commits=100)
                    _, commit_signals = self.commit_analyzer.analyze_commits(commits)
                    progress.update(task, completed=1)

                # Combine definitive signals with commit signals
                commit_signals.extend(definitive_file_signals)

        # Aggregate results
        analysis_duration = time.time() - start_time

        result = self.aggregator.aggregate_repository(
            file_analyses=file_analyses,
            commit_signals=commit_signals,
            repo_url=repo_url,
            repo_name=repo_name,
            analysis_duration=analysis_duration
        )

        return result

    def _analyze_file(self, path: Path, content: str) -> Optional[FileAnalysis]:
        """Analyze a single file."""
        # Classify file type
        content_type = self.file_classifier.classify(path)

        if content_type == ContentType.UNKNOWN:
            return None

        # Skip very small files
        line_count = len(content.split("\n"))
        if line_count < 5:
            return None

        # Collect signals from all detectors
        signals: list[DetectionSignal] = []

        # FIRST: Check for definitive AI markers in file content
        # These are smoking gun signals that override heuristics
        definitive_signals = self.definitive_detector.check_content(content, str(path))
        signals.extend(definitive_signals)

        # Also check filename for marker files
        filename_signals = self.definitive_detector.check_filename(path)
        signals.extend(filename_signals)

        if self.heuristic_detector:
            signals.extend(
                self.heuristic_detector.analyze(content, content_type, str(path))
            )

        if self.statistical_detector:
            signals.extend(
                self.statistical_detector.analyze(content, content_type)
            )

        if self.ml_detector:
            signals.extend(
                self.ml_detector.analyze(content, content_type)
            )

        # Aggregate into file analysis
        analysis = self.aggregator.aggregate_file_signals(
            signals, content_type, line_count
        )
        analysis.path = path

        return analysis

    def analyze_text(
        self,
        content: str,
        content_type: ContentType = ContentType.CODE,
        filename: str = ""
    ) -> FileAnalysis:
        """
        Analyze a single piece of text.

        Useful for testing or analyzing snippets.
        """
        signals: list[DetectionSignal] = []

        if self.heuristic_detector:
            signals.extend(
                self.heuristic_detector.analyze(content, content_type, filename)
            )

        if self.statistical_detector:
            signals.extend(
                self.statistical_detector.analyze(content, content_type)
            )

        if self.ml_detector:
            signals.extend(
                self.ml_detector.analyze(content, content_type)
            )

        line_count = len(content.split("\n"))
        analysis = self.aggregator.aggregate_file_signals(
            signals, content_type, line_count
        )
        analysis.path = Path(filename) if filename else Path("snippet")

        return analysis
