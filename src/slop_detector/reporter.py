"""
Report generation for slop detection results.

Generates human-readable and machine-readable reports.
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich import box

from .models import (
    CategorySummary,
    Confidence,
    ContentType,
    FileAnalysis,
    RepositoryAnalysis,
)


class Reporter:
    """Generate reports from analysis results."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def print_summary(self, analysis: RepositoryAnalysis) -> None:
        """Print a summary report to the console."""
        # Header
        self.console.print()
        title = f"[bold]Slop Detector Report[/bold] - {analysis.repo_name}"
        self.console.print(Panel(title, style="blue"))

        # Overall score with visual indicator
        score_color = self._get_score_color(analysis.overall_ai_probability)
        score_bar = self._create_score_bar(analysis.overall_ai_probability)

        overall_panel = Panel(
            f"[bold]Overall AI Probability:[/bold] [{score_color}]{analysis.overall_ai_probability:.1%}[/{score_color}]\n"
            f"{score_bar}\n\n"
            f"[bold]Confidence:[/bold] {analysis.overall_confidence.value.replace('_', ' ').title()}\n"
            f"[bold]Files Analyzed:[/bold] {analysis.total_files}\n"
            f"[bold]Lines Analyzed:[/bold] {analysis.total_lines:,}",
            title="Summary",
            border_style=score_color
        )
        self.console.print(overall_panel)

        # Definitive signals (smoking gun evidence) - FIRST AND PROMINENT
        self._print_definitive_signals(analysis)

        # Commit analysis
        self._print_commit_analysis(analysis)

        # Category breakdown
        self._print_category_breakdown(analysis)

        # Top signals
        self._print_signals(analysis)

        # File details (top flagged files)
        self._print_top_files(analysis)

        # Footer
        self.console.print()
        self.console.print(
            f"[dim]Analysis completed in {analysis.analysis_duration_seconds:.1f}s[/dim]"
        )

    def _get_score_color(self, probability: float) -> str:
        """Get color based on AI probability."""
        if probability < 0.3:
            return "green"
        elif probability < 0.5:
            return "yellow"
        elif probability < 0.7:
            return "orange1"
        else:
            return "red"

    def _create_score_bar(self, probability: float, width: int = 40) -> str:
        """Create a visual probability bar."""
        filled = int(probability * width)
        empty = width - filled
        color = self._get_score_color(probability)

        bar = f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"
        return f"Human {bar} AI"

    def _print_definitive_signals(self, analysis: RepositoryAnalysis) -> None:
        """Print definitive (smoking gun) signals prominently."""
        if not analysis.definitive_signals:
            return

        # Create a prominent warning panel for definitive signals
        signal_lines = []
        for signal in analysis.definitive_signals[:10]:
            score_pct = f"{signal.score:.0%}"
            signal_lines.append(f"[bold]🚨 {signal.name}[/bold] ({score_pct})")
            if signal.description:
                signal_lines.append(f"   [dim]{signal.description}[/dim]")
            if signal.evidence:
                for ev in signal.evidence[:2]:
                    signal_lines.append(f"   [cyan]→ {ev[:80]}[/cyan]")

        if signal_lines:
            panel = Panel(
                "\n".join(signal_lines),
                title="[bold red]⚠️  DEFINITIVE AI MARKERS DETECTED[/bold red]",
                border_style="red",
                subtitle="[dim]High-confidence evidence of AI tool usage[/dim]"
            )
            self.console.print(panel)

    def _print_commit_analysis(self, analysis: RepositoryAnalysis) -> None:
        """Print commit history analysis."""
        if not analysis.commit_signals:
            return

        # Filter for commit-related signals
        commit_related = [
            s for s in analysis.commit_signals
            if any(keyword in s.name.lower() for keyword in
                   ["commit", "author", "co-author", "attributed", "marker file"])
        ]

        if not commit_related:
            return

        signal_lines = []
        for signal in commit_related[:8]:
            score_pct = f"{signal.score:.0%}"
            icon = "🤖" if signal.score > 0.8 else "📝"
            signal_lines.append(f"{icon} [bold]{signal.name}[/bold] ({score_pct})")
            if signal.description:
                signal_lines.append(f"   [dim]{signal.description}[/dim]")
            if signal.evidence:
                for ev in signal.evidence[:2]:
                    # Truncate long evidence
                    ev_display = ev[:70] + "..." if len(ev) > 70 else ev
                    signal_lines.append(f"   [yellow]→ {ev_display}[/yellow]")

        if signal_lines:
            title = "[bold magenta]📊 Commit History Analysis[/bold magenta]"
            if analysis.ai_attributed_commits > 0:
                title += f" [dim]({analysis.ai_attributed_commits} AI-attributed commits)[/dim]"

            panel = Panel(
                "\n".join(signal_lines),
                title=title,
                border_style="magenta"
            )
            self.console.print(panel)

    def _print_category_breakdown(self, analysis: RepositoryAnalysis) -> None:
        """Print breakdown by content type."""
        table = Table(title="Breakdown by Content Type", box=box.ROUNDED)
        table.add_column("Category", style="bold")
        table.add_column("Files", justify="right")
        table.add_column("Lines", justify="right")
        table.add_column("AI Probability", justify="center")
        table.add_column("Confidence", justify="center")

        categories = [
            ("Code", analysis.code_analysis),
            ("Documentation", analysis.docs_analysis),
            ("Configuration", analysis.config_analysis),
        ]

        for name, cat in categories:
            if cat:
                color = self._get_score_color(cat.ai_probability)
                prob_str = f"[{color}]{cat.ai_probability:.1%}[/{color}]"
                table.add_row(
                    name,
                    str(cat.total_files),
                    f"{cat.total_lines:,}",
                    prob_str,
                    cat.confidence.value.replace("_", " ").title()
                )
            else:
                table.add_row(name, "-", "-", "-", "-")

        self.console.print(table)

    def _print_signals(self, analysis: RepositoryAnalysis) -> None:
        """Print top detection signals."""
        if analysis.top_ai_signals:
            ai_panel = Panel(
                "\n".join(f"• {signal}" for signal in analysis.top_ai_signals[:7]),
                title="[red]Top AI Indicators[/red]",
                border_style="red"
            )
            self.console.print(ai_panel)

        if analysis.top_human_signals:
            human_panel = Panel(
                "\n".join(f"• {signal}" for signal in analysis.top_human_signals[:5]),
                title="[green]Human Writing Indicators[/green]",
                border_style="green"
            )
            self.console.print(human_panel)

    def _print_top_files(self, analysis: RepositoryAnalysis, limit: int = 10) -> None:
        """Print files with highest AI probability."""
        if not analysis.file_analyses:
            return

        # Sort by AI probability
        sorted_files = sorted(
            analysis.file_analyses,
            key=lambda f: f.ai_probability,
            reverse=True
        )

        # Only show files with significant AI probability
        flagged_files = [f for f in sorted_files if f.ai_probability > 0.4][:limit]

        if not flagged_files:
            self.console.print(
                Panel(
                    "[green]No files flagged with significant AI content.[/green]",
                    title="Flagged Files"
                )
            )
            return

        table = Table(title="Files with Highest AI Probability", box=box.ROUNDED)
        table.add_column("File", style="cyan", max_width=50)
        table.add_column("Type", justify="center")
        table.add_column("Lines", justify="right")
        table.add_column("AI Prob", justify="center")
        table.add_column("Top Signal", max_width=30)

        for f in flagged_files:
            color = self._get_score_color(f.ai_probability)
            prob_str = f"[{color}]{f.ai_probability:.1%}[/{color}]"
            top_signal = f.signals[0].name if f.signals else "-"

            table.add_row(
                str(f.path),
                f.content_type.value[:4].title(),
                str(f.line_count),
                prob_str,
                top_signal
            )

        self.console.print(table)

    def print_detailed(self, analysis: RepositoryAnalysis) -> None:
        """Print a detailed report with all file analyses."""
        self.print_summary(analysis)

        self.console.print()
        self.console.print("[bold]Detailed File Analysis[/bold]")
        self.console.print("=" * 60)

        for file_analysis in sorted(
            analysis.file_analyses,
            key=lambda f: f.ai_probability,
            reverse=True
        ):
            self._print_file_detail(file_analysis)

    def _print_file_detail(self, file_analysis: FileAnalysis) -> None:
        """Print detailed analysis for a single file."""
        color = self._get_score_color(file_analysis.ai_probability)

        self.console.print(f"\n[bold cyan]{file_analysis.path}[/bold cyan]")
        self.console.print(f"  Type: {file_analysis.content_type.value}")
        self.console.print(
            f"  AI Probability: [{color}]{file_analysis.ai_probability:.1%}[/{color}]"
        )
        self.console.print(f"  Confidence: {file_analysis.confidence.value}")

        if file_analysis.signals:
            self.console.print("  Signals:")
            for signal in file_analysis.signals[:5]:
                self.console.print(
                    f"    • {signal.name}: {signal.score:.0%} (weight: {signal.weight:.1f})"
                )
                if signal.evidence:
                    for ev in signal.evidence[:2]:
                        self.console.print(f"      [dim]{ev[:60]}...[/dim]")

    def to_json(self, analysis: RepositoryAnalysis) -> str:
        """Convert analysis to JSON format."""
        def serialize(obj):
            if hasattr(obj, 'value'):  # Enum
                return obj.value
            if hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            if hasattr(obj, 'as_posix'):  # Path
                return str(obj)
            return obj

        data = {
            "repository": analysis.repo_name,
            "url": analysis.repo_url,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "overall_ai_probability": analysis.overall_ai_probability,
                "confidence": analysis.overall_confidence.value,
                "total_files": analysis.total_files,
                "total_lines": analysis.total_lines,
                "analysis_duration_seconds": analysis.analysis_duration_seconds,
            },
            "breakdown": {
                "code": self._category_to_dict(analysis.code_analysis),
                "documentation": self._category_to_dict(analysis.docs_analysis),
                "configuration": self._category_to_dict(analysis.config_analysis),
            },
            "top_ai_signals": analysis.top_ai_signals,
            "top_human_signals": analysis.top_human_signals,
            "files": [
                {
                    "path": str(f.path),
                    "type": f.content_type.value,
                    "ai_probability": f.ai_probability,
                    "confidence": f.confidence.value,
                    "line_count": f.line_count,
                    "signals": [
                        {
                            "name": s.name,
                            "score": s.score,
                            "weight": s.weight,
                            "description": s.description,
                        }
                        for s in f.signals
                    ]
                }
                for f in analysis.file_analyses
            ]
        }

        return json.dumps(data, indent=2)

    def _category_to_dict(self, category: Optional[CategorySummary]) -> Optional[dict]:
        """Convert category summary to dictionary."""
        if not category:
            return None

        return {
            "total_files": category.total_files,
            "total_lines": category.total_lines,
            "ai_probability": category.ai_probability,
            "ai_line_estimate": category.ai_line_estimate,
            "confidence": category.confidence.value,
            "top_signals": category.top_signals,
        }

    def to_markdown(self, analysis: RepositoryAnalysis) -> str:
        """Convert analysis to Markdown format."""
        lines = [
            f"# Slop Detector Report: {analysis.repo_name}",
            "",
            f"**Repository:** {analysis.repo_url}",
            f"**Analyzed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall AI Probability | {analysis.overall_ai_probability:.1%} |",
            f"| Confidence | {analysis.overall_confidence.value.replace('_', ' ').title()} |",
            f"| Files Analyzed | {analysis.total_files} |",
            f"| Lines Analyzed | {analysis.total_lines:,} |",
            f"| Analysis Time | {analysis.analysis_duration_seconds:.1f}s |",
            "",
            "## Breakdown by Content Type",
            "",
            "| Category | Files | Lines | AI Probability | Confidence |",
            "|----------|-------|-------|----------------|------------|",
        ]

        categories = [
            ("Code", analysis.code_analysis),
            ("Documentation", analysis.docs_analysis),
            ("Configuration", analysis.config_analysis),
        ]

        for name, cat in categories:
            if cat:
                lines.append(
                    f"| {name} | {cat.total_files} | {cat.total_lines:,} | "
                    f"{cat.ai_probability:.1%} | {cat.confidence.value} |"
                )
            else:
                lines.append(f"| {name} | - | - | - | - |")

        lines.extend([
            "",
            "## Top AI Indicators",
            "",
        ])

        if analysis.top_ai_signals:
            for signal in analysis.top_ai_signals[:10]:
                lines.append(f"- {signal}")
        else:
            lines.append("- No significant AI indicators found")

        lines.extend([
            "",
            "## Files with Highest AI Probability",
            "",
            "| File | Type | Lines | AI Probability |",
            "|------|------|-------|----------------|",
        ])

        sorted_files = sorted(
            analysis.file_analyses,
            key=lambda f: f.ai_probability,
            reverse=True
        )[:15]

        for f in sorted_files:
            if f.ai_probability > 0.3:
                lines.append(
                    f"| {f.path} | {f.content_type.value} | "
                    f"{f.line_count} | {f.ai_probability:.1%} |"
                )

        return "\n".join(lines)
