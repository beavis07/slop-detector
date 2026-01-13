"""
Command-line interface for slop-detector.
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from . import __version__
from .analyzer import SlopDetector, AnalyzerConfig
from .models import ContentType
from .reporter import Reporter


console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """
    Slop Detector - Detect AI-generated content in repositories.

    Analyzes GitHub repositories to detect AI-generated code,
    documentation, and configuration files.
    """
    pass


@main.command()
@click.argument("repo", required=True)
@click.option(
    "--output", "-o",
    type=click.Choice(["console", "json", "markdown"]),
    default="console",
    help="Output format"
)
@click.option(
    "--output-file", "-f",
    type=click.Path(),
    help="Output file path (defaults to stdout for json/markdown)"
)
@click.option(
    "--detailed", "-d",
    is_flag=True,
    help="Show detailed file-by-file analysis"
)
@click.option(
    "--max-files", "-m",
    type=int,
    default=500,
    help="Maximum number of files to analyze"
)
@click.option(
    "--enable-ml",
    is_flag=True,
    help="Enable ML-based detection (requires model download)"
)
@click.option(
    "--no-commits",
    is_flag=True,
    help="Skip commit history analysis"
)
@click.option(
    "--shallow",
    type=int,
    default=100,
    help="Clone depth for remote repositories (0 for full clone)"
)
def analyze(
    repo: str,
    output: str,
    output_file: Optional[str],
    detailed: bool,
    max_files: int,
    enable_ml: bool,
    no_commits: bool,
    shallow: int,
):
    """
    Analyze a repository for AI-generated content.

    REPO can be a GitHub URL or a local path to a repository.

    Examples:

        slop-detector analyze https://github.com/user/repo

        slop-detector analyze ./my-local-repo

        slop-detector analyze https://github.com/user/repo -o json -f report.json

        slop-detector analyze ./repo --detailed --enable-ml
    """
    try:
        # Configure analyzer
        config = AnalyzerConfig(
            max_files=max_files,
            enable_ml=enable_ml,
            enable_commit_analysis=not no_commits,
            clone_depth=shallow if shallow > 0 else None,
        )

        detector = SlopDetector(config=config, console=console)
        reporter = Reporter(console=console)

        # Run analysis
        console.print(f"[bold blue]Analyzing:[/bold blue] {repo}")
        console.print()

        result = detector.analyze_repository(repo)

        # Output results
        if output == "console":
            if detailed:
                reporter.print_detailed(result)
            else:
                reporter.print_summary(result)
        elif output == "json":
            json_output = reporter.to_json(result)
            if output_file:
                Path(output_file).write_text(json_output)
                console.print(f"[green]Report saved to {output_file}[/green]")
            else:
                click.echo(json_output)
        elif output == "markdown":
            md_output = reporter.to_markdown(result)
            if output_file:
                Path(output_file).write_text(md_output)
                console.print(f"[green]Report saved to {output_file}[/green]")
            else:
                click.echo(md_output)

        # Exit with appropriate code
        # 0: Low AI probability (<30%)
        # 1: Medium AI probability (30-70%)
        # 2: High AI probability (>70%)
        if result.overall_ai_probability < 0.3:
            sys.exit(0)
        elif result.overall_ai_probability < 0.7:
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise


@main.command()
@click.argument("text", required=False)
@click.option(
    "--file", "-f",
    type=click.Path(exists=True),
    help="Read content from file"
)
@click.option(
    "--type", "-t",
    "content_type",
    type=click.Choice(["code", "documentation", "configuration"]),
    default="code",
    help="Content type to analyze"
)
@click.option(
    "--enable-ml",
    is_flag=True,
    help="Enable ML-based detection"
)
def check(
    text: Optional[str],
    file: Optional[str],
    content_type: str,
    enable_ml: bool,
):
    """
    Check a single text snippet for AI-generated content.

    Provide text directly or use --file to read from a file.

    Examples:

        slop-detector check "def hello(): print('world')"

        slop-detector check --file ./my_script.py --type code

        echo "Some text" | slop-detector check -
    """
    # Get content
    if text == "-" or (not text and not file):
        # Read from stdin
        content = sys.stdin.read()
    elif file:
        content = Path(file).read_text()
    elif text:
        content = text
    else:
        console.print("[red]Error: Provide text or use --file[/red]")
        sys.exit(1)

    if not content.strip():
        console.print("[red]Error: No content to analyze[/red]")
        sys.exit(1)

    # Map content type
    type_map = {
        "code": ContentType.CODE,
        "documentation": ContentType.DOCUMENTATION,
        "configuration": ContentType.CONFIGURATION,
    }
    ct = type_map[content_type]

    # Analyze
    config = AnalyzerConfig(enable_ml=enable_ml)
    detector = SlopDetector(config=config, console=console)

    result = detector.analyze_text(content, content_type=ct)

    # Output
    reporter = Reporter(console=console)

    color = reporter._get_score_color(result.ai_probability)
    console.print()
    console.print(f"[bold]AI Probability:[/bold] [{color}]{result.ai_probability:.1%}[/{color}]")
    console.print(f"[bold]Confidence:[/bold] {result.confidence.value.replace('_', ' ').title()}")
    console.print()

    if result.signals:
        console.print("[bold]Signals Detected:[/bold]")
        for signal in sorted(result.signals, key=lambda s: s.score * s.weight, reverse=True)[:10]:
            console.print(f"  • {signal.name}: {signal.score:.0%} (weight: {signal.weight:.1f})")
            if signal.evidence:
                for ev in signal.evidence[:2]:
                    console.print(f"    [dim]{ev[:60]}[/dim]")

    # Exit code based on probability
    sys.exit(0 if result.ai_probability < 0.5 else 1)


@main.command()
def info():
    """Show information about slop-detector and its capabilities."""
    console.print()
    console.print("[bold blue]Slop Detector[/bold blue]")
    console.print(f"Version: {__version__}")
    console.print()
    console.print("[bold]Detection Methods:[/bold]")
    console.print()
    console.print("  [cyan]Heuristic Detection[/cyan]")
    console.print("    Pattern matching for known AI writing patterns,")
    console.print("    comment analysis, naming conventions, and code structure.")
    console.print()
    console.print("  [cyan]Statistical Analysis[/cyan]")
    console.print("    Burstiness measurement, Zipf's law analysis,")
    console.print("    vocabulary diversity, and entropy calculations.")
    console.print()
    console.print("  [cyan]ML-Based Detection[/cyan] (optional)")
    console.print("    Transformer-based perplexity analysis using")
    console.print("    CodeBERT or similar models.")
    console.print()
    console.print("  [cyan]Commit Analysis[/cyan]")
    console.print("    Commit message patterns, timing analysis,")
    console.print("    and authorship patterns.")
    console.print()
    console.print("[bold]Supported Content Types:[/bold]")
    console.print("  • Source code (Python, JavaScript, TypeScript, etc.)")
    console.print("  • Documentation (Markdown, RST, etc.)")
    console.print("  • Configuration files (JSON, YAML, TOML, etc.)")
    console.print()
    console.print("[bold]Usage:[/bold]")
    console.print("  slop-detector analyze <repo_url>")
    console.print("  slop-detector check <text>")
    console.print()
    console.print("For more details: slop-detector --help")


if __name__ == "__main__":
    main()
