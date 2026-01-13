# Slop Detector

Detect AI-generated content in GitHub repositories with high accuracy.

## Overview

Slop Detector analyzes repositories to determine:
- **IF** a repository contains AI-generated content (with % probability)
- **HOW MUCH** of the content is AI-generated (estimated %)
- **WHAT TYPE** of content is AI-generated (code vs documentation vs configuration)

## Installation

```bash
pip install -e .
```

For ML-based detection (optional, requires ~500MB model download):
```bash
pip install -e ".[ml]"
```

## Quick Start

Analyze a GitHub repository:
```bash
slop-detector analyze https://github.com/username/repo
```

Analyze a local repository:
```bash
slop-detector analyze ./path/to/repo
```

Check a code snippet:
```bash
slop-detector check "def hello(): print('Hello, World!')"
```

## Output Formats

```bash
# Console output (default)
slop-detector analyze https://github.com/user/repo

# JSON output
slop-detector analyze https://github.com/user/repo -o json -f report.json

# Markdown output
slop-detector analyze https://github.com/user/repo -o markdown -f report.md

# Detailed file-by-file analysis
slop-detector analyze https://github.com/user/repo --detailed
```

## Detection Methods

### 1. Heuristic Detection (Default)
Pattern matching for known AI writing signatures:
- AI-typical phrases ("Here's a simple implementation", "Let me explain")
- Comment patterns (over-commenting, obvious comments)
- Code structure (perfect indentation, uniform formatting)
- Naming conventions (verbose/textbook-style names)
- Placeholder patterns ("YOUR_API_KEY_HERE")

### 2. Statistical Analysis (Default)
Mathematical analysis of content patterns:
- **Burstiness**: Human text has "bursty" word usage; AI is more uniform
- **Zipf's Law**: Natural language follows Zipf's distribution
- **Vocabulary Diversity**: Type-token ratio analysis
- **Entropy Measures**: Character and line-level entropy

### 3. ML-Based Detection (Optional)
Transformer-based analysis using CodeBERT:
- Perplexity scoring (AI text is more "predictable" to LLMs)
- Requires `--enable-ml` flag and model download

### 4. Commit History Analysis (Default)
Git history patterns that suggest AI assistance:
- Commit message patterns
- Bulk commit detection
- Timing anomalies
- AI co-author tags

## Accuracy Considerations

Detection accuracy varies by content type:

| Content Type | Estimated Accuracy | Notes |
|--------------|-------------------|-------|
| Documentation | 70-85% | Most reliable (prose-like) |
| Code | 55-70% | Harder (functional constraints) |
| Configuration | 40-55% | Hardest (highly structured) |

**Important**: No AI detection is 100% accurate. Results should be interpreted as probability indicators, not definitive classifications.

### Factors that reduce accuracy:
- Heavily edited AI content
- Short files (<50 lines)
- Highly templated code
- Multiple authors mixing styles

### Factors that increase accuracy:
- Longer files
- Consistent patterns across repository
- Multiple detection signals agreeing
- Documentation and comments present

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Low AI probability (<30%) |
| 1 | Medium AI probability (30-70%) |
| 2 | High AI probability (>70%) |

Useful for CI/CD integration:
```bash
slop-detector analyze ./repo && echo "Likely human-written"
```

## CLI Options

```
slop-detector analyze [OPTIONS] REPO

Options:
  -o, --output [console|json|markdown]  Output format
  -f, --output-file PATH                Output file path
  -d, --detailed                        Show file-by-file analysis
  -m, --max-files INTEGER               Max files to analyze (default: 500)
  --enable-ml                           Enable ML detection
  --no-commits                          Skip commit analysis
  --shallow INTEGER                     Clone depth (default: 100)
  --help                                Show help
```

## Programmatic Usage

```python
from slop_detector import SlopDetector, AnalyzerConfig
from slop_detector.models import ContentType

# Configure
config = AnalyzerConfig(
    enable_ml=False,
    max_files=200,
)

detector = SlopDetector(config)

# Analyze repository
result = detector.analyze_repository("https://github.com/user/repo")
print(f"AI Probability: {result.overall_ai_probability:.1%}")
print(f"Confidence: {result.overall_confidence.value}")

# Check individual text
analysis = detector.analyze_text(
    "def hello(): print('world')",
    content_type=ContentType.CODE
)
print(f"AI Probability: {analysis.ai_probability:.1%}")
```

## Understanding Results

### Overall AI Probability
A weighted average across all analyzed content, with code weighted more heavily than documentation or configuration.

### Confidence Level
How certain the tool is about its assessment:
- **Very High**: Many signals, strong agreement, extreme probability
- **High**: Good signal count, reasonable agreement
- **Medium**: Some signals, moderate agreement
- **Low**: Few signals or mixed signals
- **Very Low**: Insufficient data for reliable assessment

### Detection Signals
Individual indicators that contribute to the overall score. Each signal has:
- **Score**: 0-1 (how strongly it indicates AI)
- **Weight**: How much to trust this signal
- **Evidence**: Specific examples found

## Limitations

1. **False Positives**: Clean, well-documented code may be flagged
2. **False Negatives**: Heavily edited AI content may pass detection
3. **Content Type Bias**: Better at detecting AI prose than AI code
4. **Training Data**: Based on patterns from GPT/Claude/Copilot circa 2023-2024

## Research Background

This tool is based on research from:
- [DetectCodeGPT](https://github.com/YerbaPage/DetectCodeGPT) (ICSE 2025)
- [CodeGPTSensor](https://dl.acm.org/doi/10.1145/3705300) (ACM TOSEM)
- [aboutcode-org/ai-gen-code-search](https://github.com/aboutcode-org/ai-gen-code-search)
- Statistical analysis methods from computational linguistics

## License

MIT License
