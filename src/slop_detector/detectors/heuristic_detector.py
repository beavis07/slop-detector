"""
Heuristic-based AI content detection.

This module contains pattern matching and rule-based detection for AI-generated content.
These heuristics are based on known patterns from various LLMs (GPT, Claude, Copilot, etc.)
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..models import ContentType, DetectionSignal


@dataclass
class HeuristicConfig:
    """Configuration for heuristic detection."""
    # Minimum content length to analyze
    min_content_length: int = 50
    # Minimum matches to report a pattern
    min_pattern_matches: int = 2


class HeuristicDetector:
    """Detect AI-generated content using pattern matching and heuristics."""

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.config = config or HeuristicConfig()

    def analyze(self, content: str, content_type: ContentType, filename: str = "") -> list[DetectionSignal]:
        """Analyze content and return detection signals."""
        signals = []

        if len(content) < self.config.min_content_length:
            return signals

        # Common AI patterns (apply to all content types)
        signals.extend(self._detect_ai_phrases(content))

        # Content-type specific detection
        if content_type == ContentType.CODE:
            signals.extend(self._detect_code_patterns(content, filename))
        elif content_type == ContentType.DOCUMENTATION:
            signals.extend(self._detect_doc_patterns(content))
        elif content_type == ContentType.CONFIGURATION:
            signals.extend(self._detect_config_patterns(content))

        return signals

    def _detect_ai_phrases(self, content: str) -> list[DetectionSignal]:
        """Detect common AI-generated phrases."""
        signals = []
        content_lower = content.lower()

        # Phrases commonly used by AI assistants
        ai_phrases = [
            # Introduction phrases
            (r"\bhere(?:'s| is) (?:a|an|the) (?:simple |basic |complete )?(?:example|implementation|solution|code|function|class)\b", "AI intro phrase", 0.7),
            (r"\blet me (?:explain|show|help|create|write)\b", "AI conversational phrase", 0.8),
            (r"\bi(?:'ll| will) (?:create|write|implement|show)\b", "AI action phrase", 0.75),
            (r"\bas (?:requested|mentioned|you can see)\b", "AI reference phrase", 0.6),

            # Explanation phrases
            (r"\bthis (?:function|class|method|code|implementation) (?:will |does |is designed to )", "AI explanation pattern", 0.65),
            (r"\bthe above (?:code|function|implementation|example)\b", "AI reference pattern", 0.6),
            (r"\bnote that\b", "AI note pattern", 0.3),
            (r"\bkeep in mind\b", "AI reminder pattern", 0.4),
            (r"\bimportant(?:ly)?:\s", "AI emphasis pattern", 0.4),

            # Structure phrases
            (r"\bfirst,?\s+(?:we |you |let's |I )", "AI step pattern", 0.5),
            (r"\bnext,?\s+(?:we |you |let's |I )", "AI step pattern", 0.5),
            (r"\bfinally,?\s+(?:we |you |let's |I )", "AI step pattern", 0.5),

            # Politeness markers (uncommon in human code comments)
            (r"\bhope this helps\b", "AI politeness", 0.9),
            (r"\bfeel free to\b", "AI politeness", 0.7),
            (r"\bdon't hesitate to\b", "AI politeness", 0.8),
            (r"\blet me know if\b", "AI politeness", 0.75),
            (r"\bhappy to help\b", "AI politeness", 0.85),

            # Over-explanation patterns
            (r"(?:this|which) (?:ensures?|guarantees?|makes? sure) that\b", "AI over-explanation", 0.5),
            (r"\bfor (?:better |improved )?(?:readability|maintainability|performance|clarity)\b", "AI justification", 0.4),

            # Placeholder patterns often left by AI
            (r"\bYOUR_[A-Z_]+_HERE\b", "AI placeholder pattern", 0.85),
            (r"\b(?:your|my)[-_]?(?:api[-_]?key|token|secret)\b", "AI placeholder pattern", 0.7),
            (r"<your[_-].*?>", "AI placeholder pattern", 0.85),
            (r"\breplace (?:this |with )?\byour\b", "AI placeholder instruction", 0.75),
        ]

        for pattern, name, weight in ai_phrases:
            matches = re.findall(pattern, content_lower)
            if matches:
                score = min(0.95, 0.3 + len(matches) * 0.15)
                signals.append(DetectionSignal(
                    name=name,
                    score=score,
                    weight=weight,
                    description=f"Found {len(matches)} instance(s) of '{name}'",
                    evidence=[m if isinstance(m, str) else m[0] for m in matches[:3]]
                ))

        return signals

    def _detect_code_patterns(self, content: str, filename: str = "") -> list[DetectionSignal]:
        """Detect AI patterns specific to code."""
        signals = []
        lines = content.split("\n")

        # 1. Check comment patterns
        signals.extend(self._analyze_comments(content, lines))

        # 2. Check code structure patterns
        signals.extend(self._analyze_code_structure(lines))

        # 3. Check variable naming patterns
        signals.extend(self._analyze_naming(content))

        # 4. Check for AI-specific code patterns
        signals.extend(self._analyze_ai_code_patterns(content))

        # 5. Check docstring patterns
        signals.extend(self._analyze_docstrings(content))

        return signals

    def _analyze_comments(self, content: str, lines: list[str]) -> list[DetectionSignal]:
        """Analyze comment patterns for AI signals."""
        signals = []

        # Comment extraction (handles multiple languages)
        single_line_comments = []
        for line in lines:
            stripped = line.strip()
            for prefix in ["//", "#", "--", ";"]:
                if stripped.startswith(prefix) and not stripped.startswith(prefix * 2):
                    single_line_comments.append(stripped[len(prefix):].strip())
                    break

        if not single_line_comments:
            return signals

        # Pattern: Over-commenting (AI tends to comment every line)
        code_lines = [l for l in lines if l.strip() and not any(
            l.strip().startswith(p) for p in ["//", "#", "--", ";", "/*", "*", "'''", '"""']
        )]
        comment_ratio = len(single_line_comments) / max(len(code_lines), 1)

        if comment_ratio > 0.5:
            signals.append(DetectionSignal(
                name="High comment ratio",
                score=min(0.8, comment_ratio * 0.6),
                weight=0.6,
                description=f"Comment ratio is {comment_ratio:.1%} (AI tends to over-comment)",
                evidence=[f"Ratio: {len(single_line_comments)}/{len(code_lines)}"]
            ))

        # Pattern: Comments that explain the obvious
        obvious_patterns = [
            (r"^(?:initialize|initializing|init)\s+(?:the\s+)?(?:variable|value|array|list|dict)", "Obvious initialization comment"),
            (r"^(?:increment|decrement|add|subtract)\s+(?:the\s+)?(?:counter|value|number)", "Obvious operation comment"),
            (r"^(?:return|returns?)\s+(?:the\s+)?(?:result|value|data|response)", "Obvious return comment"),
            (r"^(?:loop|iterate|iterating)\s+(?:through|over)\s+(?:the\s+)?(?:array|list|items)", "Obvious loop comment"),
            (r"^(?:check|checking)\s+if\s+", "Obvious check comment"),
            (r"^(?:get|set|create|delete|remove|update|add)\s+(?:the\s+)?(?:user|data|item|value)", "Obvious CRUD comment"),
        ]

        obvious_count = 0
        for comment in single_line_comments:
            comment_lower = comment.lower()
            for pattern, _ in obvious_patterns:
                if re.search(pattern, comment_lower):
                    obvious_count += 1
                    break

        if obvious_count >= 3:
            signals.append(DetectionSignal(
                name="Obvious comments",
                score=min(0.85, 0.3 + obvious_count * 0.1),
                weight=0.7,
                description=f"Found {obvious_count} comments stating the obvious",
                evidence=[]
            ))

        # Pattern: Uniform comment style (AI is very consistent)
        if len(single_line_comments) >= 5:
            # Check capitalization consistency
            capitalized = sum(1 for c in single_line_comments if c and c[0].isupper())
            cap_ratio = capitalized / len(single_line_comments)
            if cap_ratio > 0.9 or cap_ratio < 0.1:
                signals.append(DetectionSignal(
                    name="Uniform comment capitalization",
                    score=0.5,
                    weight=0.4,
                    description="Comments have unusually consistent capitalization style",
                    evidence=[f"Capitalized: {cap_ratio:.0%}"]
                ))

        return signals

    def _analyze_code_structure(self, lines: list[str]) -> list[DetectionSignal]:
        """Analyze code structure patterns."""
        signals = []

        # Pattern: Consistent indentation (AI is very consistent)
        indents = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.append(indent)

        if len(indents) >= 10:
            unique_indents = set(indents)
            # AI typically uses exactly 2 or 4 space indents, nothing irregular
            irregular = [i for i in unique_indents if i % 2 != 0]
            if not irregular and len(unique_indents) <= 4:
                signals.append(DetectionSignal(
                    name="Perfect indentation",
                    score=0.4,
                    weight=0.3,
                    description="Indentation is unusually consistent (only multiples of 2)",
                    evidence=[f"Indent levels: {sorted(unique_indents)}"]
                ))

        # Pattern: Blank line usage (AI has consistent patterns)
        blank_sequences = []
        current_blank = 0
        for line in lines:
            if not line.strip():
                current_blank += 1
            else:
                if current_blank > 0:
                    blank_sequences.append(current_blank)
                current_blank = 0

        if len(blank_sequences) >= 5:
            # AI typically uses exactly 1 or 2 blank lines, never irregular numbers
            if all(b in [1, 2] for b in blank_sequences):
                signals.append(DetectionSignal(
                    name="Consistent blank line usage",
                    score=0.35,
                    weight=0.25,
                    description="Blank line usage is unusually uniform",
                    evidence=[f"Pattern: {blank_sequences[:10]}"]
                ))

        return signals

    def _analyze_naming(self, content: str) -> list[DetectionSignal]:
        """Analyze variable and function naming patterns."""
        signals = []

        # Pattern: Overly descriptive names (AI tends toward verbose names)
        verbose_patterns = [
            r"\b[a-z]+(?:[A-Z][a-z]+){4,}\b",  # camelCase with 5+ words
            r"\b[a-z]+(?:_[a-z]+){4,}\b",       # snake_case with 5+ words
        ]

        verbose_count = 0
        for pattern in verbose_patterns:
            matches = re.findall(pattern, content)
            verbose_count += len(matches)

        if verbose_count >= 3:
            signals.append(DetectionSignal(
                name="Verbose naming",
                score=min(0.7, 0.3 + verbose_count * 0.08),
                weight=0.5,
                description=f"Found {verbose_count} overly descriptive variable names",
                evidence=[]
            ))

        # Pattern: Textbook-style names
        textbook_names = [
            r"\b(?:result|data|response|output|input|value|item|element|temp|tmp)\d*\b",
            r"\b(?:arr|lst|dict|obj|str|num|idx|cnt|len|max|min)(?:[A-Z]|_|$)",
            r"\b(?:isValid|isActive|isEnabled|hasPermission|canAccess)\b",
            r"\b(?:handleClick|handleSubmit|handleChange|handleError)\b",
            r"\b(?:fetchData|getData|setData|updateData|deleteData)\b",
        ]

        textbook_count = 0
        for pattern in textbook_names:
            textbook_count += len(re.findall(pattern, content))

        # Only flag if there are many such names
        if textbook_count >= 10:
            signals.append(DetectionSignal(
                name="Textbook naming conventions",
                score=min(0.6, 0.2 + textbook_count * 0.03),
                weight=0.4,
                description=f"High use of generic/textbook variable names ({textbook_count} found)",
                evidence=[]
            ))

        return signals

    def _analyze_ai_code_patterns(self, content: str) -> list[DetectionSignal]:
        """Analyze code patterns specific to AI generation."""
        signals = []

        # Pattern: Try-catch wrapping everything (AI overuses error handling)
        try_blocks = len(re.findall(r"\btry\s*[{:]", content))
        catch_blocks = len(re.findall(r"\b(?:catch|except)\b", content))

        lines = content.split("\n")
        code_lines = len([l for l in lines if l.strip()])

        if code_lines > 20 and try_blocks > 0:
            try_ratio = try_blocks / (code_lines / 20)
            if try_ratio > 0.5:
                signals.append(DetectionSignal(
                    name="Excessive error handling",
                    score=min(0.6, try_ratio * 0.4),
                    weight=0.4,
                    description=f"High ratio of try-catch blocks ({try_blocks} in {code_lines} lines)",
                    evidence=[]
                ))

        # Pattern: Console.log/print for debugging (AI often adds these)
        debug_patterns = [
            r"console\.log\s*\(",
            r"\bprint\s*\(",
            r"System\.out\.println\s*\(",
            r"fmt\.Println\s*\(",
            r"puts\s+['\"]",
        ]

        debug_count = 0
        for pattern in debug_patterns:
            debug_count += len(re.findall(pattern, content))

        if debug_count >= 5:
            signals.append(DetectionSignal(
                name="Debug statements",
                score=min(0.5, 0.2 + debug_count * 0.05),
                weight=0.3,
                description=f"Multiple debug/log statements found ({debug_count})",
                evidence=[]
            ))

        # Pattern: TODO comments with specific phrasing
        todo_patterns = re.findall(
            r"(?:TODO|FIXME|NOTE|HACK|XXX):\s*(.{10,60})",
            content,
            re.IGNORECASE
        )

        ai_todo_phrases = [
            "implement", "add error handling", "add validation",
            "optimize", "refactor", "improve", "consider"
        ]

        ai_todos = 0
        for todo in todo_patterns:
            if any(phrase in todo.lower() for phrase in ai_todo_phrases):
                ai_todos += 1

        if ai_todos >= 2:
            signals.append(DetectionSignal(
                name="AI-style TODO comments",
                score=min(0.7, 0.3 + ai_todos * 0.15),
                weight=0.5,
                description=f"Found {ai_todos} TODOs with AI-typical phrasing",
                evidence=todo_patterns[:3]
            ))

        return signals

    def _analyze_docstrings(self, content: str) -> list[DetectionSignal]:
        """Analyze docstring patterns."""
        signals = []

        # Extract docstrings (Python style)
        docstrings = re.findall(
            r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')' ,
            content,
            re.DOTALL
        )

        if not docstrings:
            return signals

        # Pattern: Docstrings that follow exact templates
        template_patterns = [
            r"Args:\s*\n\s+\w+.*?:\s+.+",
            r"Returns:\s*\n\s+.+",
            r"Raises:\s*\n\s+\w+.*?:\s+.+",
            r"Parameters\s*\n\s*-+\s*\n",
            r"Example(?:s)?:\s*\n\s+>>>",
        ]

        template_matches = 0
        for doc in docstrings:
            for pattern in template_patterns:
                if re.search(pattern, doc):
                    template_matches += 1

        if template_matches >= 3:
            signals.append(DetectionSignal(
                name="Templated docstrings",
                score=min(0.7, 0.3 + template_matches * 0.1),
                weight=0.5,
                description=f"Docstrings follow consistent AI-style templates",
                evidence=[]
            ))

        # Pattern: Overly detailed docstrings
        for doc in docstrings:
            words = doc.split()
            if len(words) > 100:
                signals.append(DetectionSignal(
                    name="Verbose docstrings",
                    score=0.5,
                    weight=0.4,
                    description="Found unusually detailed docstrings",
                    evidence=[f"Docstring with {len(words)} words"]
                ))
                break

        return signals

    def _detect_doc_patterns(self, content: str) -> list[DetectionSignal]:
        """Detect AI patterns in documentation."""
        signals = []
        content_lower = content.lower()

        # Markdown-specific patterns
        md_ai_patterns = [
            # Overly structured headers
            (r"^## (?:Overview|Introduction|Getting Started|Prerequisites|Installation|Usage|Features|Configuration|API|Examples|Contributing|License)\s*$",
             "AI-style section headers", 0.4),

            # Emoji usage (AI loves to use emojis in docs)
            (r"[\U0001F300-\U0001F9FF]", "Emoji usage", 0.3),

            # Badges in a row (AI-generated READMEs often have many badges)
            (r"\[!\[.*?\]\(.*?\)\]\(.*?\)(?:\s*\[!\[.*?\]\(.*?\)\]\(.*?\)){2,}",
             "Multiple badges pattern", 0.5),

            # Table of contents with exact formatting
            (r"## Table of Contents\s*\n(?:\s*-\s*\[.*?\]\(#.*?\)\s*\n){5,}",
             "Structured table of contents", 0.5),

            # AI-style feature lists
            (r"(?:^|\n)(?:[*-]\s+\*\*.*?\*\*:?\s+.*?\n){3,}",
             "Formatted feature list", 0.4),
        ]

        for pattern, name, weight in md_ai_patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
            if matches:
                signals.append(DetectionSignal(
                    name=name,
                    score=min(0.8, 0.3 + len(matches) * 0.1),
                    weight=weight,
                    description=f"Found {len(matches)} instance(s) of {name}",
                    evidence=[]
                ))

        # Check for AI-generated prose patterns
        prose_patterns = [
            (r"\b(?:comprehensive|robust|seamless|leverage|utilize|streamline)\b", "AI buzzwords", 0.4),
            (r"\bThis (?:project|repository|package|library|tool) (?:provides|offers|includes|features)\b", "AI intro phrase", 0.5),
            (r"\b(?:easy[- ]to[- ]use|user[- ]friendly|out[- ]of[- ]the[- ]box)\b", "AI marketing phrases", 0.4),
            (r"\bwhether you(?:'re| are) (?:a |looking |building |developing )", "AI audience targeting", 0.6),
        ]

        for pattern, name, weight in prose_patterns:
            matches = re.findall(pattern, content_lower)
            if len(matches) >= 2:
                signals.append(DetectionSignal(
                    name=name,
                    score=min(0.7, 0.2 + len(matches) * 0.1),
                    weight=weight,
                    description=f"Found {len(matches)} instances of {name}",
                    evidence=matches[:3]
                ))

        # Sentence structure analysis
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if len(sentences) >= 5:
            # Check for uniform sentence length (AI is very consistent)
            lengths = [len(s.split()) for s in sentences]
            avg_len = sum(lengths) / len(lengths)
            variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

            if variance < 50:  # Low variance = suspiciously uniform
                signals.append(DetectionSignal(
                    name="Uniform sentence length",
                    score=0.5,
                    weight=0.4,
                    description="Sentences have unusually consistent length",
                    evidence=[f"Average: {avg_len:.1f} words, variance: {variance:.1f}"]
                ))

        return signals

    def _detect_config_patterns(self, content: str) -> list[DetectionSignal]:
        """Detect AI patterns in configuration files."""
        signals = []

        # Config files are harder to detect as AI-generated
        # Focus on comments within config files

        # Pattern: Verbose comments in JSON (unusual for hand-written)
        if "/*" in content or "//" in content:
            comment_blocks = re.findall(r'/\*.*?\*/', content, re.DOTALL)
            comment_blocks.extend(re.findall(r'//.*$', content, re.MULTILINE))

            if len(comment_blocks) >= 5:
                signals.append(DetectionSignal(
                    name="Verbose config comments",
                    score=0.4,
                    weight=0.3,
                    description="Configuration file has unusual number of comments",
                    evidence=[f"{len(comment_blocks)} comment blocks"]
                ))

        # Pattern: Placeholder values
        placeholders = re.findall(
            r'["\'](?:your[-_]?|my[-_]?|example[-_]?|sample[-_]?)[\w-]+["\']',
            content,
            re.IGNORECASE
        )

        if len(placeholders) >= 3:
            signals.append(DetectionSignal(
                name="Placeholder values",
                score=min(0.7, 0.3 + len(placeholders) * 0.1),
                weight=0.6,
                description=f"Found {len(placeholders)} placeholder values",
                evidence=placeholders[:3]
            ))

        return signals
