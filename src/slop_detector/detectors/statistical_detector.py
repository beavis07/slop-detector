"""
Statistical analysis for AI content detection.

This module uses statistical methods like perplexity, burstiness,
and linguistic analysis to detect AI-generated content.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..models import ContentType, DetectionSignal


@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis."""
    min_tokens: int = 50
    ngram_sizes: tuple[int, ...] = (2, 3, 4)


class StatisticalDetector:
    """Detect AI-generated content using statistical analysis."""

    def __init__(self, config: Optional[StatisticalConfig] = None):
        self.config = config or StatisticalConfig()

    def analyze(self, content: str, content_type: ContentType) -> list[DetectionSignal]:
        """Analyze content and return detection signals."""
        signals = []

        tokens = self._tokenize(content, content_type)
        if len(tokens) < self.config.min_tokens:
            return signals

        # Burstiness analysis (human text is more "bursty")
        signals.extend(self._analyze_burstiness(tokens))

        # Zipf's law compliance
        signals.extend(self._analyze_zipf(tokens))

        # N-gram repetition patterns
        signals.extend(self._analyze_ngram_repetition(tokens))

        # Vocabulary richness
        signals.extend(self._analyze_vocabulary(tokens))

        # Sentence/line structure for code
        if content_type == ContentType.CODE:
            signals.extend(self._analyze_code_entropy(content))
        else:
            signals.extend(self._analyze_prose_patterns(content))

        return signals

    def _tokenize(self, content: str, content_type: ContentType) -> list[str]:
        """Tokenize content based on type."""
        if content_type == ContentType.CODE:
            # Code tokenization: split on whitespace and operators
            tokens = re.findall(r'\b\w+\b|[^\w\s]', content.lower())
        else:
            # Natural language tokenization
            tokens = re.findall(r'\b\w+\b', content.lower())
        return tokens

    def _analyze_burstiness(self, tokens: list[str]) -> list[DetectionSignal]:
        """
        Analyze burstiness of token usage.

        Human text tends to use words in bursts - when a word is used,
        it's more likely to be used again soon. AI text tends to be more uniform.
        """
        signals = []

        if len(tokens) < 100:
            return signals

        # Calculate inter-arrival times for common tokens
        token_positions: dict[str, list[int]] = {}
        for i, token in enumerate(tokens):
            if token not in token_positions:
                token_positions[token] = []
            token_positions[token].append(i)

        # Filter to tokens that appear multiple times
        frequent_tokens = {
            k: v for k, v in token_positions.items()
            if len(v) >= 3 and len(k) > 2
        }

        if len(frequent_tokens) < 10:
            return signals

        # Calculate burstiness for each token
        burstiness_scores = []
        for token, positions in frequent_tokens.items():
            if len(positions) < 3:
                continue

            # Inter-arrival times
            intervals = [
                positions[i+1] - positions[i]
                for i in range(len(positions) - 1)
            ]

            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)

            if mean_interval > 0:
                # Burstiness: B = (σ - μ) / (σ + μ)
                # Range: -1 (regular/periodic) to 1 (bursty)
                burstiness = (std_interval - mean_interval) / (std_interval + mean_interval)
                burstiness_scores.append(burstiness)

        if burstiness_scores:
            avg_burstiness = np.mean(burstiness_scores)

            # AI tends to have lower burstiness (more uniform distribution)
            # Human text: avg burstiness around 0.3-0.6
            # AI text: avg burstiness around -0.2 to 0.2

            if avg_burstiness < 0.15:
                ai_score = 0.3 + (0.15 - avg_burstiness) * 2
                signals.append(DetectionSignal(
                    name="Low burstiness",
                    score=min(0.8, ai_score),
                    weight=0.6,
                    description=f"Token distribution is unusually uniform (burstiness: {avg_burstiness:.2f})",
                    evidence=[f"Average burstiness: {avg_burstiness:.3f}"]
                ))
            elif avg_burstiness > 0.5:
                # High burstiness suggests human writing
                signals.append(DetectionSignal(
                    name="High burstiness",
                    score=0.2,  # Low AI score (suggests human)
                    weight=0.5,
                    description=f"Token distribution shows natural burstiness ({avg_burstiness:.2f})",
                    evidence=[f"Average burstiness: {avg_burstiness:.3f}"]
                ))

        return signals

    def _analyze_zipf(self, tokens: list[str]) -> list[DetectionSignal]:
        """
        Analyze conformance to Zipf's law.

        Natural language closely follows Zipf's law (frequency inversely
        proportional to rank). AI text may deviate from this pattern.
        """
        signals = []

        if len(tokens) < 200:
            return signals

        # Count token frequencies
        counter = Counter(tokens)
        frequencies = sorted(counter.values(), reverse=True)

        if len(frequencies) < 20:
            return signals

        # Calculate expected Zipf distribution
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks[:50])  # Use top 50 words
        log_freqs = np.log(frequencies[:50])

        # Linear regression to find Zipf exponent
        # Perfect Zipf: log(freq) = -α * log(rank) + C
        try:
            # Fit line
            slope, intercept = np.polyfit(log_ranks, log_freqs, 1)

            # Calculate R² (goodness of fit)
            predicted = slope * log_ranks + intercept
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Zipf's law typically has α ≈ 1 and high R²
            # AI text often has different α or lower R²

            if r_squared < 0.85:
                signals.append(DetectionSignal(
                    name="Poor Zipf fit",
                    score=min(0.7, 0.3 + (0.85 - r_squared) * 2),
                    weight=0.4,
                    description=f"Word frequency distribution deviates from Zipf's law (R²: {r_squared:.2f})",
                    evidence=[f"Zipf exponent: {abs(slope):.2f}, R²: {r_squared:.2f}"]
                ))

            # Check if exponent is unusual
            if abs(slope) < 0.7 or abs(slope) > 1.3:
                signals.append(DetectionSignal(
                    name="Unusual Zipf exponent",
                    score=0.5,
                    weight=0.35,
                    description=f"Zipf exponent ({abs(slope):.2f}) outside normal range (0.7-1.3)",
                    evidence=[f"Expected ~1.0, got {abs(slope):.2f}"]
                ))

        except Exception:
            pass

        return signals

    def _analyze_ngram_repetition(self, tokens: list[str]) -> list[DetectionSignal]:
        """Analyze n-gram patterns for repetition."""
        signals = []

        for n in self.config.ngram_sizes:
            if len(tokens) < n * 10:
                continue

            # Create n-grams
            ngrams = [
                tuple(tokens[i:i+n])
                for i in range(len(tokens) - n + 1)
            ]

            counter = Counter(ngrams)
            total_ngrams = len(ngrams)
            unique_ngrams = len(counter)

            # Calculate repetition ratio
            repetition_ratio = 1 - (unique_ngrams / total_ngrams)

            # AI tends to have more n-gram repetition in certain contexts
            # However, code naturally has more repetition than prose

            # Find most repeated n-grams
            most_common = counter.most_common(5)
            high_repeat = [
                ng for ng, count in most_common
                if count >= 5 and not all(t in {"the", "a", "to", "of", "and", "in", "is"} for t in ng)
            ]

            if len(high_repeat) >= 2:
                signals.append(DetectionSignal(
                    name=f"Repeated {n}-grams",
                    score=min(0.6, 0.2 + len(high_repeat) * 0.1),
                    weight=0.35,
                    description=f"Found {len(high_repeat)} frequently repeated {n}-grams",
                    evidence=[" ".join(ng) for ng in high_repeat[:3]]
                ))

        return signals

    def _analyze_vocabulary(self, tokens: list[str]) -> list[DetectionSignal]:
        """Analyze vocabulary richness and diversity."""
        signals = []

        if len(tokens) < 100:
            return signals

        unique_tokens = set(tokens)
        vocab_ratio = len(unique_tokens) / len(tokens)

        # Type-token ratio (TTR)
        # AI text often has more consistent TTR
        # Human text varies more based on content

        # Calculate moving average TTR to detect consistency
        window_size = 50
        ttr_values = []

        for i in range(0, len(tokens) - window_size, window_size // 2):
            window = tokens[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttr_values.append(ttr)

        if len(ttr_values) >= 4:
            ttr_std = np.std(ttr_values)
            ttr_mean = np.mean(ttr_values)

            # Low variance in TTR suggests AI generation
            if ttr_std < 0.05:
                signals.append(DetectionSignal(
                    name="Uniform vocabulary density",
                    score=min(0.7, 0.3 + (0.05 - ttr_std) * 10),
                    weight=0.5,
                    description=f"Vocabulary density is unusually consistent (std: {ttr_std:.3f})",
                    evidence=[f"TTR mean: {ttr_mean:.3f}, std: {ttr_std:.3f}"]
                ))

        # Check for unusually high or low vocabulary diversity
        if vocab_ratio > 0.8:  # Very diverse vocabulary
            # This could be human or AI, not strongly indicative
            pass
        elif vocab_ratio < 0.3:  # Very repetitive
            signals.append(DetectionSignal(
                name="Low vocabulary diversity",
                score=0.4,
                weight=0.3,
                description=f"Limited vocabulary usage ({vocab_ratio:.1%} unique tokens)",
                evidence=[f"Unique: {len(unique_tokens)}, Total: {len(tokens)}"]
            ))

        return signals

    def _analyze_code_entropy(self, content: str) -> list[DetectionSignal]:
        """Analyze entropy patterns in code."""
        signals = []

        lines = [l for l in content.split("\n") if l.strip()]
        if len(lines) < 10:
            return signals

        # Line length entropy
        lengths = [len(line) for line in lines]
        length_entropy = self._calculate_entropy(lengths)

        # AI-generated code often has more uniform line lengths
        # Calculate normalized entropy
        max_entropy = math.log2(len(set(lengths))) if len(set(lengths)) > 1 else 1
        normalized_entropy = length_entropy / max_entropy if max_entropy > 0 else 0

        if normalized_entropy < 0.7:
            signals.append(DetectionSignal(
                name="Uniform line lengths",
                score=min(0.6, 0.2 + (0.7 - normalized_entropy)),
                weight=0.4,
                description=f"Code line lengths are unusually uniform (entropy: {normalized_entropy:.2f})",
                evidence=[f"Normalized entropy: {normalized_entropy:.3f}"]
            ))

        # Indentation pattern entropy
        indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
        indent_counter = Counter(indents)

        # AI tends to use very consistent indentation
        if len(indent_counter) <= 3 and len(lines) > 20:
            signals.append(DetectionSignal(
                name="Limited indent levels",
                score=0.4,
                weight=0.3,
                description=f"Only {len(indent_counter)} indentation levels in {len(lines)} lines",
                evidence=[f"Levels: {sorted(indent_counter.keys())}"]
            ))

        return signals

    def _analyze_prose_patterns(self, content: str) -> list[DetectionSignal]:
        """Analyze statistical patterns in prose/documentation."""
        signals = []

        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if len(sentences) < 5:
            return signals

        # Sentence length analysis
        lengths = [len(s.split()) for s in sentences]
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)

        # Coefficient of variation (CV)
        cv = std_length / mean_length if mean_length > 0 else 0

        # AI text tends to have lower CV (more uniform sentence lengths)
        # Human writing typically has CV > 0.4
        if cv < 0.35:
            signals.append(DetectionSignal(
                name="Uniform sentence lengths",
                score=min(0.7, 0.3 + (0.35 - cv)),
                weight=0.55,
                description=f"Sentence lengths are unusually consistent (CV: {cv:.2f})",
                evidence=[f"Mean: {mean_length:.1f} words, Std: {std_length:.1f}"]
            ))

        # First word analysis (AI often starts sentences similarly)
        first_words = [s.split()[0].lower() for s in sentences if s.split()]
        first_word_counts = Counter(first_words)

        # Calculate concentration (how many sentences start with top words)
        top_3_count = sum(c for _, c in first_word_counts.most_common(3))
        concentration = top_3_count / len(first_words) if first_words else 0

        if concentration > 0.5:
            signals.append(DetectionSignal(
                name="Repetitive sentence starts",
                score=min(0.65, 0.2 + concentration * 0.5),
                weight=0.45,
                description=f"Many sentences start with same words ({concentration:.0%})",
                evidence=[f"Top: {first_word_counts.most_common(3)}"]
            ))

        return signals

    def _calculate_entropy(self, values: list) -> float:
        """Calculate Shannon entropy of a distribution."""
        counter = Counter(values)
        total = len(values)
        probabilities = [count / total for count in counter.values()]

        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        return entropy
