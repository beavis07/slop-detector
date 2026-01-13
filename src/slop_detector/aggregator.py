"""
Aggregation and scoring system.

Combines signals from multiple detectors into overall scores.
"""

from dataclasses import dataclass
from typing import Optional

from .models import (
    CategorySummary,
    Confidence,
    ContentType,
    DetectionSignal,
    FileAnalysis,
    RepositoryAnalysis,
)


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregation."""
    # Weights for different detector types
    heuristic_weight: float = 0.35
    statistical_weight: float = 0.30
    ml_weight: float = 0.35

    # Weights for content types in overall score
    code_weight: float = 0.5
    docs_weight: float = 0.3
    config_weight: float = 0.2

    # Confidence calculation
    min_signals_for_high_confidence: int = 5
    min_files_for_high_confidence: int = 3


class SignalAggregator:
    """Aggregate detection signals into scores."""

    def __init__(self, config: Optional[AggregatorConfig] = None):
        self.config = config or AggregatorConfig()

    def aggregate_file_signals(
        self,
        signals: list[DetectionSignal],
        content_type: ContentType,
        line_count: int = 0
    ) -> FileAnalysis:
        """
        Aggregate signals for a single file into a score.

        Returns:
            FileAnalysis with computed probability and confidence.
        """
        if not signals:
            return FileAnalysis(
                path=None,  # Will be set by caller
                content_type=content_type,
                ai_probability=0.0,
                confidence=Confidence.VERY_LOW,
                signals=[],
                line_count=line_count,
                ai_line_estimate=0
            )

        # Check for DEFINITIVE signals first - these override normal weighting
        definitive_signals = [s for s in signals if s.name.startswith("DEFINITIVE:") or s.weight >= 0.9]

        if definitive_signals:
            # If we have definitive signals, they dominate the calculation
            # Use max of definitive scores as floor
            max_definitive = max(s.score for s in definitive_signals)

            # Calculate normal weighted average for remaining signals
            other_signals = [s for s in signals if s not in definitive_signals]
            if other_signals:
                other_weight = sum(s.weight for s in other_signals)
                other_sum = sum(s.score * s.weight for s in other_signals)
                other_avg = other_sum / other_weight if other_weight > 0 else 0.5
            else:
                other_avg = 0.5

            # Definitive signals set the floor, other signals can only push it higher
            ai_probability = max(max_definitive, other_avg * 0.3 + max_definitive * 0.7)

            # Definitive signals = high confidence
            confidence = Confidence.VERY_HIGH
        else:
            # Normal weighted average calculation
            total_weight = sum(s.weight for s in signals)
            if total_weight == 0:
                ai_probability = 0.0
            else:
                weighted_sum = sum(s.score * s.weight for s in signals)
                ai_probability = weighted_sum / total_weight

            # Apply signal count adjustment
            # More signals = more confident in either direction
            signal_count = len(signals)
            if signal_count >= 10:
                # Many signals - push toward extremes
                if ai_probability > 0.5:
                    ai_probability = ai_probability + (1 - ai_probability) * 0.1
                else:
                    ai_probability = ai_probability * 0.9

            # Calculate confidence based on signal quality
            confidence = self._calculate_confidence(signals, ai_probability)

        # Estimate AI lines
        ai_line_estimate = int(line_count * ai_probability)

        return FileAnalysis(
            path=None,
            content_type=content_type,
            ai_probability=ai_probability,
            confidence=confidence,
            signals=signals,
            line_count=line_count,
            ai_line_estimate=ai_line_estimate
        )

    def _calculate_confidence(
        self,
        signals: list[DetectionSignal],
        probability: float
    ) -> Confidence:
        """Calculate confidence level based on signals."""
        if not signals:
            return Confidence.VERY_LOW

        # Factors that increase confidence:
        # 1. Number of signals
        # 2. Agreement between signals
        # 3. High-weight signals
        # 4. Extreme probability (near 0 or 1)

        # Signal count factor
        signal_factor = min(1.0, len(signals) / self.config.min_signals_for_high_confidence)

        # Agreement factor (how much signals agree)
        scores = [s.score for s in signals]
        if len(scores) > 1:
            import numpy as np
            score_std = np.std(scores)
            agreement_factor = 1.0 - min(1.0, score_std / 0.3)
        else:
            agreement_factor = 0.5

        # Weight quality factor
        high_weight_signals = sum(1 for s in signals if s.weight >= 0.5)
        weight_factor = min(1.0, high_weight_signals / 3)

        # Extremity factor
        extremity = abs(probability - 0.5) * 2
        extremity_factor = extremity

        # Combine factors
        confidence_score = (
            signal_factor * 0.3 +
            agreement_factor * 0.3 +
            weight_factor * 0.2 +
            extremity_factor * 0.2
        )

        return Confidence.from_score(confidence_score)

    def aggregate_category(
        self,
        file_analyses: list[FileAnalysis],
        content_type: ContentType
    ) -> Optional[CategorySummary]:
        """Aggregate file analyses for a content category."""
        relevant = [f for f in file_analyses if f.content_type == content_type]

        if not relevant:
            return None

        total_files = len(relevant)
        total_lines = sum(f.line_count for f in relevant)

        # Weight by file size for overall probability
        if total_lines > 0:
            weighted_prob = sum(
                f.ai_probability * f.line_count
                for f in relevant
            ) / total_lines
        else:
            weighted_prob = sum(f.ai_probability for f in relevant) / total_files

        ai_line_estimate = sum(f.ai_line_estimate for f in relevant)

        # Collect top signals across all files
        all_signals = []
        for f in relevant:
            all_signals.extend(f.signals)

        # Get unique signal names by highest score
        signal_scores: dict[str, float] = {}
        for s in all_signals:
            if s.name not in signal_scores or s.score > signal_scores[s.name]:
                signal_scores[s.name] = s.score

        top_signals = sorted(
            signal_scores.keys(),
            key=lambda k: signal_scores[k],
            reverse=True
        )[:5]

        # Calculate confidence for category
        confidence = self._calculate_category_confidence(relevant, weighted_prob)

        return CategorySummary(
            content_type=content_type,
            total_files=total_files,
            total_lines=total_lines,
            ai_probability=weighted_prob,
            ai_line_estimate=ai_line_estimate,
            confidence=confidence,
            top_signals=top_signals
        )

    def _calculate_category_confidence(
        self,
        analyses: list[FileAnalysis],
        probability: float
    ) -> Confidence:
        """Calculate confidence for a category."""
        if len(analyses) < self.config.min_files_for_high_confidence:
            return Confidence.LOW

        # Check agreement between files
        import numpy as np
        probs = [a.ai_probability for a in analyses]
        prob_std = np.std(probs)

        # More agreement = higher confidence
        agreement = 1.0 - min(1.0, prob_std / 0.3)

        # More files = higher confidence
        file_factor = min(1.0, len(analyses) / 10)

        confidence_score = agreement * 0.6 + file_factor * 0.4

        return Confidence.from_score(confidence_score)

    def aggregate_repository(
        self,
        file_analyses: list[FileAnalysis],
        commit_signals: list[DetectionSignal],
        repo_url: str,
        repo_name: str,
        analysis_duration: float = 0.0
    ) -> RepositoryAnalysis:
        """Aggregate all analyses into a repository-level result."""

        # Category summaries
        code_analysis = self.aggregate_category(file_analyses, ContentType.CODE)
        docs_analysis = self.aggregate_category(file_analyses, ContentType.DOCUMENTATION)
        config_analysis = self.aggregate_category(file_analyses, ContentType.CONFIGURATION)

        # Overall probability (weighted by content type importance)
        category_probs = []
        category_weights = []

        if code_analysis:
            category_probs.append(code_analysis.ai_probability)
            category_weights.append(self.config.code_weight * code_analysis.total_lines)

        if docs_analysis:
            category_probs.append(docs_analysis.ai_probability)
            category_weights.append(self.config.docs_weight * docs_analysis.total_lines)

        if config_analysis:
            category_probs.append(config_analysis.ai_probability)
            category_weights.append(self.config.config_weight * config_analysis.total_lines)

        # Check for DEFINITIVE signals in commits/files - these should dominate
        all_signals_for_definitive = []
        for f in file_analyses:
            all_signals_for_definitive.extend(f.signals)
        all_signals_for_definitive.extend(commit_signals)

        definitive_signals = [
            s for s in all_signals_for_definitive
            if s.name.startswith("DEFINITIVE:") or s.weight >= 0.9
        ]

        # Include commit signals
        if commit_signals:
            commit_total_weight = sum(s.weight for s in commit_signals)
            if commit_total_weight > 0:
                commit_prob = sum(s.score * s.weight for s in commit_signals) / commit_total_weight
            else:
                commit_prob = 0.0

            # If there are definitive signals in commits, weight them MUCH higher
            definitive_commit_signals = [s for s in commit_signals if s.name.startswith("DEFINITIVE:") or s.weight >= 0.9]
            if definitive_commit_signals:
                # Definitive commit signals get very high weight
                category_probs.append(commit_prob)
                category_weights.append(2.0)  # High weight when definitive
            else:
                category_probs.append(commit_prob)
                category_weights.append(0.1)  # Lower weight for heuristic commit analysis

        if category_weights:
            total_weight = sum(category_weights)
            overall_probability = sum(
                p * w for p, w in zip(category_probs, category_weights)
            ) / total_weight
        else:
            overall_probability = 0.0

        # If we have definitive signals, ensure the probability reflects that
        if definitive_signals:
            max_definitive_score = max(s.score for s in definitive_signals)
            # Definitive signals set a floor on the overall probability
            overall_probability = max(overall_probability, max_definitive_score * 0.85)

        # Overall confidence
        confidences = []
        if code_analysis:
            confidences.append(code_analysis.confidence)
        if docs_analysis:
            confidences.append(docs_analysis.confidence)
        if config_analysis:
            confidences.append(config_analysis.confidence)

        if confidences:
            # Average confidence levels
            conf_values = [
                {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}[c.value]
                for c in confidences
            ]
            avg_conf = sum(conf_values) / len(conf_values)
            overall_confidence = Confidence.from_score(avg_conf / 4)
        else:
            overall_confidence = Confidence.VERY_LOW

        # Definitive signals = high confidence regardless of other factors
        if definitive_signals:
            overall_confidence = Confidence.VERY_HIGH

        # Collect top signals
        all_signals = []
        for f in file_analyses:
            all_signals.extend(f.signals)
        all_signals.extend(commit_signals)

        # AI signals (high score)
        ai_signals = sorted(
            [s for s in all_signals if s.score > 0.5],
            key=lambda s: s.score * s.weight,
            reverse=True
        )
        top_ai_signals = [s.name for s in ai_signals[:10]]

        # Human signals (low score)
        human_signals = sorted(
            [s for s in all_signals if s.score < 0.4],
            key=lambda s: (1 - s.score) * s.weight,
            reverse=True
        )
        top_human_signals = [s.name for s in human_signals[:5]]

        # Count AI-attributed commits
        ai_commit_count = 0
        for signal in commit_signals:
            if "commit" in signal.name.lower() and signal.score > 0.8:
                ai_commit_count += 1

        return RepositoryAnalysis(
            repo_url=repo_url,
            repo_name=repo_name,
            overall_ai_probability=overall_probability,
            overall_confidence=overall_confidence,
            code_analysis=code_analysis,
            docs_analysis=docs_analysis,
            config_analysis=config_analysis,
            file_analyses=file_analyses,
            commit_analyses=[],  # Populated separately
            total_files=len(file_analyses),
            total_lines=sum(f.line_count for f in file_analyses),
            analysis_duration_seconds=analysis_duration,
            top_ai_signals=top_ai_signals,
            top_human_signals=top_human_signals,
            definitive_signals=definitive_signals,
            commit_signals=commit_signals,
            ai_attributed_commits=ai_commit_count,
        )
