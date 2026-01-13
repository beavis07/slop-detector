"""
ML-based AI content detection.

This module uses transformer models (CodeBERT, etc.) and
perplexity-based methods for detecting AI-generated content.
"""

import re
from dataclasses import dataclass
from typing import Optional

from ..models import ContentType, DetectionSignal


@dataclass
class MLConfig:
    """Configuration for ML-based detection."""
    # Model selection
    use_local_model: bool = True
    model_name: str = "microsoft/codebert-base"

    # Perplexity thresholds
    code_perplexity_threshold: float = 50.0
    text_perplexity_threshold: float = 30.0

    # Chunking
    max_chunk_size: int = 512
    min_chunk_size: int = 50


class MLDetector:
    """ML-based detection for AI-generated content."""

    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False

    def _lazy_init(self):
        """Lazily initialize the model."""
        if self._initialized:
            return

        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)
            self._model.eval()

            if torch.cuda.is_available():
                self._model = self._model.cuda()

            self._initialized = True
        except ImportError:
            # Models not available, will use fallback methods
            self._initialized = True
        except Exception as e:
            # Model loading failed, will use fallback methods
            print(f"Warning: Could not load ML model: {e}")
            self._initialized = True

    def analyze(self, content: str, content_type: ContentType) -> list[DetectionSignal]:
        """Analyze content using ML methods."""
        self._lazy_init()
        signals = []

        # If no model available, use simpler perplexity estimation
        if self._model is None:
            signals.extend(self._analyze_without_model(content, content_type))
        else:
            signals.extend(self._analyze_with_model(content, content_type))

        return signals

    def _analyze_with_model(self, content: str, content_type: ContentType) -> list[DetectionSignal]:
        """Analyze using loaded transformer model."""
        signals = []

        try:
            import torch

            # Chunk content for analysis
            chunks = self._chunk_content(content)
            if not chunks:
                return signals

            perplexities = []

            for chunk in chunks[:10]:  # Limit chunks for performance
                perplexity = self._calculate_perplexity(chunk)
                if perplexity is not None:
                    perplexities.append(perplexity)

            if perplexities:
                avg_perplexity = sum(perplexities) / len(perplexities)
                threshold = (
                    self.config.code_perplexity_threshold
                    if content_type == ContentType.CODE
                    else self.config.text_perplexity_threshold
                )

                # Lower perplexity can indicate AI generation
                # (model finds the text very predictable)
                if avg_perplexity < threshold * 0.5:
                    ai_score = 0.3 + (threshold * 0.5 - avg_perplexity) / threshold
                    signals.append(DetectionSignal(
                        name="Low perplexity (ML)",
                        score=min(0.85, ai_score),
                        weight=0.7,
                        description=f"Content is highly predictable to language model (perplexity: {avg_perplexity:.1f})",
                        evidence=[f"Average perplexity: {avg_perplexity:.2f}, threshold: {threshold}"]
                    ))
                elif avg_perplexity > threshold * 2:
                    # Very high perplexity suggests human (or unusual AI)
                    signals.append(DetectionSignal(
                        name="High perplexity (ML)",
                        score=0.2,
                        weight=0.6,
                        description=f"Content shows natural variation (perplexity: {avg_perplexity:.1f})",
                        evidence=[f"Average perplexity: {avg_perplexity:.2f}"]
                    ))

        except Exception as e:
            # Fall back to non-model analysis
            signals.extend(self._analyze_without_model(content, content_type))

        return signals

    def _calculate_perplexity(self, text: str) -> Optional[float]:
        """Calculate perplexity of text using the model."""
        if self._model is None or self._tokenizer is None:
            return None

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_chunk_size
            )

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            perplexity = torch.exp(torch.tensor(loss)).item()
            return perplexity

        except Exception:
            return None

    def _analyze_without_model(self, content: str, content_type: ContentType) -> list[DetectionSignal]:
        """
        Fallback analysis when model is not available.

        Uses simpler heuristics that approximate perplexity-like measures.
        """
        signals = []

        # Character-level entropy as a proxy for perplexity
        char_entropy = self._calculate_char_entropy(content)

        # Token prediction patterns
        predictability = self._estimate_predictability(content)

        # AI-generated text tends to have moderate entropy (not too high, not too low)
        if 3.5 < char_entropy < 4.5:
            signals.append(DetectionSignal(
                name="Moderate entropy",
                score=0.4,
                weight=0.3,
                description=f"Character entropy in AI-typical range ({char_entropy:.2f})",
                evidence=[f"Entropy: {char_entropy:.3f}"]
            ))

        if predictability > 0.6:
            signals.append(DetectionSignal(
                name="High predictability",
                score=min(0.7, 0.3 + predictability * 0.5),
                weight=0.5,
                description=f"Content shows high predictability patterns ({predictability:.1%})",
                evidence=[]
            ))

        return signals

    def _calculate_char_entropy(self, content: str) -> float:
        """Calculate character-level entropy."""
        import math
        from collections import Counter

        if not content:
            return 0.0

        counter = Counter(content)
        total = len(content)

        entropy = -sum(
            (count / total) * math.log2(count / total)
            for count in counter.values()
        )

        return entropy

    def _estimate_predictability(self, content: str) -> float:
        """
        Estimate content predictability using simple patterns.

        This is a rough approximation of what a language model would compute.
        """
        # Common bigram patterns in AI-generated text
        ai_bigrams = [
            "the ", " is ", " to ", " of ", " a ", " in ",
            " and", " for", "tion", "ing ", " it ", "this",
            " we ", "you ", " be ", "that", "with", " as ",
            "are ", "was ", "will", "have", " an ", " or ",
        ]

        content_lower = content.lower()
        bigram_count = sum(content_lower.count(bg) for bg in ai_bigrams)
        total_bigrams = max(len(content) - 1, 1)

        predictability = bigram_count / (total_bigrams / 10)
        return min(1.0, predictability)

    def _chunk_content(self, content: str) -> list[str]:
        """Split content into chunks for analysis."""
        # Split by natural boundaries
        if "\n\n" in content:
            chunks = content.split("\n\n")
        else:
            chunks = content.split("\n")

        # Filter by size
        result = []
        current_chunk = ""

        for chunk in chunks:
            if len(current_chunk) + len(chunk) < self.config.max_chunk_size:
                current_chunk += "\n" + chunk if current_chunk else chunk
            else:
                if len(current_chunk) >= self.config.min_chunk_size:
                    result.append(current_chunk)
                current_chunk = chunk

        if len(current_chunk) >= self.config.min_chunk_size:
            result.append(current_chunk)

        return result
