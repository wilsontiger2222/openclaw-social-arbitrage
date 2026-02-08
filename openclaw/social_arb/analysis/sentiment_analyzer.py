"""
Financial sentiment analysis using FinBERT and LLM fallback.

FinBERT (ProsusAI/finbert) runs locally for zero API cost and is
specifically trained on financial text. For complex/ambiguous text,
falls back to OpenAI API.

Also computes a "hype score" — high volume + overly positive sentiment
+ low substance = potential pump & dump warning.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import (
    MentionData,
    SentimentLabel,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAnalyzer:
    """
    Classifies financial text as BULLISH / BEARISH / NEUTRAL.

    Primary: FinBERT (local model, no API cost)
    Fallback: OpenAI API for complex/ambiguous text
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self.batch_size = config.sentiment.batch_size
        self.confidence_threshold = config.sentiment.confidence_threshold
        # FinBERT label mapping: index → label
        self._label_map = {0: SentimentLabel.BULLISH, 1: SentimentLabel.BEARISH, 2: SentimentLabel.NEUTRAL}

    def _load_model(self) -> None:
        """Lazy-load FinBERT model (avoids startup cost if not needed)."""
        if self._model is not None:
            return

        logger.info("Loading FinBERT model...")
        model_name = config.sentiment.finbert_model
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        logger.info(f"FinBERT loaded on {self._device}")

    def analyze_batch(
        self, texts: list[str]
    ) -> list[tuple[SentimentLabel, float]]:
        """
        Analyze a batch of texts with FinBERT.

        Returns list of (label, confidence) tuples.
        Processes in batches for GPU/CPU efficiency.
        """
        self._load_model()
        results: list[tuple[SentimentLabel, float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Truncate to FinBERT's max length
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                )

            for probs in probabilities:
                max_idx = int(torch.argmax(probs))
                confidence = float(probs[max_idx])
                label = self._label_map.get(max_idx, SentimentLabel.NEUTRAL)

                # If confidence is below threshold, default to NEUTRAL
                if confidence < self.confidence_threshold:
                    label = SentimentLabel.NEUTRAL

                results.append((label, confidence))

        return results

    def analyze_single(self, text: str) -> tuple[SentimentLabel, float]:
        """Analyze a single text. Convenience wrapper around batch."""
        results = self.analyze_batch([text])
        return results[0] if results else (SentimentLabel.NEUTRAL, 0.0)

    async def analyze_with_llm_fallback(
        self, text: str
    ) -> tuple[SentimentLabel, float]:
        """
        Try FinBERT first; if confidence is too low, fall back to OpenAI.

        Used for complex or ambiguous text where FinBERT's confidence
        is below the threshold.
        """
        label, confidence = self.analyze_single(text)

        if confidence >= self.confidence_threshold:
            return label, confidence

        # Fallback to OpenAI for better classification
        if not config.api_keys.openai_api_key:
            return label, confidence

        try:
            result = await self._classify_with_openai(text)
            return result
        except Exception as e:
            logger.error(f"OpenAI sentiment fallback failed: {e}")
            return label, confidence

    async def _classify_with_openai(
        self, text: str
    ) -> tuple[SentimentLabel, float]:
        """Use OpenAI API to classify sentiment of financial text."""
        prompt = (
            "Classify the following financial text's market sentiment as exactly one of: "
            "BULLISH, BEARISH, or NEUTRAL. Also provide a confidence score from 0.0 to 1.0.\n\n"
            f"Text: {text[:1000]}\n\n"
            "Respond in exactly this format:\n"
            "SENTIMENT: <label>\n"
            "CONFIDENCE: <score>"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_keys.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        # Parse response
        label = SentimentLabel.NEUTRAL
        confidence = 0.5

        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("SENTIMENT:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw in ("BULLISH", "BEARISH", "NEUTRAL"):
                    label = SentimentLabel(raw)
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        return label, confidence

    def compute_hype_score(
        self,
        mentions: list[MentionData],
        avg_sentiment: float,
        avg_text_length: float,
    ) -> float:
        """
        Compute a hype/pump-and-dump risk score.

        High hype = lots of mentions + very positive sentiment + short low-substance posts.
        This pattern is characteristic of coordinated pump schemes.

        Returns 0.0 (no hype risk) to 1.0 (very likely pump & dump).
        """
        volume = len(mentions)
        volume_threshold = config.sentiment.hype_volume_threshold
        sentiment_threshold = config.sentiment.hype_sentiment_threshold
        substance_threshold = config.sentiment.hype_substance_threshold

        # Volume factor: scales from 0 to 1 as volume increases
        volume_factor = min(volume / volume_threshold, 1.0) if volume_threshold > 0 else 0.0

        # Sentiment factor: overly positive is suspicious
        # avg_sentiment is -1 to 1; map to 0-1 for hype scoring
        sentiment_factor = max(0, (avg_sentiment - 0.3) / 0.7)  # only scores if > 0.3

        # Substance factor: short posts with no analysis = low substance
        # avg_text_length < 100 chars is suspicious for "DD" posts
        substance_factor = max(0, 1.0 - (avg_text_length / 500))

        # Combined hype score — all three factors must be present
        hype_score = (volume_factor * 0.3 + sentiment_factor * 0.4 + substance_factor * 0.3)
        return min(1.0, hype_score)

    def enrich_mentions(
        self, mentions: list[MentionData]
    ) -> list[MentionData]:
        """
        Add sentiment scores to a batch of MentionData objects.

        Processes all texts through FinBERT in a single batch for efficiency.
        """
        if not mentions:
            return mentions

        texts = [m.text_snippet or m.keyword or "" for m in mentions]
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]

        if not valid_indices:
            return mentions

        valid_texts = [texts[i] for i in valid_indices]
        results = self.analyze_batch(valid_texts)

        for idx, (label, confidence) in zip(valid_indices, results):
            mentions[idx].sentiment_label = label
            mentions[idx].sentiment_score = (
                confidence if label == SentimentLabel.BULLISH
                else -confidence if label == SentimentLabel.BEARISH
                else 0.0
            )
            mentions[idx].confidence = confidence

        logger.info(
            f"Enriched {len(valid_indices)} mentions with sentiment",
            extra={"data": {"count": len(valid_indices)}},
        )
        return mentions
