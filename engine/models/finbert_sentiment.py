"""
FinBERT Sentiment Analyzer
===========================
Performs financial sentiment analysis on central bank communications
using ProsusAI/FinBERT — an open-source BERT model fine-tuned on
financial text data.

FinBERT was trained on 10,000+ financial articles, analyst reports,
and earnings calls. It classifies text into three categories:
positive, negative, neutral — and provides confidence probabilities.

Reference: https://huggingface.co/ProsusAI/finbert
"""

import logging
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Financial sentiment analyzer using ProsusAI/FinBERT.

    Runs locally — no API keys required. Model weights are
    automatically downloaded from Hugging Face on first use
    and cached locally.
    """

    def __init__(self, model_name: str = None):
        """
        Initialize the FinBERT sentiment analyzer.

        Args:
            model_name: Hugging Face model identifier.
                        Defaults to 'ProsusAI/finbert'.
        """
        self.model_name = model_name or config.FINBERT_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self._loaded = False
        logger.info(
            f"FinBERTSentimentAnalyzer initialized "
            f"(model: {self.model_name})"
        )

    def _ensure_loaded(self):
        """Lazy-load the model and tokenizer on first use."""
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            logger.info(f"Loading FinBERT model: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.eval()  # Set to evaluation mode
            self._loaded = True
            logger.info("FinBERT model loaded successfully.")
        except ImportError as e:
            logger.error(
                f"Required packages not installed: {e}. "
                f"Install with: pip install transformers torch"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            raise

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text (FOMC minutes, ECB press release, etc.)

        Returns:
            Dict with keys:
              - sentiment: float in [-1, 1] (negative to positive)
              - label: "positive", "negative", or "neutral"
              - confidence: float in [0, 1]
              - probabilities: dict of label → probability
        """
        self._ensure_loaded()
        import torch

        # Tokenize (truncate to FinBERT's max length)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_SEQUENCE_LENGTH,
            padding=True,
        )

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(
                outputs.logits, dim=-1
            )[0]

        # FinBERT labels: positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        probs = {label: prob.item() for label, prob in zip(labels, probabilities)}

        # Determine primary label
        primary_label = max(probs, key=probs.get)
        confidence = probs[primary_label]

        # Compute continuous sentiment score in [-1, 1]
        sentiment_score = (
            probs["positive"] * config.SENTIMENT_MAP["positive"]
            + probs["negative"] * config.SENTIMENT_MAP["negative"]
            + probs["neutral"] * config.SENTIMENT_MAP["neutral"]
        )

        return {
            "sentiment": round(sentiment_score, 4),
            "label": primary_label,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
        }

    def analyze_batch(self, texts: list) -> list:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of input text strings.

        Returns:
            List of sentiment result dicts.
        """
        self._ensure_loaded()
        import torch

        results = []
        batch_size = config.SENTIMENT_BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_SEQUENCE_LENGTH,
                padding=True,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                )

            labels = ["positive", "negative", "neutral"]

            for probs in probabilities:
                prob_dict = {
                    label: prob.item()
                    for label, prob in zip(labels, probs)
                }
                primary_label = max(prob_dict, key=prob_dict.get)
                confidence = prob_dict[primary_label]
                sentiment_score = (
                    prob_dict["positive"] * config.SENTIMENT_MAP["positive"]
                    + prob_dict["negative"] * config.SENTIMENT_MAP["negative"]
                    + prob_dict["neutral"] * config.SENTIMENT_MAP["neutral"]
                )

                results.append({
                    "sentiment": round(sentiment_score, 4),
                    "label": primary_label,
                    "confidence": round(confidence, 4),
                    "probabilities": {
                        k: round(v, 4) for k, v in prob_dict.items()
                    },
                })

        logger.info(f"Analyzed {len(results)} texts.")
        return results

    def analyze_dataframe(self, df: pd.DataFrame,
                           text_column: str = "text",
                           date_column: str = "date") -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame of texts.

        Args:
            df: DataFrame containing text data
            text_column: Name of the column with text
            date_column: Name of the date column

        Returns:
            DataFrame with added sentiment columns, indexed by date.
        """
        texts = df[text_column].tolist()
        results = self.analyze_batch(texts)

        sentiment_df = pd.DataFrame(results)
        sentiment_df[date_column] = df[date_column].values
        sentiment_df = sentiment_df.set_index(date_column)

        # Add source information if available
        if "source" in df.columns:
            sentiment_df["source"] = df["source"].values
        if "event_type" in df.columns:
            sentiment_df["event_type"] = df["event_type"].values

        return sentiment_df

    def get_sentiment_time_series(self, df: pd.DataFrame,
                                   text_column: str = "text",
                                   date_column: str = "date") -> pd.Series:
        """
        Get a time series of sentiment scores.

        Returns:
            Series with DatetimeIndex and sentiment scores.
        """
        result = self.analyze_dataframe(df, text_column, date_column)
        return result["sentiment"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = FinBERTSentimentAnalyzer()

    # Test with sample central bank texts
    test_texts = [
        (
            "The Federal Reserve raised rates by 75 basis points, the largest "
            "increase since 1994, as inflation remains elevated at 8.6 percent."
        ),
        (
            "The Committee decided to maintain rates near zero. Economic outlook "
            "remains uncertain with significant downside risks to growth."
        ),
        (
            "Inflation has fallen substantially. The ECB cut rates by 25 basis "
            "points, signaling confidence in the disinflation process."
        ),
    ]

    print(f"\n{'='*60}")
    print("FinBERT Sentiment Analysis Demo")
    print(f"{'='*60}")

    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze(text)
        print(f"\nText {i}: {text[:80]}...")
        print(f"  Sentiment: {result['sentiment']:.4f}")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Probabilities: {result['probabilities']}")
