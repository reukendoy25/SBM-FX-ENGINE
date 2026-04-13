"""
Anomaly Detector Evaluation Suite
===================================
Unsupervised and supervised (when labels are available) metrics
for comparing anomaly detectors.

Unsupervised proxy metrics (no ground-truth required):
  1. Silhouette score      — how well normalized scores separate -1 / +1 groups
  2. Contamination-precision proxy — mean score of top-k vs rest (k = contamination * N)
  3. Score AUC proxy       — area under the sorted anomaly score curve (Gini-like)
  4. Score separation      — mean(anomaly scores) - mean(normal scores)

Supervised metrics (ground-truth labels required):
  - Precision, Recall, F1 (binary: anomaly = positive)
  - These are only used in tests with injected synthetic anomalies.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Computes and compares evaluation metrics for anomaly detectors.
    """

    @staticmethod
    def evaluate(results: pd.DataFrame,
                 contamination: float = 0.01) -> Dict[str, float]:
        """
        Compute unsupervised proxy metrics from a detector's output DataFrame.

        Args:
            results: DataFrame with columns anomaly_label (-1/1)
                     and anomaly_score_normalized (0-1).
            contamination: Expected anomaly fraction (used for precision proxy).

        Returns:
            Dictionary of metric names → values.
        """
        scores = results["anomaly_score_normalized"].values
        labels = results["anomaly_label"].values

        n = len(scores)
        n_anomalies = (labels == -1).sum()

        # ── 1. Silhouette-like score separation ──────────────────────────────
        # Measures mean score gap between anomalies and normals.
        if n_anomalies > 0 and n_anomalies < n:
            mean_anomaly_score = scores[labels == -1].mean()
            mean_normal_score = scores[labels == 1].mean()
            score_separation = float(mean_anomaly_score - mean_normal_score)
        else:
            score_separation = 0.0

        # ── 2. Silhouette score (from sklearn) ────────────────────────────────
        try:
            from sklearn.metrics import silhouette_score
            if len(np.unique(labels)) == 2:
                sil_score = float(silhouette_score(
                    scores.reshape(-1, 1), labels
                ))
            else:
                sil_score = float("nan")
        except Exception:
            sil_score = float("nan")

        # ── 3. Contamination-precision proxy ──────────────────────────────────
        # Top-k by score — what fraction are flagged as -1?
        k = max(1, int(contamination * n))
        top_k_idx = np.argsort(scores)[-k:]
        top_k_labels = labels[top_k_idx]
        contamination_precision = float((top_k_labels == -1).mean())

        # ── 4. Score AUC proxy (Gini coefficient of scores) ───────────────────
        # Gini = area above Lorenz curve = 2*AUC - 1 for sorted distribution.
        sorted_scores = np.sort(scores)
        n_s = len(sorted_scores)
        cum = np.cumsum(sorted_scores)
        gini = float(
            (2 * np.sum((np.arange(1, n_s + 1) * sorted_scores))) /
            (n_s * cum[-1]) - (n_s + 1) / n_s
        ) if cum[-1] > 0 else 0.0

        metrics = {
            "n_samples": int(n),
            "n_anomalies": int(n_anomalies),
            "anomaly_rate_pct": float(round(n_anomalies / n * 100, 3)),
            "score_separation": round(score_separation, 4),
            "silhouette_score": round(sil_score, 4) if not np.isnan(sil_score) else None,
            "contamination_precision": round(contamination_precision, 4),
            "score_gini": round(gini, 4),
        }

        return metrics

    @staticmethod
    def evaluate_with_labels(results: pd.DataFrame,
                              true_labels: np.ndarray,
                              contamination: float = 0.01) -> Dict[str, float]:
        """
        Compute supervised metrics when ground-truth labels are available.
        Use this in unit tests with injected synthetic anomalies.

        Args:
            results: Detector output DataFrame.
            true_labels: Array of true labels — 1 = anomaly, 0 = normal
                         (integer array aligned with results index).
            contamination: Passed through to evaluate() for proxy metrics.

        Returns:
            Combined dict of proxy + supervised metrics.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        proxy = AnomalyEvaluator.evaluate(results, contamination)

        pred_binary = (results["anomaly_label"].values == -1).astype(int)
        true_binary = np.asarray(true_labels).astype(int)

        if len(np.unique(true_binary)) < 2:
            # No positive class present
            supervised = {"precision": None, "recall": None, "f1_score": None}
        else:
            supervised = {
                "precision": round(float(
                    precision_score(true_binary, pred_binary, zero_division=0)
                ), 4),
                "recall": round(float(
                    recall_score(true_binary, pred_binary, zero_division=0)
                ), 4),
                "f1_score": round(float(
                    f1_score(true_binary, pred_binary, zero_division=0)
                ), 4),
            }

        return {**proxy, **supervised}

    @staticmethod
    def compare(results_dict: Dict[str, pd.DataFrame],
                contamination: float = 0.01) -> pd.DataFrame:
        """
        Compare multiple detectors and print a formatted table.

        Args:
            results_dict: Mapping of detector name → predictions DataFrame.
            contamination: Used for proxy metric calculation.

        Returns:
            DataFrame with metrics as columns and detectors as rows.
        """
        rows = {}
        for name, results in results_dict.items():
            rows[name] = AnomalyEvaluator.evaluate(results, contamination)

        comparison = pd.DataFrame(rows).T

        # Print formatted table
        logger.info("\n" + "=" * 72)
        logger.info("  ANOMALY DETECTOR COMPARISON")
        logger.info("=" * 72)
        logger.info(
            f"  {'Detector':<22} {'Anomalies':>9} {'Rate%':>7} "
            f"{'Sep':>7} {'Silhouette':>11} {'ContPrec':>9} {'Gini':>7}"
        )
        logger.info("-" * 72)
        for name, row in comparison.iterrows():
            sil = f"{row['silhouette_score']:.4f}" if row['silhouette_score'] is not None else "  N/A  "
            logger.info(
                f"  {name:<22} {int(row['n_anomalies']):>9} "
                f"{row['anomaly_rate_pct']:>7.2f} "
                f"{row['score_separation']:>7.4f} "
                f"{sil:>11} "
                f"{row['contamination_precision']:>9.4f} "
                f"{row['score_gini']:>7.4f}"
            )
        logger.info("=" * 72)

        return comparison
