"""
Central Bank Text Scraper
=========================
Provides curated FOMC meeting minutes summaries and ECB press release
excerpts for FinBERT sentiment analysis.

Note: Rather than scraping live websites (which is fragile and subject to
rate limiting / legal restrictions), this module provides a curated dataset
of historically significant central bank communications. This ensures
reproducibility and reliability of the sentiment pipeline.
"""

import logging
import pandas as pd
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)


# ============================================================
# Curated Central Bank Communications Dataset
# ============================================================
# Each entry represents a key policy communication that moved
# FX markets. These are factual summaries of public documents.

CENTRAL_BANK_TEXTS = [
    # --- FOMC / Federal Reserve ---
    {
        "date": "2014-10-29",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC decided to conclude its asset purchase program, noting "
            "substantial improvement in the outlook for the labor market. The "
            "Committee maintains its target range for the federal funds rate at "
            "0 to 0.25 percent and anticipates it will be appropriate to maintain "
            "this stance for a considerable time."
        ),
    },
    {
        "date": "2015-12-16",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve raised the target range for the federal funds "
            "rate to 0.25 to 0.50 percent, the first rate increase in nearly a "
            "decade. The decision reflects the Committee's confidence that the "
            "economy has continued to strengthen with solid job gains and declining "
            "unemployment."
        ),
    },
    {
        "date": "2016-06-15",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC decided to maintain the target range for the federal funds "
            "rate at 0.25 to 0.50 percent, citing global economic and financial "
            "developments that continue to pose risks. Pace of job gains has "
            "diminished. Committee is closely monitoring inflation indicators "
            "and global economic developments."
        ),
    },
    {
        "date": "2017-12-13",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC raised the target range for the federal funds rate to 1.25 "
            "to 1.50 percent. Labor market has continued to strengthen and economic "
            "activity has been rising at a solid rate. Inflation on a 12-month "
            "basis is expected to remain below 2 percent in the near term."
        ),
    },
    {
        "date": "2018-12-19",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve raised the target range for the federal funds "
            "rate to 2.25 to 2.50 percent. The Committee judges that some further "
            "gradual increases will be consistent with sustained expansion of "
            "economic activity, strong labor market conditions, and inflation near "
            "the 2 percent objective."
        ),
    },
    {
        "date": "2019-07-31",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve lowered the target range for the federal funds "
            "rate to 2.00 to 2.25 percent, the first cut since 2008. The adjustment "
            "reflects global developments citing trade tensions and muted inflation "
            "pressures. The decision was not the beginning of a long series of "
            "rate cuts."
        ),
    },
    {
        "date": "2020-03-15",
        "source": "FOMC",
        "event_type": "emergency_rate_decision",
        "text": (
            "In an emergency action, the Federal Reserve cut the target range for "
            "the federal funds rate to 0 to 0.25 percent and launched a massive "
            "$700 billion quantitative easing program. The coronavirus outbreak has "
            "harmed communities and disrupted economic activity in many countries "
            "including the United States. Financial conditions have tightened "
            "significantly."
        ),
    },
    {
        "date": "2020-06-10",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC maintained the target range at 0 to 0.25 percent. The "
            "ongoing public health crisis will weigh heavily on economic activity, "
            "employment, and inflation in the near term and poses considerable "
            "risks to the economic outlook over the medium term. The Committee "
            "plans to maintain rates near zero until confident the economy has "
            "weathered recent events."
        ),
    },
    {
        "date": "2021-11-03",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC decided to begin reducing the monthly pace of net asset "
            "purchases by $10 billion for Treasury securities and $5 billion for "
            "agency mortgage-backed securities. The Committee judges that similar "
            "reductions will likely be appropriate each month but is prepared to "
            "adjust. Inflation is elevated reflecting supply disruptions."
        ),
    },
    {
        "date": "2022-03-16",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve raised the target range for the federal funds rate "
            "to 0.25 to 0.50 percent, beginning an aggressive tightening cycle. "
            "Inflation remains elevated reflecting supply and demand imbalances "
            "related to the pandemic, higher energy prices, and broader price "
            "pressures. Russia's invasion of Ukraine is causing tremendous human "
            "and economic hardship and creating additional upward pressure on "
            "inflation."
        ),
    },
    {
        "date": "2022-06-15",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC raised the target range by 75 basis points to 1.50 to 1.75 "
            "percent, the largest single increase since 1994. Inflation remains "
            "elevated at 8.6 percent. The Committee is strongly committed to "
            "returning inflation to its 2 percent objective and is highly attentive "
            "to inflation risks."
        ),
    },
    {
        "date": "2022-11-02",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve raised the target range to 3.75 to 4.00 percent, "
            "the fourth consecutive 75 basis point increase. The Committee "
            "anticipates that ongoing increases will be appropriate. Cumulative "
            "tightening and monetary policy lags will be taken into account."
        ),
    },
    {
        "date": "2023-07-26",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The FOMC raised the federal funds rate target range to 5.25 to 5.50 "
            "percent, the highest level in 22 years. Economic activity has been "
            "expanding at a moderate pace. Job gains have been robust and the "
            "unemployment rate has remained low. Inflation remains elevated."
        ),
    },
    {
        "date": "2024-09-18",
        "source": "FOMC",
        "event_type": "rate_decision",
        "text": (
            "The Federal Reserve lowered the target range by 50 basis points to "
            "4.75 to 5.00 percent, the first rate cut since 2020. The Committee "
            "gained greater confidence that inflation is moving sustainably toward "
            "2 percent and judged risks to employment and inflation goals are "
            "roughly in balance."
        ),
    },
    # --- ECB Press Releases ---
    {
        "date": "2014-06-05",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB cut the main refinancing rate to 0.15 percent and introduced "
            "negative deposit rates at -0.10 percent for the first time. President "
            "Draghi announced targeted longer-term refinancing operations to "
            "enhance lending to the euro area economy. Prepared to act swiftly "
            "with further unconventional instruments."
        ),
    },
    {
        "date": "2015-01-22",
        "source": "ECB",
        "event_type": "policy_announcement",
        "text": (
            "The ECB announced an expanded asset purchase programme of €60 "
            "billion per month in combined purchases of public and private sector "
            "securities to address risks of a too prolonged period of low inflation. "
            "Purchases intended to be carried out until end of September 2016."
        ),
    },
    {
        "date": "2016-03-10",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB cut all three key interest rates and expanded monthly asset "
            "purchases to €80 billion. The deposit facility rate was lowered to "
            "-0.40 percent. The corporate sector purchase programme was launched. "
            "President Draghi stated these measures aim to further ease financing "
            "conditions and stimulate credit provision."
        ),
    },
    {
        "date": "2019-09-12",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB cut the deposit facility rate by 10 basis points to -0.50 "
            "percent and restarted net asset purchases at €20 billion per month. "
            "A tiered system for reserve remuneration was introduced to mitigate "
            "the adverse effects of negative rates on bank profitability."
        ),
    },
    {
        "date": "2020-03-18",
        "source": "ECB",
        "event_type": "emergency_action",
        "text": (
            "The ECB launched the €750 billion Pandemic Emergency Purchase "
            "Programme (PEPP) to counter serious risks to monetary policy "
            "transmission and the outlook for the euro area. President Lagarde "
            "stated there are no limits to commitment to the euro and the ECB "
            "will do everything necessary within its mandate."
        ),
    },
    {
        "date": "2022-07-21",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB raised all three key interest rates by 50 basis points, "
            "ending eight years of negative rates. The deposit facility rate "
            "was raised to 0 percent. The Transmission Protection Instrument "
            "was approved to ensure monetary policy transmission across all "
            "euro area countries."
        ),
    },
    {
        "date": "2023-09-14",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB raised the deposit facility rate to 4.00 percent, a record "
            "high. The Governing Council considers that rates have reached levels "
            "that, maintained for a sufficiently long duration, will make a "
            "substantial contribution to the timely return of inflation to target. "
            "Inflation continues to decline but is still expected to remain too "
            "high for too long."
        ),
    },
    {
        "date": "2024-06-06",
        "source": "ECB",
        "event_type": "rate_decision",
        "text": (
            "The ECB cut the deposit facility rate by 25 basis points to 3.75 "
            "percent, the first rate decrease in five years. Inflation has fallen "
            "by more than 2.5 percentage points since September 2023 and the "
            "inflation outlook has improved markedly. Monetary policy will continue "
            "to be data-dependent and made on a meeting-by-meeting basis."
        ),
    },
    # --- Geopolitical / Market-Moving Events ---
    {
        "date": "2016-06-24",
        "source": "Market",
        "event_type": "geopolitical",
        "text": (
            "Brexit referendum results shocked global markets as the United "
            "Kingdom voted to leave the European Union. The British pound sterling "
            "plunged to its lowest level in 31 years against the US dollar, falling "
            "over 10 percent. Global equity markets saw sharp selloffs and safe "
            "haven currencies like the Japanese yen and Swiss franc surged."
        ),
    },
    {
        "date": "2020-03-09",
        "source": "Market",
        "event_type": "geopolitical",
        "text": (
            "Oil prices crashed over 30 percent following the collapse of OPEC+ "
            "talks, marking the worst single-day decline since 1991. Combined "
            "with the escalating coronavirus pandemic, US stock futures triggered "
            "circuit breakers. The US dollar index fell as markets priced in "
            "emergency Federal Reserve rate cuts."
        ),
    },
    {
        "date": "2022-02-24",
        "source": "Market",
        "event_type": "geopolitical",
        "text": (
            "Russia launched a full-scale invasion of Ukraine, triggering massive "
            "market volatility. Energy prices surged with Brent crude approaching "
            "$130 per barrel. European currencies weakened significantly against "
            "the US dollar. The euro fell below parity with the dollar for the "
            "first time in 20 years in subsequent months."
        ),
    },
    {
        "date": "2023-03-10",
        "source": "Market",
        "event_type": "financial_crisis",
        "text": (
            "Silicon Valley Bank collapsed, the largest US bank failure since 2008, "
            "triggering a regional banking crisis. Markets repriced Federal Reserve "
            "rate expectations dramatically. The dollar weakened as traders "
            "anticipated the Fed would pause tightening to prevent further "
            "financial instability. Credit Suisse subsequently faced liquidity "
            "crisis and was acquired by UBS."
        ),
    },
]


class TextScraper:
    """
    Provides curated central bank communications for sentiment analysis.

    This module uses a pre-compiled dataset of historically significant
    monetary policy communications rather than live web scraping to
    ensure reproducibility and reliability.
    """

    def __init__(self):
        self.data = pd.DataFrame(CENTRAL_BANK_TEXTS)
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data = self.data.sort_values("date").reset_index(drop=True)
        logger.info(
            f"Loaded {len(self.data)} central bank text entries "
            f"({self.data['date'].min().date()} to {self.data['date'].max().date()})"
        )

    def get_all(self) -> pd.DataFrame:
        """Return all central bank texts."""
        return self.data.copy()

    def get_by_source(self, source: str) -> pd.DataFrame:
        """Filter by source ('FOMC', 'ECB', 'Market')."""
        return self.data[self.data["source"] == source].copy()

    def get_by_date_range(self, start: str, end: str) -> pd.DataFrame:
        """Filter by date range."""
        mask = (self.data["date"] >= start) & (self.data["date"] <= end)
        return self.data[mask].copy()

    def get_texts_for_sentiment(self) -> list:
        """
        Return list of (date, text) tuples for sentiment analysis.
        Ordered chronologically.
        """
        return list(zip(self.data["date"], self.data["text"]))

    def get_nearest_text(self, target_date: str,
                         window_days: int = 30) -> pd.DataFrame:
        """
        Get the most recent communication within a window before target_date.

        Args:
            target_date: Target date string
            window_days: Number of days to look back

        Returns:
            DataFrame of matching texts, or empty DataFrame if none found.
        """
        target = pd.to_datetime(target_date)
        window_start = target - pd.Timedelta(days=window_days)
        mask = (self.data["date"] >= window_start) & (self.data["date"] <= target)
        result = self.data[mask].copy()
        return result

    def save_to_cache(self, filename: str = "central_bank_texts.csv"):
        """Save texts to local cache."""
        path = os.path.join(config.DATA_DIR, filename)
        self.data.to_csv(path, index=False)
        logger.info(f"Texts cached to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = TextScraper()

    print(f"\n{'='*60}")
    print("Central Bank Communications Dataset")
    print(f"{'='*60}")
    print(f"Total entries: {len(scraper.data)}")
    print(f"\nBy source:")
    print(scraper.data["source"].value_counts().to_string())
    print(f"\nBy event type:")
    print(scraper.data["event_type"].value_counts().to_string())
    print(f"\nSample entries:")
    print(scraper.data[["date", "source", "event_type"]].head(10).to_string())
