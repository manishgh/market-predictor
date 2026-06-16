from __future__ import annotations

import pandas as pd


class FinbertScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", *, torch_num_threads: int = 0) -> None:
        import torch
        from transformers import pipeline

        self.model_name = model_name
        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )

    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["sentiment_label", "sentiment_score", "sentiment_numeric"])
        results = self.classifier(texts, batch_size=batch_size)
        frame = pd.DataFrame(results).rename(columns={"label": "sentiment_label", "score": "sentiment_score"})
        frame["sentiment_numeric"] = frame.apply(self._numeric_score, axis=1)
        return frame

    @staticmethod
    def _numeric_score(row: pd.Series) -> float:
        label = str(row["sentiment_label"]).lower()
        score = float(row["sentiment_score"])
        if "positive" in label:
            return score
        if "negative" in label:
            return -score
        return 0.0


def download_model(model_name: str) -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSequenceClassification.from_pretrained(model_name)
