from __future__ import annotations

import pandas as pd

SENTIMENT_INPUT_MODES = ("title", "title_summary", "text")


def build_sentiment_inputs(events: pd.DataFrame, *, mode: str = "title_summary") -> pd.Series:
    """Build bounded-purpose FinBERT inputs without modifying raw provider text."""
    if mode not in SENTIMENT_INPUT_MODES:
        choices = ", ".join(SENTIMENT_INPUT_MODES)
        raise ValueError(f"Unsupported sentiment input mode {mode!r}; expected one of: {choices}.")

    def clean(column: str) -> pd.Series:
        if column not in events.columns:
            return pd.Series("", index=events.index, dtype="object")
        return events[column].fillna("").astype(str).str.strip()

    title = clean("title")
    summary = clean("summary")
    text = clean("text")
    if mode == "title":
        return title.mask(title.eq(""), summary).mask(lambda values: values.eq(""), text)
    if mode == "text":
        return text.mask(text.eq(""), summary).mask(lambda values: values.eq(""), title)

    distinct_summary = summary.mask(summary.eq(title), "")
    combined = (title + ". " + distinct_summary).str.strip(". ")
    return combined.mask(combined.eq(""), text)


class FinbertScorer:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        *,
        torch_num_threads: int = 0,
        max_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        self.model_name = model_name
        self.max_length = max(1, int(max_length))
        if torch_num_threads > 0:
            torch.set_num_threads(torch_num_threads)
        device = 0 if torch.cuda.is_available() else -1
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except OSError as exc:
            raise RuntimeError(
                f"FinBERT model {model_name!r} is not available in the local cache; "
                "run `market-predictor download-model` before offline inference."
            ) from exc
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=True,
            max_length=self.max_length,
        )

    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(columns=["sentiment_label", "sentiment_score", "sentiment_numeric"])
        results = self.classifier(
            texts,
            batch_size=batch_size,
            truncation=True,
            max_length=self.max_length,
        )
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
