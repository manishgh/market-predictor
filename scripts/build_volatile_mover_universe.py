from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SEED_MOVERS = [
    "RGTI",
    "POET",
    "PATH",
    "LUNR",
    "RKLB",
    "IONQ",
    "QBTS",
    "QUBT",
    "RDW",
    "ACHR",
    "JOBY",
    "ASTS",
    "OUST",
    "SOUN",
    "BBAI",
    "APLD",
    "CRDO",
    "ALAB",
    "SMCI",
    "MRVL",
    "NVTS",
    "LASE",
    "PLTR",
    "RXRX",
    "CRSP",
    "EDIT",
    "NTLA",
    "BEAM",
    "VKTX",
    "ALT",
    "IBRX",
    "TERN",
    "BDTX",
    "DNA",
    "SANA",
    "ARWR",
    "RNA",
]


def main() -> None:
    raw_dir = Path("data/external/finviz/volatile_probe")
    out_dir = Path("data/universe")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in raw_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        if frame.empty or "Ticker" not in frame.columns:
            continue
        screen = path.stem
        match = re.search(r"sec_([^_]+)_cap_([^_]+)", screen)
        frame["finviz_screen"] = screen
        frame["finviz_sector_filter"] = match.group(1) if match else ""
        frame["finviz_cap_filter"] = match.group(2) if match else ""
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No usable Finviz CSVs found under {raw_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined["Ticker"] = combined["Ticker"].astype(str).str.upper().str.strip()
    combined = combined.drop_duplicates("Ticker")
    combined["change_pct"] = pd.to_numeric(combined["Change"].astype(str).str.replace("%", "", regex=False), errors="coerce")
    combined["abs_change_pct"] = combined["change_pct"].abs()
    combined["market_cap_m"] = pd.to_numeric(combined["Market Cap"], errors="coerce")
    combined["price_num"] = pd.to_numeric(combined["Price"], errors="coerce")
    combined["volume_num"] = pd.to_numeric(combined["Volume"], errors="coerce")
    combined["theme_bucket"] = combined.apply(_theme_bucket, axis=1)
    combined["source"] = "finviz_live_relvol_screen"
    combined["candidate_score"] = (
        combined["abs_change_pct"].fillna(0) * 0.55
        + combined["volume_num"].fillna(0).rank(pct=True) * 25
        + combined["price_num"].between(2, 80).astype(int) * 10
        + combined["market_cap_m"].between(100, 10_000).astype(int) * 10
    ).round(3)

    cols = [
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "Country",
        "theme_bucket",
        "finviz_cap_filter",
        "Market Cap",
        "Price",
        "Change",
        "Volume",
        "abs_change_pct",
        "candidate_score",
        "finviz_screen",
        "source",
    ]
    finviz = combined[[col for col in cols if col in combined.columns]].sort_values(
        ["candidate_score", "abs_change_pct"], ascending=False
    )
    finviz_path = out_dir / "finviz_volatile_movers_20260704.csv"
    finviz.to_csv(finviz_path, index=False)

    existing = set(finviz["Ticker"])
    seed_rows = []
    for ticker in SEED_MOVERS:
        if ticker in existing:
            continue
        seed_rows.append(
            {
                "Ticker": ticker,
                "Company": "",
                "Sector": "",
                "Industry": "",
                "Country": "",
                "theme_bucket": "seed_high_beta_mover",
                "finviz_cap_filter": "",
                "Market Cap": "",
                "Price": "",
                "Change": "",
                "Volume": "",
                "abs_change_pct": "",
                "candidate_score": 0,
                "finviz_screen": "manual_seed_from_strategy",
                "source": "seed_watchlist",
            }
        )
    research = pd.concat([finviz, pd.DataFrame(seed_rows)], ignore_index=True)
    research_path = out_dir / "volatile_mover_research_universe_20260704.csv"
    research.to_csv(research_path, index=False)
    (out_dir / "volatile_mover_research_tickers_20260704_commas.txt").write_text(
        ",".join(research["Ticker"].astype(str)),
        encoding="utf-8",
    )
    research["Ticker"].to_csv(out_dir / "volatile_mover_research_tickers_20260704.txt", index=False, header=False)
    summary = (
        research.groupby(["source", "theme_bucket"])
        .size()
        .reset_index(name="count")
        .sort_values(["source", "count"], ascending=[True, False])
    )
    summary_path = out_dir / "volatile_mover_research_universe_20260704_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"finviz_rows={len(finviz)} research_rows={len(research)}")
    print(summary.to_string(index=False))
    print("top_candidates")
    display_cols = ["Ticker", "Company", "Sector", "Industry", "theme_bucket", "Price", "Change", "Volume", "candidate_score"]
    print(finviz[[col for col in display_cols if col in finviz.columns]].head(40).to_string(index=False))
    print(f"wrote={research_path}")
    print(f"finviz_only={finviz_path}")


def _theme_bucket(row: pd.Series) -> str:
    text = f"{row.get('Sector', '')} {row.get('Industry', '')} {row.get('Company', '')}".lower()
    if any(item in text for item in ["biotechnology", "biosciences", "therapeutics", "pharmaceutical", "drug manufacturers"]):
        return "healthcare_biotech_catalyst"
    if any(item in text for item in ["medical devices", "medical instruments", "diagnostics", "health information"]):
        return "healthcare_tools_devices"
    if any(item in text for item in ["semiconductor", "semiconductors", "electronic components"]):
        return "ai_semis_photonics_hardware"
    if any(item in text for item in ["software", "information technology", "data", "cloud", "infrastructure", "analytics"]):
        return "ai_data_software_infra"
    if any(item in text for item in ["aerospace", "defense", "airlines", "rental", "business services"]):
        return "industrial_space_mobility"
    return "other_volatile"


if __name__ == "__main__":
    main()
