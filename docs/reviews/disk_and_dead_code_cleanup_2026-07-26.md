# Disk And Dead-Code Cleanup - 2026-07-26

## Scope

This cleanup removes only artifacts that are reproducible, superseded, rejected,
or unreachable from the current production, collection, research, API, CI, and
container entry points. Raw provider history and artifacts bound to current model
or dataset fingerprints remain protected.

The audit was performed independently for source reachability and disk lineage.
No Market Predictor Python process was active when deletion started. The five-year
Alpaca news and FinBERT manifests both reported `complete`.

## Disk Cleanup

Removed 5,987 files in 67 verified targets:

- rejected `entry_exit.v1` intraday row-level prediction and selected-trade dumps;
- obsolete `intraday_nasdaq_activity_5m_12b` feature matrices;
- V4-H1 dataset `v1`, superseded by fingerprint-bound `v2`;
- incomplete or superseded V3-C8 dataset builds through `v8`;
- rejected V3/V4 row-level OOF and ablation evidence larger than 1 MiB;
- incomplete five-year daily collections `v1` and `v2`, superseded by `v3`;
- parser probes, smoke datasets, test datasets, and the old profile collection `v1`;
- retired predictor-owned live/alert data, temporary files, completed runtime logs,
  stale heavy-job lock, and reproducible Python/Ruff/mypy caches.

Exact reclaimed size: 12,425,097,663 bytes (11.572 GiB).

`data/` decreased from 32.570 GiB to 21.104 GiB.

## Protected Data

The following current or expensive-to-reproduce artifacts were retained:

- `data/raw/alpaca_news_20210709_20260708_v1`;
- `data/features/swing/alpaca_finbert_20210709_20260708_v1`;
- `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3`;
- `data/work/v3_c8_technical_20260711`;
- `data/features/v3_c8_development_20260711_v9`;
- `data/features/v4_h1_120m_development_20260721_v2`;
- `data/artifacts`, `data/canonical`, and `data/universe`;
- `data/external/seeking_alpha_profiles_sp500_20260726_v2`;
- `data/cache/seeking_alpha`, because API quota makes it expensive to rebuild;
- all candidate/rejected model artifacts and their compact audit lineage.

## Dead Code

Removed:

- the unreachable Seeking Alpha MCP placeholder and its unused host template;
- the unreferenced dated volatile-mover script;
- zero-reference CLI, audit, release, feature-list, and version declarations;
- zero-reference Azure Blob, Alpaca hourly, GDELT convenience, Seeking Alpha
  convenience, and SEC company-fact adapter methods.

The test-only canonical symbol master was retained. Although it currently has no
runtime caller, it defines the required rename, delisting, and ticker-reuse identity
invariant. Removing it would erase an architectural contract rather than dead code.

## Verification

Required closure checks:

- complete unit-test suite;
- repository-wide Ruff;
- strict mypy across `src`;
- `compileall` across source, tests, and scripts;
- CLI surface and architecture-boundary tests;
- Git whitespace check;
- protected-path existence and current-manifest integrity checks.
