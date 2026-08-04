# Single-Asset Analyzer

Deep-dive analysis tool for a single ticker: technical signals, fundamentals,
options positioning, and macro regime context. Companion to the
mean-reversion scanner — where that tool flags candidates across the
universe, this tool does the full drill-down on one.

## Dependencies

Shared infrastructure comes from
[`shared-data-layer`](https://github.com/GkolfosGeorge/shared-data-layer)
(pinned version, not `@main`):

```
trading-shared-data @ git+https://github.com/GkolfosGeorge/shared-data-layer.git@v0.1.5
```

Provides (flat imports, no code changes needed if the package updates):
```python
from data_engine import get_ticker_data
from signals import compute_signals
from regime_detector import RegimeDetector
from options_archive import load_options_archive
from options_scanner import OptionsScanner   # live scan, rarely needed here
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows
pip install -r requirements.txt
```

Then place `data/sp500_membership.csv` (copied from `shared-data-layer/data/`)
in this repo's `data/` folder — required by `ticker_provider`/`data_engine`
for point-in-time index membership. Changes only a few times a year; manual
copy, not an ongoing sync.

## Fundamentals data

Historical fundamentals (SEC EDGAR, `us-gaap`/`ifrs-full` XBRL tags) live in
a separate Postgres `fundamentals` table — not part of `shared-data-layer`.
Connection/query logic: see `db.py` (to be added, step 2 of the build plan).

**Look-ahead bias note:** always join fundamentals to price/signal timelines
on `filed_date`, never `period_end` — filings publish 4-6 weeks after period
end.

## Options data

`options_archive.load_options_archive()` reads daily Parquet snapshots
written by the separate `daily-market-data` repo's cron job — this repo does
not run its own live scan by default. As of writing, `daily-market-data` has
under a month of history, so IV-rank/percentile-style features aren't
reliable yet.

## Architecture

Part of a shared ecosystem — see meta-README *(link once created)* for
the full picture (portfolio analyzer, mean-reversion, hype script, macro
analyzer, this repo).