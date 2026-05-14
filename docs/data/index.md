# Data Guide

Historical market data belongs in `data/raw/`. The directory is gitignored because files are large and may come from Bloomberg, Futu, Webull, or manual exports.

## Required Fields

The common adapter interface needs:

| Field | Meaning |
|---|---|
| `timestamp` | Bar timestamp, preferably Hong Kong local time or timezone-aware |
| `open` | Bar open price |
| `high` | Bar high price |
| `low` | Bar low price |
| `close` | Bar close or last price |
| `volume` | Optional for current baseline, useful for future features |

## Accepted Formats

### Normal CSV

```csv
timestamp,open,high,low,close,volume
2018-04-03 10:00,24.07,24.22,24.02,24.15,1815500
```

### Futu CSV

The Futu adapter accepts exports with:

```csv
code,name,time_key,open,close,high,low,volume,turnover
```

### Bloomberg XLSX

The Bloomberg adapter expects sheet `Data` with logical columns:

| Column | Meaning |
|---|---|
| `Date` | Bar timestamp |
| `Open` | Bar open |
| `High` | Bar high |
| `Low` | Bar low |
| `Last_Price` | Bar close / last price |
| `Volume` | Bar volume |

Bloomberg field names vary. Keep these names where possible to avoid cleanup.

## Interval Guidance

| Interval | Best Use |
|---|---|
| 1-minute | Best raw source; can be aggregated later into 5m, 15m, or 30m |
| 5-minute | Useful for future high-frequency Q-table experiments |
| 30-minute | Current production-style research path |
| 1-hour | Lower noise, fewer decisions, easier diagnostics |
| EOD | Coarse benchmark; not enough for day-trading behavior |

For current live-style experiments, 30-minute data should include the final `16:00` close so the bot can evaluate end-of-day flattening correctly.

## Bloomberg Export Checklist

Pull one file per asset and interval where possible:

- `2800.HK` from 2018 onward, minimum.
- `2828.HK`, `7299.HK`, `7568.HK` if available and useful for lab research.
- HK local timestamps.
- Full morning and afternoon sessions.
- Bars around every configured decision time.
- Volume included.

Good filenames:

```text
data/raw/2800_hk_1m.csv
data/raw/2800_hk_30m.xlsx
data/raw/7299_hk_30m.xlsx
```

## Common Pitfalls

- HK lunch break is expected; missing `12:00-13:00` bars are not corruption.
- A file ending at `15:30` is not the same as having a `16:00` close.
- If timestamps are bar-open instead of bar-close, configure decision times accordingly.
- Do not commit raw market data to Git.
