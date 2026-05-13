# Data Guide

Historical market data belongs in `data/raw/`. This folder is gitignored because files are large and may come from Bloomberg/Futu.

## Accepted Formats

### Normal CSV

Use this for hand-cleaned bars:

```csv
timestamp,open,high,low,close,volume
2018-04-03 09:30,24.07,24.22,24.02,24.07,1815500
```

### Futu CSV

The loader accepts Futu exports with:

```csv
code,name,time_key,open,close,high,low,volume,turnover,...
```

### Bloomberg XLSX

The Bloomberg adapter expects sheet `Data` with these logical columns:

| Column | Meaning |
|---|---|
| `Date` | Bar timestamp, preferably Hong Kong local time. |
| `Open` | Bar open. |
| `High` | Bar high. |
| `Low` | Bar low. |
| `Last_Price` | Bar close / last price. |
| `Volume` | Bar volume. |

Bloomberg field names vary, but keep these names if possible to avoid manual cleanup.

## What To Pull From Bloomberg

Minimum useful set:

- `2800.HK` with 1-minute or 30-minute bars from 2018 onward.
- If available/tradeable: `2828.HK`, `7299.HK`, `7568.HK`, and any leveraged ETF approved by Quantphemes.

Preferred:

- 1-minute bars, 3-5+ years.
- HK local timestamps.
- Full day coverage including morning session, afternoon session, and final tradable bar.
- Include volume.

## Where To Put Files

Examples:

```text
data/raw/2800_hk_2018start_30min.csv
data/raw/2800_hk_1m.csv
data/raw/HK_07226_1m.csv
data/raw/7299_hk_1m.xlsx
```

Then point an experiment YAML at the file:

```yaml
data:
  source: csv
  path: data/raw/2800_hk_2018start_30min.csv
  symbol: 2800.HK
```

For 30-minute files ending at `15:30`, use `15:30` as `force_close_time` and do not include it as a decision time.
