# Cleaned Access Point Metrics (5‑minute)

## Overview

This dataset contains gzipped CSV exports of access point (AP) metrics at 5‑minute resolution. Each file focuses on a specific metric, keyed by `ap_name` and `timestamp`.

- Format: CSV compressed with gzip (`.csv.gz`)
- Timezone: UTC; timestamps include timezone info
- Keys: `ap_name` (AP identifier), `timestamp`
- Frequency: 5 minutes

## Files

- `client_metrics_uap_5min.csv.gz`: client counts per AP and time
  - Columns: `ap_name`, `timestamp`, `client_count`
- `cpu_metrics_uap_5min.csv.gz`: CPU usage ratio per AP and time
  - Columns: `ap_name`, `timestamp`, `cpu_usage_ratio` (0–1)
- `memory_metrics_uap_5min.csv.gz`: memory usage ratio per AP and time
  - Columns: `ap_name`, `timestamp`, `memory_usage_ratio` (0–1)
- `signal_24g_metrics_uap_5min.csv.gz`: 2.4 GHz signal strength (dBm)
  - Columns: `ap_name`, `timestamp`, `signal_dbm` (dBm)
- `signal_5g_metrics_uap_5min.csv.gz`: 5 GHz signal strength (dBm)
  - Columns: `ap_name`, `timestamp`, `signal_dbm` (dBm)

## Schema

- `ap_name`: string; AP identifier (may be a device name or MAC address).
- `timestamp`: UTC timestamp. Observed formats include `YYYY-MM-DD HH:MM:SS+00:00` and ISO 8601 `YYYY-MM-DDTHH:MM:SSZ`.
- `client_count`: integer ≥ 0; number of associated clients.
- `cpu_usage_ratio`: float in [0, 1]; CPU utilization as a fraction.
- `memory_usage_ratio`: float in [0, 1]; memory utilization as a fraction.
- `signal_dbm`: integer (typically negative); received signal strength in dBm.

## Quick Peek (shell)

- First rows of each file: `for f in *.csv.gz; do echo "--- $f ---"; gzip -dc "$f" | head; done`
- Row count (approximate): `for f in *.csv.gz; do echo "--- $f ---"; gzip -dc "$f" | wc -l; done`

## Usage (Python/pandas)

```python
import pandas as pd
from functools import reduce

client = pd.read_csv('client_metrics_uap_5min.csv.gz', compression='infer', parse_dates=['timestamp'])
cpu    = pd.read_csv('cpu_metrics_uap_5min.csv.gz',    compression='infer', parse_dates=['timestamp'])
mem    = pd.read_csv('memory_metrics_uap_5min.csv.gz', compression='infer', parse_dates=['timestamp'])
sig24  = pd.read_csv('signal_24g_metrics_uap_5min.csv.gz', compression='infer', parse_dates=['timestamp'])
sig5   = pd.read_csv('signal_5g_metrics_uap_5min.csv.gz',  compression='infer', parse_dates=['timestamp'])

# Normalize timestamps to consistent UTC dtype
for df in [client, cpu, mem, sig24, sig5]:
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

# Example: merge all metrics into a single frame
frames = [
    client.rename(columns={'client_count': 'clients'}),
    cpu.rename(columns={'cpu_usage_ratio': 'cpu'}),
    mem.rename(columns={'memory_usage_ratio': 'mem'}),
    sig24.rename(columns={'signal_dbm': 'sig_24g_dbm'}),
    sig5.rename(columns={'signal_dbm': 'sig_5g_dbm'}),
]
merged = reduce(lambda l, r: pd.merge(l, r, on=['ap_name', 'timestamp'], how='outer'), frames)
merged = merged.sort_values(['ap_name', 'timestamp'])
```

## Notes

- Mixed timestamp styles are present; ensure parsers handle both and keep timezone (UTC).
- `ap_name` format varies (human‑readable name vs MAC address); standardize if needed for joins.
- Ratios are unitless fractions in [0, 1]; signal strengths are dBm.
- Some intervals or metrics may be missing for certain APs; plan for outer joins when combining.

## License / Attribution

If not stated elsewhere, treat this dataset as internal/proprietary. Replace this section with the appropriate license and attribution if sharing externally.

