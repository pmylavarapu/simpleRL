# PWRL EDGAR Monitoring Agent

Monitors SEC EDGAR filings for **Powerlaw Corp** (CIK: 2052053, Ticker: PWRL) and assesses whether a Nasdaq listing appears imminent.

---

## What it does

- Polls the [SEC EDGAR API](https://data.sec.gov/submissions/CIK0002052053.json) every **60 minutes**
- Flags new **N-2, N-2/A, 8-K, and EFFECT** filings
- For N-2/N-2A filings, fetches the full document and scans for listing-imminence signals:
  - Remaining bracketed placeholders like `[DATE]` or `[FINANCIAL ADVISOR]`
  - Stifel named explicitly without brackets
  - A concrete calendar listing date
  - EFFECT filing type (SEC declared registration effective)
- Delivers a plain-English assessment:
  - `Routine amendment`
  - `Looks close — placeholders mostly cleared`
  - `IMMINENT — registration effective or date confirmed`
- Checks **Google News RSS** for any recent PWRL/Nasdaq listing mentions
- Sends alerts via **SendGrid** (or writes to a local log file as fallback)
- Persists state in `pwrl_monitor_state.json` — no duplicate alerts across restarts

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Required? | Description |
|---|---|---|
| `SENDGRID_API_KEY` | Optional | SendGrid API key for email alerts |
| `ALERT_EMAIL` | Optional | Address to send alerts to (and from) |
| `MONITOR_EMAIL` | Optional | Contact email in the SEC User-Agent header |

If `SENDGRID_API_KEY` is not set, all alerts are appended to `pwrl_monitor_alerts.log`.

---

## Usage

### Continuous monitoring (checks every 60 minutes)

```bash
python edgar_monitor.py
```

### Single immediate check

```bash
python edgar_monitor.py --check-now
```

### Print current status

```bash
python edgar_monitor.py --status
```

---

## Running in the background

### With `nohup` (keeps running after you log out)

```bash
nohup python edgar_monitor.py > edgar_monitor.out 2>&1 &
echo "PID: $!"
```

To stop it later:
```bash
# Find the process
ps aux | grep edgar_monitor.py

# Kill by PID
kill <PID>
```

Tail the output:
```bash
tail -f edgar_monitor.out
```

### As a cron job (runs every 60 minutes)

Use `--check-now` mode with cron so cron handles the scheduling:

```bash
crontab -e
```

Add this line (adjust paths):
```
0 * * * * /usr/bin/python3 /path/to/edgar_monitor.py --check-now >> /path/to/edgar_monitor.out 2>&1
```

Or every 30 minutes for higher frequency:
```
*/30 * * * * /usr/bin/python3 /path/to/edgar_monitor.py --check-now >> /path/to/edgar_monitor.out 2>&1
```

### As a systemd service (Linux, survives reboots)

Create `/etc/systemd/system/pwrl-monitor.service`:

```ini
[Unit]
Description=PWRL EDGAR Monitor
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/simpleRL
ExecStart=/usr/bin/python3 /path/to/simpleRL/edgar_monitor.py
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pwrl-monitor
sudo systemctl start pwrl-monitor
sudo systemctl status pwrl-monitor
```

---

## State file

`pwrl_monitor_state.json` is created automatically. Example:

```json
{
  "last_checked": "2025-03-24T14:00:00+00:00",
  "seen_accession_numbers": ["0002052053-25-000001"],
  "last_filing": {
    "accessionNumber": "0002052053-25-000001",
    "form": "N-2/A",
    "filingDate": "2025-03-20",
    "assessment": "Looks close — placeholders mostly cleared"
  }
}
```

Do **not** commit this file — it contains runtime state specific to your instance.

---

## Alert log (no SendGrid)

When `SENDGRID_API_KEY` is not configured, alerts are written to `pwrl_monitor_alerts.log`:

```
tail -f pwrl_monitor_alerts.log
```

---

## SEC User-Agent note

The SEC requires a descriptive `User-Agent` header on all EDGAR API requests. Set `MONITOR_EMAIL` in your `.env` to a real contact address. The agent defaults to `monitor@example.com` but using a real address is recommended per SEC guidelines.

---

## Filing types monitored

| Form | Significance |
|---|---|
| **N-2** | Initial registration statement (closed-end fund / BDC) |
| **N-2/A** | Amendment — watch for placeholder clearance |
| **8-K** | Material event — may announce listing date |
| **EFFECT** | SEC declared registration effective — listing is imminent |
