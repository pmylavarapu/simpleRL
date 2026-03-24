"""
EDGAR Monitoring Agent for Powerlaw Corp (PWRL)
CIK: 2052053 | Ticker: PWRL

Polls SEC EDGAR every 60 minutes for new filings and assesses
whether a Nasdaq listing appears imminent.

Usage:
    python edgar_monitor.py             # start continuous monitoring
    python edgar_monitor.py --check-now # single immediate check
    python edgar_monitor.py --status    # print last check time & recent filing
"""

import argparse
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import schedule
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

CIK_RAW = "2052053"
CIK_PADDED = "0002052053"
EDGAR_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK_PADDED}.json"
EDGAR_ARCHIVE_BASE = f"https://www.sec.gov/Archives/edgar/data/{CIK_RAW}"
EDGAR_FILING_PAGE = (
    f"https://www.sec.gov/cgi-bin/browse-edgar"
    f"?action=getcompany&CIK={CIK_RAW}&type=N-2&dateb=&owner=include&count=10"
)

WATCH_TYPES = {"N-2", "N-2/A", "8-K", "EFFECT"}
NEWS_RSS_URL = (
    "https://news.google.com/rss/search?q=Powerlaw+PWRL+Nasdaq+listing"
)
STATE_FILE = "pwrl_monitor_state.json"
ALERT_LOG_FILE = "pwrl_monitor_alerts.log"
POLL_INTERVAL_MINUTES = 60

# SEC requires a descriptive User-Agent; use env var or sensible default
MONITOR_EMAIL = os.getenv("MONITOR_EMAIL", "monitor@example.com")
HEADERS = {
    "User-Agent": f"PWRL EDGAR Monitor {MONITOR_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}
ARCHIVE_HEADERS = {
    "User-Agent": f"PWRL EDGAR Monitor {MONITOR_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
}

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("edgar_monitor")

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def load_state() -> dict:
    """Load persisted state from JSON file."""
    p = Path(STATE_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Could not read state file (%s); starting fresh.", exc)
    return {"last_checked": None, "seen_accession_numbers": [], "last_filing": None}


def save_state(state: dict) -> None:
    """Persist state to JSON file."""
    Path(STATE_FILE).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# EDGAR polling
# ---------------------------------------------------------------------------


def fetch_submissions() -> dict:
    """Fetch the EDGAR submissions JSON for PWRL."""
    resp = requests.get(EDGAR_SUBMISSIONS_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_recent_filings(data: dict) -> list[dict]:
    """
    Convert the flat parallel arrays in `filings.recent` into a list of
    filing dicts, keeping only those in WATCH_TYPES.
    """
    recent = data.get("filings", {}).get("recent", {})
    keys = [
        "accessionNumber",
        "filingDate",
        "form",
        "primaryDocument",
        "primaryDocDescription",
    ]
    # All arrays are the same length
    accessions = recent.get("accessionNumber", [])
    filings = []
    for i, accession in enumerate(accessions):
        form = recent.get("form", [])[i]
        if form not in WATCH_TYPES:
            continue
        filings.append(
            {
                "accessionNumber": accession,
                "filingDate": recent.get("filingDate", [])[i],
                "form": form,
                "primaryDocument": recent.get("primaryDocument", [])[i],
                "description": recent.get("primaryDocDescription", [])[i],
            }
        )
    return filings


def get_new_filings(all_filings: list[dict], seen: list[str]) -> list[dict]:
    """Return filings whose accession numbers haven't been seen before."""
    seen_set = set(seen)
    return [f for f in all_filings if f["accessionNumber"] not in seen_set]


# ---------------------------------------------------------------------------
# Document fetching & signal analysis
# ---------------------------------------------------------------------------


def accession_to_path(accession: str) -> str:
    """Convert '0002052053-24-000042' → '000205205324000042'."""
    return accession.replace("-", "")


def fetch_filing_text(accession: str, primary_doc: str) -> str:
    """Fetch the full text of an N-2 or N-2/A primary document."""
    path = accession_to_path(accession)
    url = f"{EDGAR_ARCHIVE_BASE}/{path}/{primary_doc}"
    log.info("Fetching document: %s", url)
    try:
        resp = requests.get(url, headers=ARCHIVE_HEADERS, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "html" in content_type or primary_doc.endswith((".htm", ".html")):
            soup = BeautifulSoup(resp.content, "lxml")
            return soup.get_text(separator=" ", strip=True)
        return resp.text
    except Exception as exc:
        log.warning("Could not fetch document %s: %s", url, exc)
        return ""


# Patterns for signal detection
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z\s\-/]{1,40}\]")  # e.g. [DATE], [FINANCIAL ADVISOR]
_CONCRETE_DATE_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+20\d{2}",
    re.IGNORECASE,
)
_LISTING_DATE_RE = re.compile(
    r"(listing|trading|commence[sd]?|effective)\s+.{0,60}?"
    r"(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+20\d{2}",
    re.IGNORECASE,
)
_STIFEL_BRACKET_RE = re.compile(r"\[Stifel\]|\[FINANCIAL ADVISOR\]|\[UNDERWRITER\]", re.IGNORECASE)


def analyze_filing(text: str) -> dict:
    """
    Scan document text for imminence signals.

    Returns a dict with boolean flags:
      - has_placeholders:       True if [BRACKETED PLACEHOLDERS] remain
      - placeholder_count:      how many distinct placeholder matches
      - stifel_confirmed:       Stifel named without brackets
      - has_concrete_date:      a concrete calendar date near listing language
      - as_soon_as_practicable: the vague "as soon as practicable" language remains
    """
    if not text:
        return {
            "has_placeholders": None,
            "placeholder_count": 0,
            "stifel_confirmed": False,
            "has_concrete_date": False,
            "as_soon_as_practicable": False,
            "fetch_failed": True,
        }

    placeholders = _PLACEHOLDER_RE.findall(text)
    placeholder_count = len(placeholders)

    stifel_in_text = bool(re.search(r"\bStifel\b", text, re.IGNORECASE))
    stifel_bracketed = bool(_STIFEL_BRACKET_RE.search(text))
    stifel_confirmed = stifel_in_text and not stifel_bracketed

    has_concrete_date = bool(_LISTING_DATE_RE.search(text))
    as_soon_as_practicable = bool(
        re.search(r"as soon as practicable", text, re.IGNORECASE)
    )

    return {
        "has_placeholders": placeholder_count > 0,
        "placeholder_count": placeholder_count,
        "stifel_confirmed": stifel_confirmed,
        "has_concrete_date": has_concrete_date,
        "as_soon_as_practicable": as_soon_as_practicable,
        "fetch_failed": False,
    }


def assess_filing(form: str, signals: dict) -> str:
    """Return a plain-English listing-imminence assessment."""
    if form == "EFFECT":
        return "IMMINENT — registration effective or date confirmed"

    if signals.get("fetch_failed"):
        return "Unknown — could not fetch document"

    if signals.get("has_concrete_date") and not signals.get("has_placeholders") and signals.get("stifel_confirmed"):
        return "IMMINENT — registration effective or date confirmed"

    cleared = not signals.get("has_placeholders") and signals.get("stifel_confirmed")
    mostly_cleared = (
        signals.get("placeholder_count") is not None
        and signals.get("placeholder_count", 0) <= 3
        and signals.get("stifel_confirmed")
    )

    if cleared or mostly_cleared:
        return "Looks close — placeholders mostly cleared"

    return "Routine amendment"


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------


def check_google_news() -> list[dict]:
    """
    Fetch Google News RSS for PWRL Nasdaq listing mentions.
    Returns items published within the last 48 hours.
    """
    try:
        resp = requests.get(NEWS_RSS_URL, headers=ARCHIVE_HEADERS, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as exc:
        log.warning("Could not fetch/parse Google News RSS: %s", exc)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
    items = []
    for item in root.iter("item"):
        title = item.findtext("title", "").strip()
        link = item.findtext("link", "").strip()
        pub_date_str = item.findtext("pubDate", "").strip()
        try:
            # RFC 2822 dates, e.g. "Mon, 24 Mar 2025 10:00:00 GMT"
            from email.utils import parsedate_to_datetime
            pub_dt = parsedate_to_datetime(pub_date_str)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        except Exception:
            pub_dt = datetime.now(timezone.utc)

        if pub_dt >= cutoff:
            items.append({"title": title, "link": link, "published": pub_date_str})

    log.info("Google News: found %d recent items.", len(items))
    return items


# ---------------------------------------------------------------------------
# Alert / notification
# ---------------------------------------------------------------------------


def build_alert_body(new_filings: list[dict], news_items: list[dict]) -> tuple[str, str]:
    """Return (subject, body) for the alert message."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if len(new_filings) == 1:
        f = new_filings[0]
        subject = f"[PWRL Monitor] New {f['form']} filing — {f['assessment']}"
    else:
        subject = f"[PWRL Monitor] {len(new_filings)} new filings detected"

    lines = [
        "=" * 60,
        "POWERLAW CORP (PWRL) — EDGAR FILING ALERT",
        f"Checked at: {now_str}",
        "=" * 60,
        "",
    ]

    for f in new_filings:
        path = accession_to_path(f["accessionNumber"])
        sec_link = f"{EDGAR_ARCHIVE_BASE}/{path}/{f['primaryDocument']}"
        filing_index = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={CIK_RAW}&type={f['form'].replace('/', '%2F')}"
            f"&dateb=&owner=include&count=10"
        )
        lines += [
            f"Form:        {f['form']}",
            f"Filed:       {f['filingDate']}",
            f"Accession:   {f['accessionNumber']}",
            f"Description: {f.get('description', 'N/A')}",
            f"Assessment:  {f['assessment']}",
            f"Document:    {sec_link}",
            f"All filings: {filing_index}",
        ]
        if "signals" in f and f["signals"] and not f["signals"].get("fetch_failed"):
            s = f["signals"]
            lines += [
                "",
                "  Signal details:",
                f"    Bracketed placeholders remaining: {s.get('placeholder_count', 'N/A')}",
                f"    Stifel confirmed (no brackets):   {s.get('stifel_confirmed', 'N/A')}",
                f"    Concrete listing date found:      {s.get('has_concrete_date', 'N/A')}",
                f"    'As soon as practicable' present: {s.get('as_soon_as_practicable', 'N/A')}",
            ]
        lines.append("")

    if news_items:
        lines += ["", "-" * 60, "RECENT NEWS MENTIONS (last 48h)", "-" * 60, ""]
        for item in news_items:
            lines += [
                f"  {item['title']}",
                f"  Published: {item['published']}",
                f"  Link:      {item['link']}",
                "",
            ]

    lines += ["", "=" * 60, "End of alert", "=" * 60]
    return subject, "\n".join(lines)


def send_alert(subject: str, body: str) -> None:
    """Send alert via SendGrid or fall back to local log file."""
    if SENDGRID_API_KEY and ALERT_EMAIL:
        _send_via_sendgrid(subject, body)
    else:
        _write_to_log(subject, body)


def _send_via_sendgrid(subject: str, body: str) -> None:
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=ALERT_EMAIL,
            to_emails=ALERT_EMAIL,
            subject=subject,
            plain_text_content=body,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        log.info("SendGrid alert sent (status %s).", response.status_code)
    except Exception as exc:
        log.error("SendGrid send failed: %s — falling back to log.", exc)
        _write_to_log(subject, body)


def _write_to_log(subject: str, body: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"\n{'#'*70}\n# {timestamp}\n# {subject}\n{'#'*70}\n{body}\n"
    with open(ALERT_LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(entry)
    log.info("Alert written to %s", ALERT_LOG_FILE)


# ---------------------------------------------------------------------------
# Core check routine
# ---------------------------------------------------------------------------


def run_check() -> None:
    """Perform one full check: EDGAR + news, alert on new findings."""
    log.info("--- Starting EDGAR check ---")
    state = load_state()
    seen = state.get("seen_accession_numbers", [])

    # --- EDGAR ---
    try:
        data = fetch_submissions()
    except Exception as exc:
        log.error("Failed to fetch EDGAR submissions: %s", exc)
        return

    all_filings = parse_recent_filings(data)
    log.info("Watchlisted forms in recent filings: %d total", len(all_filings))

    new_filings = get_new_filings(all_filings, seen)
    log.info("New (unseen) filings: %d", len(new_filings))

    enriched = []
    for filing in new_filings:
        form = filing["form"]
        signals: dict = {}

        if form in ("N-2", "N-2/A"):
            time.sleep(1)  # be polite to SEC servers
            text = fetch_filing_text(filing["accessionNumber"], filing["primaryDocument"])
            signals = analyze_filing(text)
            assessment = assess_filing(form, signals)
        else:
            # 8-K or EFFECT: no document analysis needed
            assessment = assess_filing(form, {})

        filing["signals"] = signals
        filing["assessment"] = assessment
        enriched.append(filing)

        # Update state immediately so a crash mid-loop doesn't re-alert
        seen.append(filing["accessionNumber"])
        state["seen_accession_numbers"] = seen
        state["last_filing"] = {
            "accessionNumber": filing["accessionNumber"],
            "form": form,
            "filingDate": filing["filingDate"],
            "assessment": assessment,
        }

    # --- News ---
    news_items = check_google_news()

    # --- Alert ---
    if enriched or news_items:
        subject, body = build_alert_body(enriched, news_items)
        send_alert(subject, body)
        for f in enriched:
            log.info(
                "  [%s] %s filed %s → %s",
                f["accessionNumber"],
                f["form"],
                f["filingDate"],
                f["assessment"],
            )
    else:
        log.info("No new filings and no news. Nothing to report.")

    state["last_checked"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    log.info("--- Check complete. State saved. ---")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def run_scheduler() -> None:
    """Start the polling loop (runs indefinitely)."""
    log.info(
        "Scheduler started — will check every %d minutes. Press Ctrl+C to stop.",
        POLL_INTERVAL_MINUTES,
    )
    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(run_check)
    while True:
        schedule.run_pending()
        time.sleep(30)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_status() -> None:
    """Print the current monitoring state to stdout."""
    state = load_state()
    last = state.get("last_checked") or "Never"
    seen_count = len(state.get("seen_accession_numbers", []))
    last_filing = state.get("last_filing")

    print("=" * 50)
    print("PWRL EDGAR Monitor — Status")
    print("=" * 50)
    print(f"Last checked:       {last}")
    print(f"Filings seen so far: {seen_count}")
    if last_filing:
        print()
        print("Most recent filing:")
        print(f"  Form:        {last_filing['form']}")
        print(f"  Filed:       {last_filing['filingDate']}")
        print(f"  Accession:   {last_filing['accessionNumber']}")
        print(f"  Assessment:  {last_filing['assessment']}")
    else:
        print("Most recent filing: (none yet)")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EDGAR monitoring agent for Powerlaw Corp (PWRL / CIK 2052053)."
    )
    parser.add_argument(
        "--check-now",
        action="store_true",
        help="Run an immediate check and exit.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print last check time and most recent filing, then exit.",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.check_now:
        run_check()
    else:
        run_check()          # always do one check on startup
        run_scheduler()      # then schedule subsequent checks


if __name__ == "__main__":
    main()
