"""
live/notifier.py
----------------
Telegram notification for the EOD signal runner.

Setup (one-time)
----------------
1. Create a bot via @BotFather on Telegram → get BOT_TOKEN
2. Start a chat with your bot, then get your CHAT_ID:
       curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
   Look for "chat": {"id": <YOUR_CHAT_ID>}
3. Set environment variables on the VPS:
       export TELEGRAM_BOT_TOKEN="123456:ABC-..."
       export TELEGRAM_CHAT_ID="987654321"
   Add both lines to ~/.bashrc or ~/.profile so they persist.

If the env vars are not set, send_notification() is a silent no-op —
the signal runner continues normally without crashing.
"""

from __future__ import annotations

import os
from typing import List

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
_TIMEOUT = 10  # seconds


def _token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")


def _fmt_direction(d: int) -> str:
    return "LONG 🟢" if d == 1 else "SHORT 🔴"


def build_message(
    today_str:    str,
    equity_eur:   float,
    open_r:       float,
    global_cap:   float,
    signals:      List[dict],   # accepted only
    stop_alerts:  List[str],
    stop_updates: List[str],    # "INSTR [SLEEVE] → new stop X.XXXX"
) -> str:
    """Build the Telegram message string (plain text, no HTML/Markdown)."""

    lines: List[str] = []
    p = lines.append

    p(f"📡 EOD SIGNAL — {today_str}")
    p(f"💰 Equity: €{equity_eur:,.0f}  |  {open_r:.1f}R / {global_cap:.0f}R open")

    # ── New entries ───────────────────────────────────────────────────────────
    p("")
    if signals:
        p(f"📥 ENTRIES tomorrow ({len(signals)}) — place at open:")
        for s in signals:
            p(f"  {_fmt_direction(s['direction'])} {s['instrument']} [{s['sleeve']}]"
              f"  stop {s['est_stop']:.4f}  ({s['units_theory']:.1f} units theory)")
    else:
        p("📥 No new entries")

    # ── Stop updates ──────────────────────────────────────────────────────────
    p("")
    if stop_updates:
        p(f"⬆ STOP UPDATES ({len(stop_updates)}) — cancel old, place new in IBKR:")
        for u in stop_updates:
            p(f"  {u}")
    else:
        p("⬆ No stop updates")

    # ── Potential stop hits ───────────────────────────────────────────────────
    p("")
    if stop_alerts:
        p(f"⚠️ STOP HITS — verify fill in IBKR ({len(stop_alerts)}):")
        for a in stop_alerts:
            p(f"  ❗ {a}")
    else:
        p("✅ No stop hits")

    return "\n".join(lines)


def send_notification(message: str) -> bool:
    """
    Send message to Telegram.

    Returns True on success, False on failure (token missing, network error, etc.).
    Never raises — the signal runner must not crash due to a notification failure.
    """
    token   = _token()
    chat_id = _chat_id()

    if not token or not chat_id:
        return False  # silently skip — env vars not configured

    if not _REQUESTS_AVAILABLE:
        print("  [notifier] 'requests' not installed — skipping Telegram notification.")
        return False

    try:
        resp = _requests.post(
            _API_URL.format(token=token),
            data={"chat_id": chat_id, "text": message},
            timeout=_TIMEOUT,
        )
        if not resp.ok:
            print(f"  [notifier] Telegram API error {resp.status_code}: {resp.text[:120]}")
            return False
        return True
    except Exception as exc:
        print(f"  [notifier] Telegram send failed: {exc}")
        return False
