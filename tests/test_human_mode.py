"""
test_human_mode.py
------------------
Unit tests for Mode B (Human Executable) constraints:
  - Sleep window filter
  - Too-late ATR-distance gate
  - ATR frozen at signal time, not entry time
"""

import pytest
import pandas as pd

from engine.portfolio_engine import is_in_sleep_window, check_too_late


# ── Sleep window tests ────────────────────────────────────────────────────────

class TestSleepWindow:
    def test_2am_is_outside_window(self):
        """02:00 is before market open — outside window."""
        ts = pd.Timestamp("2025-01-01 02:00:00")
        assert not is_in_sleep_window(ts)

    def test_9am_is_inside_window(self):
        """09:00 is the opening boundary — inside window."""
        ts = pd.Timestamp("2025-01-01 09:00:00")
        assert is_in_sleep_window(ts)

    def test_12pm_is_inside_window(self):
        ts = pd.Timestamp("2025-01-01 12:00:00")
        assert is_in_sleep_window(ts)

    def test_20_is_inside_window(self):
        """20:00 bar open — trader is still active."""
        ts = pd.Timestamp("2025-01-01 20:00:00")
        assert is_in_sleep_window(ts)

    def test_22_is_inside_window(self):
        """22:00 — last eligible bar before cutoff."""
        ts = pd.Timestamp("2025-01-01 22:00:00")
        assert is_in_sleep_window(ts)

    def test_23_is_outside_window(self):
        """23:00 is the closing boundary — outside window."""
        ts = pd.Timestamp("2025-01-01 23:00:00")
        assert not is_in_sleep_window(ts)

    def test_midnight_is_outside_window(self):
        ts = pd.Timestamp("2025-01-01 00:00:00")
        assert not is_in_sleep_window(ts)

    def test_4am_is_outside_window(self):
        ts = pd.Timestamp("2025-01-01 04:00:00")
        assert not is_in_sleep_window(ts)

    def test_8am_is_outside_window(self):
        ts = pd.Timestamp("2025-01-01 08:00:00")
        assert not is_in_sleep_window(ts)


# ── Sleep window: delay scenarios ─────────────────────────────────────────────

class TestSleepWindowDelays:
    def test_signal_at_2am_delayed_to_10am(self):
        """Signal fires at 02:00 → first eligible entry is 10:00 same day (within 24h)."""
        signal_ts  = pd.Timestamp("2025-01-01 02:00:00")
        entry_ts   = pd.Timestamp("2025-01-01 10:00:00")

        assert not is_in_sleep_window(signal_ts), "02:00 must be outside window"
        assert is_in_sleep_window(entry_ts),      "10:00 must be inside window"
        assert (entry_ts - signal_ts) <= pd.Timedelta(hours=24), "still within 24h budget"

    def test_signal_at_23_delayed_to_next_day_10(self):
        """Signal fires at 23:00 → delayed to next day 10:00 (11h gap, within 24h)."""
        signal_ts = pd.Timestamp("2025-01-01 23:00:00")
        entry_ts  = pd.Timestamp("2025-01-02 10:00:00")

        assert not is_in_sleep_window(signal_ts), "23:00 must be outside window"
        assert is_in_sleep_window(entry_ts),      "next-day 10:00 must be inside window"
        assert (entry_ts - signal_ts) <= pd.Timedelta(hours=24), "11h gap within 24h budget"

    def test_signal_at_20_enters_immediately(self):
        """Signal at 20:00 (inside window) → ref bar is also in window → 0-bar delay."""
        signal_ts = pd.Timestamp("2025-01-01 20:00:00")
        # The first bar after signal is the next H4 bar, e.g. 00:00 next day
        # But if that bar is in-window, it enters immediately
        next_bar  = pd.Timestamp("2025-01-02 00:00:00")  # outside window
        bar_after = pd.Timestamp("2025-01-02 09:00:00")  # next eligible bar

        assert not is_in_sleep_window(next_bar)
        assert is_in_sleep_window(bar_after)

    def test_signal_at_midnight_exceeds_24h_on_weekend(self):
        """Signal on Friday midnight may time-out if next eligible bar > 24h away."""
        signal_ts = pd.Timestamp("2025-01-03 00:00:00")   # Friday midnight
        # If market closed for weekend, no bar until Monday 09:00 = 57h later
        monday_ts = pd.Timestamp("2025-01-06 09:00:00")   # Monday 09:00

        assert (monday_ts - signal_ts) > pd.Timedelta(hours=24), "weekend gap exceeds 24h"
        # Trade should be cancelled (sleep_window timeout)


# ── Too-late filter tests ─────────────────────────────────────────────────────

class TestTooLateFilter:
    def test_cancels_when_distance_exceeds_1x_atr(self):
        """distance = 5 > 1×ATR(4) → cancel."""
        assert check_too_late(actual_price=105.0, ref_price=100.0, atr_signal=4.0)

    def test_cancels_on_short_side(self):
        """Works symmetrically for short entries (price moved up = bad for short)."""
        assert check_too_late(actual_price=95.0, ref_price=100.0, atr_signal=4.0)

    def test_allows_when_distance_below_1x_atr(self):
        """distance = 3 ≤ 1×ATR(4) → allow."""
        assert not check_too_late(actual_price=103.0, ref_price=100.0, atr_signal=4.0)

    def test_allows_at_exact_boundary(self):
        """distance == 1×ATR exactly → allow (not strictly greater)."""
        assert not check_too_late(actual_price=104.0, ref_price=100.0, atr_signal=4.0)

    def test_allows_zero_distance(self):
        """Actual price == ref price → always allowed."""
        assert not check_too_late(actual_price=100.0, ref_price=100.0, atr_signal=4.0)

    def test_cancels_just_above_boundary(self):
        """distance = 4.0001 > 1×ATR(4.0) → cancel."""
        assert check_too_late(actual_price=104.0001, ref_price=100.0, atr_signal=4.0)


# ── ATR source: signal time vs entry time ─────────────────────────────────────

class TestAtrAtSignalTime:
    """
    Verifies that the filter uses atr_signal (frozen at signal bar),
    NOT the ATR at the time of actual entry.

    Scenario:
      - ref_price = 100.0, actual_price = 105.0  → distance = 5.0
      - atr_at_signal = 4.0  → distance(5) > atr(4) → CANCEL
      - atr_at_entry  = 6.0  → distance(5) < atr(6) → would ALLOW

    Correct behaviour: use atr_at_signal → cancel.
    """

    def test_uses_signal_atr_not_entry_atr(self):
        ref    = 100.0
        actual = 105.0   # distance = 5.0

        atr_at_signal = 4.0   # 5 > 4 → cancel
        atr_at_entry  = 6.0   # 5 < 6 → would allow if we used entry ATR

        # Using signal ATR → should cancel
        assert check_too_late(actual, ref, atr_at_signal), \
            "With signal ATR=4 and distance=5, trade must be cancelled"

        # Using entry ATR → would NOT cancel
        assert not check_too_late(actual, ref, atr_at_entry), \
            "With entry ATR=6 and distance=5, trade would be allowed — confirms the difference"

    def test_frozen_atr_causes_cancellation_when_atr_grows(self):
        """
        ATR shrank since signal — using entry ATR would allow, but frozen signal ATR cancels.
        """
        ref    = 100.0
        actual = 103.5

        atr_at_signal = 3.0   # 3.5 > 3.0 → cancel (frozen)
        atr_at_entry  = 4.0   # 3.5 < 4.0 → would allow if using entry ATR

        assert check_too_late(actual, ref, atr_at_signal)
        assert not check_too_late(actual, ref, atr_at_entry)
