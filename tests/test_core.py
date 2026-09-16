"""
tests/test_core.py
------------------
Unit tests for the core trading logic — fast, offline, no network, no LLM.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Parser ────────────────────────────────────────────────────────────────────

from trading.parser import parse_go_command  # noqa: E402


class TestGoParser(unittest.TestCase):

    def test_buy_only_with_budget(self):
        p = parse_go_command("only buy budget is 5000 rs only")
        self.assertEqual(p["direction"], "BUY")
        self.assertEqual(p["budget"], 5000.0)

    def test_sell_only_with_loss_and_interval(self):
        p = parse_go_command("only sell budget 50000 loss 5 percent every 3 minutes")
        self.assertEqual(p["direction"], "SELL")
        self.assertEqual(p["max_loss_pct"], 5.0)
        self.assertEqual(p["interval_seconds"], 180)

    def test_defaults(self):
        p = parse_go_command("")
        self.assertEqual(p["direction"], "BOTH")
        self.assertEqual(p["budget"], 10_000.0)
        self.assertEqual(p["max_loss_pct"], 20.0)

    def test_both_sides_budget(self):
        p = parse_go_command("budget is 100000 and loss taking capacity is 10 percent only")
        self.assertEqual(p["direction"], "BOTH")
        self.assertEqual(p["max_loss_pct"], 10.0)
        self.assertEqual(p["budget"], 100000.0)


# ── Agent JSON extractor ──────────────────────────────────────────────────────

from agent import extract_json  # noqa: E402


class TestExtractJson(unittest.TestCase):

    def test_fenced_json(self):
        self.assertEqual(extract_json('```json {"direction": "BULLISH"}```'),
                         {"direction": "BULLISH"})

    def test_bare_json(self):
        self.assertEqual(extract_json('vote: {"direction": "BEARISH"} done'),
                         {"direction": "BEARISH"})

    def test_trailing_comma_tolerated(self):
        self.assertEqual(extract_json('{"a": 1, "b": 2,}'), {"a": 1, "b": 2})

    def test_none_on_garbage(self):
        self.assertIsNone(extract_json("no json here at all"))

    def test_empty(self):
        self.assertIsNone(extract_json(""))
        self.assertIsNone(extract_json(None))


# ── Engine: Trade + TradingSession ────────────────────────────────────────────

from trading.engine import Trade, TradingSession  # noqa: E402


def _make_trade(**kw):
    defaults = dict(
        id="T001", expiry="26-Jun-2026", strike=24000, option_type="CE",
        action="BUY", qty=1, entry_price=100.0, current_price=100.0,
        sl=75.0, target=150.0,
    )
    defaults.update(kw)
    return Trade(**defaults)


class TestTradePnl(unittest.TestCase):

    def test_buy_pnl(self):
        t = _make_trade()
        t.update_price(120.0)
        self.assertEqual(t.pnl, 20.0 * 75)

    def test_sell_pnl_inverted(self):
        t = _make_trade(action="SELL")
        t.update_price(90.0)
        self.assertEqual(t.pnl, 10.0 * 75)

    def test_buy_target_hit(self):
        t = _make_trade()
        self.assertTrue(t.update_price(150.0))
        self.assertEqual(t.status, "TARGET_HIT")

    def test_buy_sl_hit(self):
        t = _make_trade()
        self.assertTrue(t.update_price(70.0))
        self.assertEqual(t.status, "SL_HIT")

    def test_sell_sl_hit_is_above(self):
        t = _make_trade(action="SELL", sl=110.0, target=60.0)
        self.assertTrue(t.update_price(115.0))
        self.assertEqual(t.status, "SL_HIT")

    def test_no_autoclose_in_between(self):
        t = _make_trade()
        self.assertFalse(t.update_price(100.0))
        self.assertEqual(t.status, "OPEN")


class TestTradingSession(unittest.TestCase):

    def test_direction_filter_blocks_opposite(self):
        s = TradingSession(budget=100_000, max_loss_pct=10, direction="BUY")
        ok, _ = s.add_trade(_make_trade(action="SELL"))
        self.assertFalse(ok)

    def test_budget_check(self):
        s = TradingSession(budget=5_000, max_loss_pct=10, direction="BOTH")
        ok, _ = s.add_trade(_make_trade(entry_price=200.0))   # 200*75 = 15000 > 5000
        self.assertFalse(ok)

    def test_recovery_mode_blocks_new_trades(self):
        s = TradingSession(budget=10_000, max_loss_pct=5, direction="BOTH")
        ok, _ = s.add_trade(_make_trade())
        self.assertTrue(ok)
        # Force a big loss → recovery
        t = s.open_trades[0]
        t.update_price(10.0)   # -90*75 = -6750 ≤ -500
        s.check_and_set_recovery_mode()
        self.assertTrue(s.recovery_mode)
        ok, reason = s.add_trade(_make_trade(id="T002"))
        self.assertFalse(ok)
        self.assertIn("Recovery", reason)

    def test_close_all(self):
        s = TradingSession(budget=100_000, max_loss_pct=10, direction="BOTH")
        s.add_trade(_make_trade())
        s.add_trade(_make_trade(id="T002", strike=24100))
        ok, _ = s.close_trade("ALL", 110.0)
        self.assertTrue(ok)
        self.assertEqual(len(s.open_trades), 0)


# ── Voting + simulation helpers ───────────────────────────────────────────────

from trading.sim_mode import _vote_decision, _fallback_premium, pick_strikes  # noqa: E402


class TestVoting(unittest.TestCase):

    def test_agree_bull(self):
        self.assertEqual(_vote_decision(1, "BULLISH"), "BULLISH")

    def test_conflict_holds(self):
        self.assertEqual(_vote_decision(1, "BEARISH"), "HOLD")

    def test_llm_conviction_when_kronos_neutral(self):
        self.assertEqual(_vote_decision(0, "BEARISH"), "BEARISH")

    def test_no_votes_holds(self):
        self.assertEqual(_vote_decision(0, "SIDEWAYS"), "HOLD")


class TestPremiums(unittest.TestCase):

    def test_itm_call_has_intrinsic(self):
        # spot 24100 vs strike 24000 CE → intrinsic 100 + time value
        p = _fallback_premium(24000, 24100, "CE", 375)
        self.assertGreaterEqual(p, 100.0)

    def test_otm_put_decays_with_time(self):
        full = _fallback_premium(24000, 23500, "PE", 375)
        near = _fallback_premium(24000, 23500, "PE", 60)
        self.assertGreater(full, near)

    def test_zero_intrinsic_otm(self):
        p = _fallback_premium(24500, 24000, "CE", 375)
        self.assertGreater(p, 0)   # time value only


class TestStrikePicker(unittest.TestCase):

    def test_picks_around_spot(self):
        s = pick_strikes(24150, n=4)
        self.assertEqual(len(s), 4)
        self.assertTrue(all(x % 50 == 0 for x in s))
        self.assertTrue(any(x < 24150 for x in s))
        self.assertTrue(any(x > 24150 for x in s))


# ── Expiry normalization (Groww → NSE format) ─────────────────────────────────

from data.groww_feed import _norm_expiry  # noqa: E402


class TestExpiryNormalization(unittest.TestCase):

    def test_iso_to_ddmon(self):
        self.assertEqual(_norm_expiry("2026-09-22"), "22-Sep-2026")

    def test_already_ddmon_unchanged(self):
        self.assertEqual(_norm_expiry("22-Sep-2026"), "22-Sep-2026")

    def test_garbage_passthrough(self):
        self.assertEqual(_norm_expiry("not-a-date"), "not-a-date")

    def test_empty(self):
        self.assertEqual(_norm_expiry(""), "")


# ── Rules file (uses a temp dir override) ─────────────────────────────────────

import trading.rules as rules_mod  # noqa: E402


class TestRules(unittest.TestCase):

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self._old = rules_mod.RULES_FILE
        rules_mod.RULES_FILE = Path(self._tmp.name) / "rule.md"

    def tearDown(self):
        rules_mod.RULES_FILE = self._old
        self._tmp.cleanup()

    def test_record_and_load(self):
        rules_mod.record_lessons(["Sell premium in low VIX"],
                                 section="REGIME SIDEWAYS", source_note="test")
        block = rules_mod.load_rules_block()
        self.assertIn("Sell premium in low VIX", block)
        self.assertIn("[SIDEWAYS]", block)   # regime context carries into prompt

    def test_dedup(self):
        rules_mod.record_lessons(["rule A"], section="GLOBAL")
        rules_mod.record_lessons(["rule A"], section="GLOBAL")
        block = rules_mod.load_rules_block()
        self.assertEqual(block.count("rule A"), 1)

    def test_session_outcome_records(self):
        rules_mod.record_session_outcome(
            regime="SIDEWAYS", kronos_agree=3, kronos_disagree=1,
            total_pnl=1500.0, trade_count=4, win_rate=75.0,
            ai_summary="Momentum votes failed near OI walls.",
        )
        block = rules_mod.load_rules_block()
        self.assertIn("SIDEWAYS", block)
        self.assertIn("profitable session", block)


# ── Groww NEXT_DATA parsing (offline fixture) ─────────────────────────────────

from data.groww_feed import _option_chain_from_next  # noqa: E402


class TestGrowwParser(unittest.TestCase):

    def _fixture(self):
        def leg(ltp, oi, prev_oi, iv):
            return {"liveData": {"ltp": ltp, "oi": oi, "prevOI": prev_oi,
                                 "close": ltp, "dayChange": 0},
                    "greeks": {"iv": iv}}
        return {"props": {"pageProps": {"data": {"optionChain": {
            "aggregatedDetails": {
                "currentExpiry": "2026-09-22",
                "expiryDates": ["2026-09-22", "2026-09-29"],
                "lotSize": 75,
            },
            "optionContracts": [
                {"strikePrice": 2300000, "ce": leg(320, 900, 700, 12), "pe": leg(2.0, 100, 90, 13)},
                {"strikePrice": 2325000, "ce": leg(105, 500, 400, 11), "pe": leg(95, 600, 550, 12)},
                {"strikePrice": 2350000, "ce": leg(25, 1200, 1000, 12.5), "pe": leg(360, 1100, 900, 13)},
            ],
        }}}}}

    def test_strikes_scaled_from_paisa(self):
        oc = _option_chain_from_next(self._fixture())
        strikes = [r["strike"] for r in oc["rows"]]
        self.assertEqual(strikes, [23000, 23250, 23500])

    def test_pcr(self):
        oc = _option_chain_from_next(self._fixture())
        ce_oi = 900 + 500 + 1200
        pe_oi = 100 + 600 + 1100
        self.assertEqual(oc["pcr"], round(pe_oi / ce_oi, 2))

    def test_max_pain(self):
        oc = _option_chain_from_next(self._fixture())
        # With these OI values pain minimum is at 23250
        self.assertEqual(oc["max_pain"], 23250)

    def test_walls(self):
        oc = _option_chain_from_next(self._fixture())
        self.assertEqual(oc["ce_wall"], 23500)   # highest CE OI
        self.assertEqual(oc["pe_wall"], 23500)   # highest PE OI

    def test_expiry_dates_preserved(self):
        oc = _option_chain_from_next(self._fixture())
        self.assertEqual(oc["expiry_dates"], ["2026-09-22", "2026-09-29"])


if __name__ == "__main__":
    unittest.main()
