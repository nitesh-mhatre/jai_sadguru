"""
tests/test_core.py
------------------
Unit tests for the core trading logic — fast, offline, no network, no LLM.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import time
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

    def test_budget_explicit_flag(self):
        self.assertTrue(parse_go_command("budget 50000")["budget_explicit"])
        self.assertTrue(parse_go_command("only sell budget 25000")["budget_explicit"])
        # "/sim 3d" states no budget — callers may substitute a sensible default
        self.assertFalse(parse_go_command("3d")["budget_explicit"])
        self.assertFalse(parse_go_command("")["budget_explicit"])


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

    def test_spot_estimate_uses_put_call_parity(self):
        # ATM is 23250 (CE 105 − PE 95 = 10) → spot ≈ 23250 + 10
        oc = _option_chain_from_next(self._fixture())
        self.assertEqual(oc["atm"], 23250)
        self.assertEqual(oc["spot"], 23260.0)
        self.assertEqual(oc["atm_straddle"], 200.0)


# ── Groww chain → predictor metrics (pure, offline) ───────────────────────────

from trading.next_predictor import option_metrics_from_groww_rows  # noqa: E402


def _groww_rows():
    def leg(ltp, oi, prev_oi, iv):
        return {"ltp": ltp, "oi": oi, "prev_oi": prev_oi, "iv": iv}
    return [
        {"strike": 23000, "ce": leg(320, 900, 700, 12),   "pe": leg(2.0, 100, 90, 13)},
        {"strike": 23250, "ce": leg(105, 500, 400, 11),   "pe": leg(95, 600, 550, 12)},
        {"strike": 23500, "ce": leg(25, 1200, 1000, 12.5), "pe": leg(360, 1100, 900, 13)},
    ]


class TestGrowwMetrics(unittest.TestCase):

    def test_core_metrics(self):
        m = option_metrics_from_groww_rows(_groww_rows(), expiry="22-Sep-2026")
        self.assertEqual(m["source"], "groww")
        self.assertEqual(m["atm"], 23250)
        self.assertEqual(m["pcr"], round(1800 / 2600, 2))
        self.assertEqual(m["max_pain"], 23250)
        self.assertEqual(m["ce_wall"], 23500)
        self.assertEqual(m["pe_wall"], 23500)
        self.assertEqual(m["fresh_ce"], 500)
        self.assertEqual(m["fresh_pe"], 260)
        self.assertEqual(m["unwind_ce"], 0)
        self.assertEqual(m["expiry"], "22-Sep-2026")

    def test_spot_parity_and_atm_strikes(self):
        m = option_metrics_from_groww_rows(_groww_rows())
        self.assertEqual(m["spot"], 23260.0)
        self.assertEqual(m["atm_ce_ltp"], 105)
        self.assertEqual(m["atm_pe_ltp"], 95)
        # OTM strikes step outward from ATM
        self.assertEqual(m["otm_ce1"], 23500)
        self.assertEqual(m["otm_pe1"], 23000)

    def test_empty_rows_returns_error(self):
        self.assertIn("error", option_metrics_from_groww_rows([]))


# ── Loss → rule.md section mapping ────────────────────────────────────────────

from trading.live_mode import loss_section  # noqa: E402


class TestLossSection(unittest.TestCase):

    def test_mapping(self):
        self.assertEqual(loss_section("SIDEWAYS"), "REGIME SIDEWAYS")
        self.assertEqual(loss_section("DIRECTIONAL_BULL"), "REGIME DIRECTIONAL")
        self.assertEqual(loss_section("STRONGLY_BEARISH"), "REGIME DIRECTIONAL")
        self.assertEqual(loss_section("VOLATILE"), "REGIME VOLATILE")
        self.assertEqual(loss_section("UNKNOWN"), "GLOBAL")
        self.assertEqual(loss_section(""), "GLOBAL")


class TestTradeCloseReason(unittest.TestCase):

    def test_sl_hit_sets_reason(self):
        t = _make_trade()
        t.update_price(70.0)
        self.assertEqual(t.status, "SL_HIT")
        self.assertEqual(t.close_reason, "stop-loss hit")

    def test_manual_close_sets_reason(self):
        s = TradingSession(budget=100_000, max_loss_pct=10, direction="BOTH")
        s.add_trade(_make_trade())
        s.close_trade("T001", 110.0)
        self.assertEqual(s.closed_trades[0].close_reason, "manual close")


# ── Breakeven + trailing stop-loss management ─────────────────────────────────

class TestStopLossManagement(unittest.TestCase):

    def test_starts_at_initial_stage(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=150.0)
        self.assertEqual(t.sl_stage, "INITIAL")
        self.assertEqual(t.initial_sl, 75.0)
        self.assertFalse(t.sl_moved)

    def test_no_move_before_trigger(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=150.0)
        t.update_price(110.0)                    # only 20% of the move
        self.assertEqual(t.sl, 75.0)
        self.assertEqual(t.sl_stage, "INITIAL")

    def test_breakeven_after_half_move(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=150.0, trail_pct=0.5)
        t.update_price(124.0)                    # 48% — not yet
        self.assertEqual(t.sl_stage, "INITIAL")
        self.assertFalse(t.update_price(126.0))  # 52% — breakeven, but trail below entry
        self.assertEqual(t.sl_stage, "BREAKEVEN")
        self.assertEqual(t.sl, 100.0)
        self.assertTrue(t.sl_moved)

    def test_breakeven_stop_exits_near_flat(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=150.0, trail_pct=0.5)
        t.update_price(126.0)                    # SL now at entry
        self.assertTrue(t.update_price(99.0))    # dips to entry
        self.assertEqual(t.status, "SL_HIT")
        self.assertIn("breakeven", t.close_reason)
        self.assertGreater(t.pnl, -100)          # tiny, vs the original 25-pt risk

    def test_trailing_only_moves_up(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=200.0)
        t.update_price(170.0)                    # BE + trail 170 × 0.75 = 127.5
        self.assertEqual(t.sl_stage, "TRAILING")
        self.assertAlmostEqual(t.sl, 127.5, places=2)
        t.update_price(160.0)                    # pullback must NOT lower the SL
        self.assertAlmostEqual(t.sl, 127.5, places=2)

    def test_trailing_stop_locks_profit(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=200.0)
        t.update_price(170.0)
        self.assertTrue(t.update_price(120.0))   # below the trailing SL
        self.assertEqual(t.status, "SL_HIT")
        self.assertIn("trailing", t.close_reason)
        self.assertGreater(t.pnl, 0)             # exited in profit, not a round-trip

    def test_sell_side_trailing(self):
        t = _make_trade(action="SELL", entry_price=100.0, sl=130.0, target=50.0)
        t.update_price(75.0)                     # BE then trail 75 × 1.25 = 93.75
        self.assertEqual(t.sl_stage, "TRAILING")
        self.assertAlmostEqual(t.sl, 93.75, places=2)
        self.assertTrue(t.update_price(95.0))    # premium back above the trail
        self.assertEqual(t.status, "SL_HIT")
        self.assertGreater(t.pnl, 0)

    def test_progress_zero_when_target_wrong_side(self):
        t = _make_trade(entry_price=100.0, sl=75.0, target=90.0)
        t.mfe_price = 120.0
        self.assertEqual(t.profit_progress, 0.0)
        t._manage_stop_loss()
        self.assertEqual(t.sl, 75.0)


# ── Partial profit booking (T1) ───────────────────────────────────────────────

class TestPartialBooking(unittest.TestCase):

    def test_single_lot_cannot_be_split(self):
        t = _make_trade(qty=1, entry_price=100.0, sl=75.0, target=200.0)
        self.assertEqual(t.planned_partial_lots, 0)
        t.update_price(160.0)                    # 60% of the move
        self.assertEqual(t.partial_booked_lots, 0)
        self.assertEqual(t.remaining_lots, 1)

    def test_books_half_at_t1(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0)
        self.assertEqual(t.planned_partial_lots, 1)
        self.assertAlmostEqual(t.partial_target_price, 150.0, places=2)
        t.update_price(149.0)                    # just below T1
        self.assertEqual(t.partial_booked_lots, 0)
        t.update_price(150.0)                    # T1 hit
        self.assertEqual(t.partial_booked_lots, 1)
        self.assertEqual(t.remaining_lots, 1)
        self.assertAlmostEqual(t.partial_pnl, 3750.0, places=2)   # 1 lot × 75 × 50

    def test_pnl_adds_partial_and_runner(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0)
        t.update_price(150.0)                    # book 1 lot
        t.update_price(160.0)                    # runner marked up
        self.assertAlmostEqual(t.pnl, 3750.0 + 4500.0, places=2)

    def test_no_partial_on_a_loser(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0)
        t.update_price(80.0)
        self.assertEqual(t.partial_booked_lots, 0)
        self.assertEqual(t.partial_pnl, 0.0)

    def test_remaining_cost_basis_frees_budget(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0)
        self.assertEqual(t.cost_basis, 100.0 * 150)
        t.update_price(150.0)
        self.assertEqual(t.remaining_cost_basis, 100.0 * 75)

    def test_explicit_partial_target(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0,
                        partial_target=130.0)
        self.assertEqual(t.partial_target_price, 130.0)
        t.update_price(130.0)
        self.assertEqual(t.partial_booked_lots, 1)

    def test_sell_side_partial(self):
        t = _make_trade(qty=2, action="SELL", entry_price=100.0,
                        sl=150.0, target=0.0)
        self.assertAlmostEqual(t.partial_target_price, 50.0, places=2)
        t.update_price(50.0)
        self.assertEqual(t.partial_booked_lots, 1)
        self.assertAlmostEqual(t.partial_pnl, 3750.0, places=2)

    def test_partial_disabled_by_pct_zero(self):
        t = _make_trade(qty=2, entry_price=100.0, sl=75.0, target=200.0,
                        partial_pct=0.0)
        t.update_price(160.0)
        self.assertEqual(t.partial_booked_lots, 0)


# ── Time-based exits ──────────────────────────────────────────────────────────

from trading.engine import _hhmm_to_minutes  # noqa: E402


class TestTimeExits(unittest.TestCase):

    def test_hhmm_parsing(self):
        self.assertEqual(_hhmm_to_minutes("15:15"), 915)
        self.assertEqual(_hhmm_to_minutes("09:05"), 545)
        self.assertEqual(_hhmm_to_minutes("bad"), -1)
        self.assertEqual(_hhmm_to_minutes("25:99"), -1)
        self.assertEqual(_hhmm_to_minutes(""), -1)

    def test_hard_exit_time(self):
        t   = _make_trade()
        now = time.time()
        self.assertEqual(t.should_time_exit(now, "15:14", "15:15"), "")
        self.assertIn("hard time exit", t.should_time_exit(now, "15:15", "15:15"))
        self.assertIn("hard time exit", t.should_time_exit(now, "15:20", "15:15"))

    def test_max_hold(self):
        t   = _make_trade(max_hold_minutes=60)
        now = time.time()
        t.opened_at_ts = now - 59 * 60
        self.assertEqual(t.should_time_exit(now, "11:00", "15:15"), "")
        t.opened_at_ts = now - 61 * 60
        self.assertIn("max hold 60m", t.should_time_exit(now, "11:00", "15:15"))

    def test_disabled_when_no_rules(self):
        t = _make_trade(max_hold_minutes=None)
        t.opened_at_ts = time.time() - 10_000
        self.assertEqual(t.should_time_exit(time.time(), "11:00", ""), "")

    def test_closed_order_never_times_out(self):
        t = _make_trade()
        t._close("CLOSED", 110.0, "manual close")
        self.assertEqual(t.should_time_exit(time.time(), "15:30", "15:15"), "")

    def test_held_minutes(self):
        t = _make_trade()
        t.opened_at_ts = 1_000_000.0
        self.assertAlmostEqual(t.held_minutes(1_000_000.0 + 90 * 60), 90.0, places=3)


# ── Level premium series (index-aligned, Groww-anchored) ──────────────────────

import pandas as pd  # noqa: E402

from trading.sim_mode import (  # noqa: E402
    SIM_WINDOW_BARS, _anchored_premium_series, _model_premium_series,
    _premium_source, plan_replay,
)


class TestReplayPlan(unittest.TestCase):
    """The replay must slide across the WHOLE download, 1 candle per decision."""

    def test_full_series_is_replayed(self):
        # 500 candles, 200-bar window: bars 0-200, then 1-201 … 299 more steps
        win, cycles = plan_replay(500)
        self.assertEqual(win, SIM_WINDOW_BARS)
        self.assertEqual(cycles, 500 - win - 1)

    def test_decisions_cover_the_tail_of_the_data(self):
        # The loop stops at n-2 so the very last candle is left to validate the
        # final prediction, so the last decision's window ends on bar n-2.
        win, cycles = plan_replay(500)
        self.assertEqual(win + cycles - 1, 498)
        self.assertEqual(plan_replay(1000)[0] + plan_replay(1000)[1] - 1, 998)

    def test_longer_download_means_more_decisions(self):
        self.assertGreater(plan_replay(1000)[1], plan_replay(500)[1])

    def test_short_download_shrinks_window_but_still_runs(self):
        win, cycles = plan_replay(150)
        self.assertLess(win, SIM_WINDOW_BARS)
        self.assertGreaterEqual(cycles, 1)
        self.assertEqual(cycles, 150 - win - 1)

    def test_explicit_window_and_cycle_cap(self):
        self.assertEqual(plan_replay(500, window=100), (100, 399))
        self.assertEqual(plan_replay(500, max_cycles=10)[1], 10)

    def test_never_exceeds_available_data(self):
        for n in (5, 10, 40, 150):
            win, cycles = plan_replay(n, max_cycles=10_000)
            self.assertLessEqual(win + cycles - 1, n - 1)
            self.assertGreaterEqual(win, 2)


def _index_grid(prices):
    idx = pd.date_range("2026-09-15 09:15", periods=len(prices), freq="5min")
    return pd.DataFrame({"open": prices, "high": prices, "low": prices,
                         "close": prices, "volume": 0}, index=idx)


class TestLevelPremiumSeries(unittest.TestCase):

    def test_anchored_to_real_chain_quote(self):
        hist = _index_grid([23250.0] * 10)
        leg  = {"ltp": 115.8, "delta": -0.39, "theta": -12.4, "iv": 13.02}
        df   = _model_premium_series(23150, "PE", hist, chain_leg=leg)
        self.assertEqual(len(df), len(hist))                  # same time frame
        self.assertEqual(_premium_source(df), "groww-anchored")
        self.assertAlmostEqual(df["close"].iloc[0], 115.8, places=2)

    def test_pe_rises_when_index_falls(self):
        hist = _index_grid([23300.0, 23250.0, 23200.0])
        leg  = {"ltp": 100.0, "delta": -0.40, "theta": 0.0}
        df   = _model_premium_series(23150, "PE", hist, chain_leg=leg)
        self.assertGreater(df["close"].iloc[-1], df["close"].iloc[0])

    def test_ce_falls_when_index_falls(self):
        hist = _index_grid([23300.0, 23250.0, 23200.0])
        leg  = {"ltp": 100.0, "delta": 0.40, "theta": 0.0}
        df   = _model_premium_series(23300, "CE", hist, chain_leg=leg)
        self.assertLess(df["close"].iloc[-1], df["close"].iloc[0])

    def test_never_below_intrinsic(self):
        hist = _index_grid([23260.0, 23100.0, 22800.0])
        leg  = {"ltp": 5.0, "delta": -0.05, "theta": 0.0}
        df   = _model_premium_series(23150, "PE", hist, chain_leg=leg)
        for spot, p in zip(hist["close"], df["close"]):
            self.assertGreaterEqual(p, max(0.0, 23150.0 - spot) - 1e-9)

    def test_theta_decays_series(self):
        hist = _index_grid([23250.0] * 20)
        leg  = {"ltp": 120.0, "delta": 0.0, "theta": -12.0}
        df   = _model_premium_series(23150, "PE", hist, chain_leg=leg)
        self.assertLess(df["close"].iloc[-1], df["close"].iloc[0])

    def test_falls_back_to_model_without_quote(self):
        hist = _index_grid([23250.0] * 5)
        self.assertEqual(_premium_source(_model_premium_series(23150, "PE", hist)),
                         "model")
        self.assertEqual(
            _premium_source(_model_premium_series(23150, "PE", hist,
                                                 chain_leg={"ltp": 0.0})),
            "model")

    def test_anchored_helper_is_pure(self):
        hist = _index_grid([23250.0] * 4)
        df   = _anchored_premium_series(23150, "PE", hist,
                                        {"ltp": 100.0, "delta": -0.5, "theta": 0.0})
        self.assertEqual(list(df.columns)[:6],
                         ["timestamp", "open", "high", "low", "close", "volume"])


if __name__ == "__main__":
    unittest.main()
