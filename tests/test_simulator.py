"""Tests for paper_exec.simulator -- fill simulation with costs and slippage."""
from __future__ import annotations

import pytest

from paper_exec.simulator import (
    COMMISSION_PER_CONTRACT_USD,
    SLIPPAGE_PER_LEG_USD,
    LegFill,
    LegOrder,
    Quote,
    TradeFill,
    compute_total_costs,
    simulate_leg_fill,
    simulate_trade_fill,
)


# ---------------------------------------------------------------------------
# Quote properties
# ---------------------------------------------------------------------------


class TestQuote:
    def test_mid(self):
        assert Quote(bid=100.0, ask=102.0).mid == pytest.approx(101.0)

    def test_spread(self):
        assert Quote(bid=100.0, ask=102.0).spread == pytest.approx(2.0)

    def test_half_spread(self):
        assert Quote(bid=100.0, ask=102.0).half_spread == pytest.approx(1.0)

    def test_zero_width_spread(self):
        q = Quote(bid=1.0, ask=1.0)
        assert q.spread == pytest.approx(0.0)
        assert q.half_spread == pytest.approx(0.0)
        assert q.mid == pytest.approx(1.0)

    def test_option_quote(self):
        q = Quote(bid=1.00, ask=1.10)
        assert q.mid == pytest.approx(1.05)
        assert q.spread == pytest.approx(0.10)
        assert q.half_spread == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Single-leg fills
# ---------------------------------------------------------------------------


class TestSingleLegFill:
    def test_long_call_fill_at_ask_plus_slippage(self):
        """Long call: fill ~= ask + slippage."""
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        # mid=1.05, half_spread=0.05, slip=0.01 -> fill = 1.05 + 0.05 + 0.01 = 1.11
        assert fill.fill_price == pytest.approx(1.11)
        # And that equals ask + slippage when half_spread*1 == quote.half_spread
        assert fill.fill_price == pytest.approx(quote.ask + SLIPPAGE_PER_LEG_USD)

    def test_short_call_fill_at_bid_minus_slippage(self):
        """Short call: fill ~= bid - slippage."""
        order = LegOrder(
            instrument_type='call',
            direction='short',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        # mid=1.05, half_spread=0.05 -> fill = 1.05 - 0.05 - 0.01 = 0.99
        assert fill.fill_price == pytest.approx(0.99)
        assert fill.fill_price == pytest.approx(quote.bid - SLIPPAGE_PER_LEG_USD)

    def test_long_put_fill(self):
        order = LegOrder(
            instrument_type='put',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=2.00,
        )
        quote = Quote(bid=1.95, ask=2.05)
        fill = simulate_leg_fill(order, quote)
        # mid=2.0, half_spread=0.05 -> fill = 2.0 + 0.05 + 0.01 = 2.06
        assert fill.fill_price == pytest.approx(2.06)

    def test_long_stock_fill(self):
        """Long stock: bid=100, ask=100.05 → fill ~= 100.06."""
        order = LegOrder(
            instrument_type='stock',
            direction='long',
            quantity=100,
            multiplier=1,
            intended_mid_price=100.025,
        )
        quote = Quote(bid=100.00, ask=100.05)
        fill = simulate_leg_fill(order, quote)
        # mid=100.025, half_spread=0.025 -> fill = 100.025 + 0.025 + 0.01 = 100.06
        assert fill.fill_price == pytest.approx(100.06)

    def test_commission_one_contract(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        assert fill.commission_usd == pytest.approx(0.65)

    def test_commission_four_contracts(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=4,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        assert fill.commission_usd == pytest.approx(4 * 0.65)
        assert fill.commission_usd == pytest.approx(2.60)

    def test_spread_cost_usd_calculation(self):
        """spread_cost_usd = half_spread × quantity × multiplier."""
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)  # half_spread = 0.05
        fill = simulate_leg_fill(order, quote)
        # 0.05 * 1 * 100 = 5.0
        assert fill.spread_cost_usd == pytest.approx(5.0)

    def test_spread_cost_scales_with_quantity(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=3,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)  # half_spread = 0.05
        fill = simulate_leg_fill(order, quote)
        # 0.05 * 3 * 100 = 15.0
        assert fill.spread_cost_usd == pytest.approx(15.0)

    def test_slippage_usd_calculation(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=2,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        # default slippage 0.01 * 2 * 100 = 2.0
        assert fill.slippage_usd == pytest.approx(2.0)

    def test_intended_mid_price_preserved(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.07,
        )
        quote = Quote(bid=1.00, ask=1.10)
        fill = simulate_leg_fill(order, quote)
        assert fill.intended_mid_price == pytest.approx(1.07)


# ---------------------------------------------------------------------------
# Cost stress multiplier
# ---------------------------------------------------------------------------


class TestCostStress:
    @pytest.fixture
    def base_setup(self):
        order = LegOrder(
            instrument_type='call',
            direction='long',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)  # half_spread=0.05, mid=1.05
        return order, quote

    def test_stress_1_0_matches_base(self, base_setup):
        order, quote = base_setup
        base = simulate_leg_fill(order, quote, cost_stress_multiplier=1.0)
        # Expected: fill = 1.05 + 0.05 + 0.01 = 1.11
        assert base.fill_price == pytest.approx(1.11)
        assert base.spread_cost_usd == pytest.approx(5.0)

    def test_stress_1_5_widens_spread(self, base_setup):
        order, quote = base_setup
        stressed = simulate_leg_fill(order, quote, cost_stress_multiplier=1.5)
        # effective half_spread = 0.05 * 1.5 = 0.075
        # fill = 1.05 + 0.075 + 0.01 = 1.135
        assert stressed.fill_price == pytest.approx(1.135)
        # spread cost = 0.075 * 1 * 100 = 7.5
        assert stressed.spread_cost_usd == pytest.approx(7.5)

    def test_stress_2_0_doubles_half_spread(self, base_setup):
        order, quote = base_setup
        stressed = simulate_leg_fill(order, quote, cost_stress_multiplier=2.0)
        # effective half_spread = 0.05 * 2.0 = 0.10
        # fill = 1.05 + 0.10 + 0.01 = 1.16
        assert stressed.fill_price == pytest.approx(1.16)
        assert stressed.spread_cost_usd == pytest.approx(10.0)

    def test_stress_pushes_long_fill_higher(self, base_setup):
        order, quote = base_setup
        base = simulate_leg_fill(order, quote, cost_stress_multiplier=1.0)
        stressed = simulate_leg_fill(order, quote, cost_stress_multiplier=1.5)
        assert stressed.fill_price > base.fill_price

    def test_stress_pushes_short_fill_lower(self):
        order = LegOrder(
            instrument_type='call',
            direction='short',
            quantity=1,
            multiplier=100,
            intended_mid_price=1.05,
        )
        quote = Quote(bid=1.00, ask=1.10)
        base = simulate_leg_fill(order, quote, cost_stress_multiplier=1.0)
        stressed = simulate_leg_fill(order, quote, cost_stress_multiplier=1.5)
        # short receives less under stress
        assert stressed.fill_price < base.fill_price

    def test_stress_zero_means_fill_at_mid(self, base_setup):
        """cost_stress_multiplier=0 zeros the spread cost; fill still has slippage."""
        order, quote = base_setup
        fill = simulate_leg_fill(order, quote, cost_stress_multiplier=0.0)
        assert fill.fill_price == pytest.approx(1.05 + 0.01)
        assert fill.spread_cost_usd == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multi-leg fills
# ---------------------------------------------------------------------------


class TestMultiLegTrade:
    def test_iron_condor_credit(self):
        """Iron condor: short call + long farther call + short put + long farther put.
        Credits (shorts) should dominate; net_credit_or_debit should be POSITIVE."""
        # Short call at 405 (bid 2.00, ask 2.10)
        sc = LegOrder('call', 'short', 1, 100, 2.05)
        sc_q = Quote(bid=2.00, ask=2.10)
        # Long farther call at 410 (bid 0.90, ask 1.00) -- cheaper wing
        lc = LegOrder('call', 'long', 1, 100, 0.95)
        lc_q = Quote(bid=0.90, ask=1.00)
        # Short put at 395 (bid 2.00, ask 2.10)
        sp = LegOrder('put', 'short', 1, 100, 2.05)
        sp_q = Quote(bid=2.00, ask=2.10)
        # Long farther put at 390 (bid 0.90, ask 1.00)
        lp = LegOrder('put', 'long', 1, 100, 0.95)
        lp_q = Quote(bid=0.90, ask=1.00)

        fill = simulate_trade_fill([sc, lc, sp, lp], [sc_q, lc_q, sp_q, lp_q])

        # 4 legs of 1 contract each
        assert fill.total_commission_usd == pytest.approx(4 * 0.65)
        assert fill.total_commission_usd == pytest.approx(2.60)
        # 4 leg fills returned
        assert len(fill.leg_fills) == 4
        # Iron condor is a credit spread -> positive net cash
        assert fill.net_credit_or_debit_usd > 0

    def test_long_straddle_debit(self):
        """Long straddle: buy ATM call + buy ATM put. Net should be a DEBIT (negative)."""
        call = LegOrder('call', 'long', 1, 100, 5.05)
        call_q = Quote(bid=5.00, ask=5.10)
        put = LegOrder('put', 'long', 1, 100, 5.05)
        put_q = Quote(bid=5.00, ask=5.10)

        fill = simulate_trade_fill([call, put], [call_q, put_q])

        assert fill.total_commission_usd == pytest.approx(2 * 0.65)
        # Long straddle: pay both legs -> negative cash flow
        assert fill.net_credit_or_debit_usd < 0
        # Magnitude check: roughly 2 * (5.10 + 0.01) * 100 = ~1022 paid + commissions/slip
        # net = -2 * fill_price * 100 - commissions - slippage
        expected_paid = 2 * (5.05 + 0.05 + 0.01) * 100  # ~1022
        expected_net = (
            -expected_paid
            - 2 * 0.65          # commissions
            - 2 * 0.01 * 100    # slippage subtracted explicitly
        )
        assert fill.net_credit_or_debit_usd == pytest.approx(expected_net)

    def test_iron_condor_total_spread_cost(self):
        """Spread cost summed across 4 legs (each half_spread * 100)."""
        sc = LegOrder('call', 'short', 1, 100, 2.05)
        sc_q = Quote(bid=2.00, ask=2.10)  # half_spread 0.05 -> $5
        lc = LegOrder('call', 'long', 1, 100, 0.95)
        lc_q = Quote(bid=0.90, ask=1.00)  # half_spread 0.05 -> $5
        sp = LegOrder('put', 'short', 1, 100, 2.05)
        sp_q = Quote(bid=2.00, ask=2.10)  # half_spread 0.05 -> $5
        lp = LegOrder('put', 'long', 1, 100, 0.95)
        lp_q = Quote(bid=0.90, ask=1.00)  # half_spread 0.05 -> $5

        fill = simulate_trade_fill([sc, lc, sp, lp], [sc_q, lc_q, sp_q, lp_q])
        assert fill.total_spread_cost_usd == pytest.approx(20.0)
        assert fill.total_slippage_usd == pytest.approx(4.0)  # 4 legs * 0.01 * 100

    def test_iron_condor_under_stress(self):
        sc = LegOrder('call', 'short', 1, 100, 2.05)
        sc_q = Quote(bid=2.00, ask=2.10)
        lc = LegOrder('call', 'long', 1, 100, 0.95)
        lc_q = Quote(bid=0.90, ask=1.00)
        sp = LegOrder('put', 'short', 1, 100, 2.05)
        sp_q = Quote(bid=2.00, ask=2.10)
        lp = LegOrder('put', 'long', 1, 100, 0.95)
        lp_q = Quote(bid=0.90, ask=1.00)

        base = simulate_trade_fill([sc, lc, sp, lp], [sc_q, lc_q, sp_q, lp_q])
        stressed_50 = simulate_trade_fill(
            [sc, lc, sp, lp], [sc_q, lc_q, sp_q, lp_q], cost_stress_multiplier=1.5
        )
        # +50% costs: spread costs 1.5x; net credit shrinks
        assert stressed_50.total_spread_cost_usd == pytest.approx(base.total_spread_cost_usd * 1.5)
        assert stressed_50.net_credit_or_debit_usd < base.net_credit_or_debit_usd


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_orders_returns_zero_trade(self):
        fill = simulate_trade_fill([], [])
        assert isinstance(fill, TradeFill)
        assert fill.leg_fills == []
        assert fill.total_commission_usd == 0.0
        assert fill.total_spread_cost_usd == 0.0
        assert fill.total_slippage_usd == 0.0
        assert fill.net_credit_or_debit_usd == 0.0

    def test_mismatched_lengths_raises(self):
        order = LegOrder('call', 'long', 1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        with pytest.raises(ValueError, match="same length"):
            simulate_trade_fill([order, order], [quote])

    def test_zero_quantity_raises(self):
        order = LegOrder('call', 'long', 0, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        with pytest.raises(ValueError, match="quantity must be positive"):
            simulate_leg_fill(order, quote)

    def test_negative_quantity_raises(self):
        order = LegOrder('call', 'long', -1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        with pytest.raises(ValueError, match="quantity must be positive"):
            simulate_leg_fill(order, quote)

    def test_negative_cost_stress_raises(self):
        order = LegOrder('call', 'long', 1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        with pytest.raises(ValueError, match="cost_stress_multiplier"):
            simulate_leg_fill(order, quote, cost_stress_multiplier=-0.1)

    def test_negative_slippage_raises(self):
        order = LegOrder('call', 'long', 1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        with pytest.raises(ValueError, match="slippage_per_share"):
            simulate_leg_fill(order, quote, slippage_per_share=-0.01)

    def test_inverted_quote_raises(self):
        order = LegOrder('call', 'long', 1, 100, 1.05)
        with pytest.raises(ValueError, match="ask"):
            simulate_leg_fill(order, Quote(bid=1.10, ask=1.00))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_args_same_result_default(self):
        """No rng_seed: fully deterministic."""
        order = LegOrder('call', 'long', 2, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        a = simulate_leg_fill(order, quote)
        b = simulate_leg_fill(order, quote)
        assert a.fill_price == b.fill_price
        assert a.commission_usd == b.commission_usd
        assert a.spread_cost_usd == b.spread_cost_usd
        assert a.slippage_usd == b.slippage_usd

    def test_same_seed_same_result(self):
        order = LegOrder('call', 'long', 1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        a = simulate_leg_fill(order, quote, rng_seed=42)
        b = simulate_leg_fill(order, quote, rng_seed=42)
        assert a.fill_price == b.fill_price
        assert a.slippage_usd == b.slippage_usd

    def test_different_seeds_may_differ(self):
        """Soft check: with std=0.25*mean, two seeds should typically yield
        distinct slippage. We check at least that one yields a non-default
        slippage_usd to confirm the RNG path runs."""
        order = LegOrder('call', 'long', 1, 100, 1.05)
        quote = Quote(bid=1.00, ask=1.10)
        results = [
            simulate_leg_fill(order, quote, rng_seed=seed).slippage_usd
            for seed in (1, 2, 3, 4, 5)
        ]
        assert any(r != results[0] for r in results), (
            "expected RNG seeds to produce at least one different slippage"
        )

    def test_trade_fill_deterministic(self):
        orders = [
            LegOrder('call', 'short', 1, 100, 2.05),
            LegOrder('call', 'long', 1, 100, 0.95),
        ]
        quotes = [Quote(bid=2.00, ask=2.10), Quote(bid=0.90, ask=1.00)]
        a = simulate_trade_fill(orders, quotes)
        b = simulate_trade_fill(orders, quotes)
        assert a.net_credit_or_debit_usd == b.net_credit_or_debit_usd
        assert a.total_commission_usd == b.total_commission_usd


# ---------------------------------------------------------------------------
# compute_total_costs helper
# ---------------------------------------------------------------------------


class TestComputeTotalCosts:
    def test_keys_present(self):
        orders = [LegOrder('call', 'long', 1, 100, 1.05)]
        quotes = [Quote(bid=1.00, ask=1.10)]
        fill = simulate_trade_fill(orders, quotes)
        costs = compute_total_costs(fill)
        assert set(costs.keys()) == {'commission', 'spread', 'slippage', 'total'}

    def test_total_is_sum(self):
        orders = [
            LegOrder('call', 'short', 1, 100, 2.05),
            LegOrder('call', 'long', 1, 100, 0.95),
        ]
        quotes = [Quote(bid=2.00, ask=2.10), Quote(bid=0.90, ask=1.00)]
        fill = simulate_trade_fill(orders, quotes)
        costs = compute_total_costs(fill)
        assert costs['total'] == pytest.approx(
            costs['commission'] + costs['spread'] + costs['slippage']
        )

    def test_empty_trade_zero_costs(self):
        fill = simulate_trade_fill([], [])
        costs = compute_total_costs(fill)
        assert costs == {
            'commission': 0.0,
            'spread': 0.0,
            'slippage': 0.0,
            'total': 0.0,
        }

    def test_costs_match_fill_attributes(self):
        orders = [LegOrder('put', 'long', 3, 100, 2.05)]
        quotes = [Quote(bid=2.00, ask=2.10)]
        fill = simulate_trade_fill(orders, quotes)
        costs = compute_total_costs(fill)
        assert costs['commission'] == pytest.approx(fill.total_commission_usd)
        assert costs['spread'] == pytest.approx(fill.total_spread_cost_usd)
        assert costs['slippage'] == pytest.approx(fill.total_slippage_usd)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_commission_constant():
    """Per capital_policy.md: $0.65 per contract per leg (IBKR Pro)."""
    assert COMMISSION_PER_CONTRACT_USD == 0.65


def test_default_slippage_constant():
    assert SLIPPAGE_PER_LEG_USD == 0.01
