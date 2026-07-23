"""Core bond valuation and risk engine — pure, stateless functions."""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq


def _cash_flows(face, coupon_rate, freq, years):
    """Periodic cash flows: coupons, plus face repaid at maturity."""
    n = int(round(freq * years))
    coupon = face * coupon_rate / freq
    cfs = np.full(n, coupon, dtype=float)
    cfs[-1] += face
    return cfs


def price_from_yield(face, coupon_rate, freq, years, ytm):
    """Price given an annual yield to maturity."""
    cfs = _cash_flows(face, coupon_rate, freq, years)
    t = np.arange(1, len(cfs) + 1)
    disc = (1 + ytm / freq) ** t
    return float(np.sum(cfs / disc))


def yield_from_price(face, coupon_rate, freq, years, price):
    """Solve for annual YTM given a market price."""
    f = lambda y: price_from_yield(face, coupon_rate, freq, years, y) - price
    return brentq(f, -0.99, 2.0, xtol=1e-8)


def macaulay_duration(face, coupon_rate, freq, years, ytm):
    cfs = _cash_flows(face, coupon_rate, freq, years)
    t = np.arange(1, len(cfs) + 1)
    pv = cfs / (1 + ytm / freq) ** t
    return float(np.sum((t / freq) * pv) / pv.sum())


def modified_duration(face, coupon_rate, freq, years, ytm):
    mac = macaulay_duration(face, coupon_rate, freq, years, ytm)
    return mac / (1 + ytm / freq)


def convexity(face, coupon_rate, freq, years, ytm):
    cfs = _cash_flows(face, coupon_rate, freq, years)
    t = np.arange(1, len(cfs) + 1)
    price = price_from_yield(face, coupon_rate, freq, years, ytm)
    c = np.sum(t * (t + 1) * cfs / (1 + ytm / freq) ** (t + 2))
    return float(c / (price * freq ** 2))


def estimated_price_change(price, mod_dur, conv, dy):
    """Duration + convexity approx of price change for yield shift dy."""
    return price * (-mod_dur * dy + 0.5 * conv * dy ** 2)
