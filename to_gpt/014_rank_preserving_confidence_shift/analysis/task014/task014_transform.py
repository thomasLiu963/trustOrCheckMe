"""Rank-preserving logit-shift of q1. Frozen before any Stage-3 calls."""

from __future__ import annotations

import math

from .study1_prompts import format_displayed_probability

DELTAS = (-1.5, -0.75, 0.0, 0.75, 1.5)


def logit(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"logit undefined at endpoint q1={p}")
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def transform_q(q1: float, delta: float) -> float:
    """Odds-space shift. delta=0 returns the original q1 float (no logit round-trip)."""
    q = float(q1)
    if q == 0.0 or q == 1.0:
        return q
    if delta == 0.0:
        return q
    return float(sigmoid(logit(q) + float(delta)))


def render_q(q: float) -> str:
    return format_displayed_probability(q)


def rendered_q(q1: float, delta: float) -> tuple[float, str]:
    value = transform_q(q1, delta)
    return value, render_q(value)
