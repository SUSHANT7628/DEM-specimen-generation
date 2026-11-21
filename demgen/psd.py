"""Particle size distribution (PSD): read a sieve curve and draw particle diameters from it.

A sieve curve gives cumulative percent passing by *mass*. In a 2-D disc model the mass of a
particle is proportional to d^2 (area), so a size class with mass fraction w_k holds a number
fraction proportional to w_k / d_k^2. Sampling by mass fraction directly would put far too few
fine particles in the specimen.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SieveCurve:
    size_mm: np.ndarray        # increasing sieve sizes
    passing: np.ndarray        # cumulative percent passing (0-100) at each size

    @classmethod
    def from_csv(cls, path: str | Path) -> "SieveCurve":
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        size = np.array([float(r["sieve_size_mm"]) for r in rows])
        pas = np.array([float(r["percent_passing"]) for r in rows])
        order = np.argsort(size)
        size, pas = size[order], pas[order]
        if np.any(np.diff(pas) < 0):
            raise ValueError("percent_passing must not decrease with sieve size")
        return cls(size, pas)

    def passing_at(self, d_mm: np.ndarray) -> np.ndarray:
        """Percent passing at diameter d, interpolated linearly in log(d) (how sieve curves are drawn)."""
        return np.interp(np.log(d_mm), np.log(self.size_mm), self.passing)

    def d_at(self, percent: float) -> float:
        """Diameter at a given percent passing (e.g. 50 -> D50)."""
        return float(np.exp(np.interp(percent, self.passing, np.log(self.size_mm))))


def number_cdf(curve: SieveCurve, d_min_mm: float, d_max_mm: float, n_bins: int = 400):
    """Number-based CDF of diameters implied by the (mass-based) sieve curve on [d_min, d_max]."""
    edges = np.geomspace(d_min_mm, d_max_mm, n_bins + 1)
    cum = curve.passing_at(edges)
    mass_frac = np.clip(np.diff(cum), 0.0, None)
    if mass_frac.sum() <= 0:
        raise ValueError("the sieve curve holds no mass between d_min and d_max")
    centres = np.sqrt(edges[:-1] * edges[1:])
    number_w = mass_frac / centres**2
    number_w /= number_w.sum()
    return edges, np.concatenate([[0.0], np.cumsum(number_w)])


def sample_diameters(curve: SieveCurve, n: int, d_min_mm: float, d_max_mm: float,
                     rng: np.random.Generator, method: str = "stratified") -> np.ndarray:
    """Draw n diameters (mm) whose area-weighted grading follows `curve` between d_min and d_max.

    The curve is truncated to [d_min, d_max] and renormalised, converted from mass fractions to
    number fractions (w / d^2), and inverted.

    method="stratified" (default) takes n evenly spaced quantiles of the number CDF, including both
    ends, so even a few hundred discs reproduce the curve closely and span the full d_min..d_max.
    method="random" draws independent samples (the coarse tail is then often missing in small seeds).
    """
    edges, cdf = number_cdf(curve, d_min_mm, d_max_mm)
    if method == "stratified":
        q = np.linspace(0.0, 1.0, n)
    elif method == "random":
        q = rng.random(n)
    else:
        raise ValueError(method)
    # invert the CDF, interpolating in log(d)
    d = np.exp(np.interp(q, cdf, np.log(edges)))
    return rng.permutation(d)


def grading_of(diameters_mm: np.ndarray, sizes_mm: np.ndarray) -> np.ndarray:
    """Area-weighted (2-D mass-weighted) percent passing of a sample at the given sizes."""
    a = diameters_mm**2
    return np.array([100.0 * a[diameters_mm <= s].sum() / a.sum() for s in sizes_mm])
