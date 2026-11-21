"""Fast checks (about 20 s): PSD sampling, flip-tiling equilibrium, independent validator."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demgen.dem import Material  # noqa: E402
from demgen.generate import generate_seed  # noqa: E402
from demgen.psd import SieveCurve, grading_of, sample_diameters  # noqa: E402
from demgen.tiling import flip_tile  # noqa: E402
from demgen.validate import check_equilibrium, write_specimen  # noqa: E402

PSD = ROOT / "psd" / "astm_c778_graded_sand.csv"


def _small_seed(n=120):
    rng = np.random.default_rng(3)
    d = sample_diameters(SieveCurve.from_csv(PSD), n, 0.15, 0.85, rng)
    seed, _ = generate_seed(d * 1e-3 / 2, 100e3, Material(viscous_damping=2e-4), rng, verbose=False)
    return seed


def test_psd_sample_spans_range_and_matches_curve():
    curve = SieveCurve.from_csv(PSD)
    d = sample_diameters(curve, 625, 0.15, 0.85, np.random.default_rng(0))
    assert abs(d.min() - 0.15) < 1e-9 and abs(d.max() - 0.85) < 1e-9
    sieves = np.array([0.3, 0.425, 0.6])
    lo, hi = curve.passing_at(np.array([0.15]))[0], curve.passing_at(np.array([0.85]))[0]
    target = (curve.passing_at(sieves) - lo) / (hi - lo) * 100
    assert np.max(np.abs(grading_of(d, sieves) - target)) < 2.0


def test_flip_tiling_preserves_equilibrium(tmp_path):
    seed = _small_seed()
    u_seed = seed.unbalanced_force_ratio()["ufr_mean"]
    assert u_seed <= 1e-4
    big = flip_tile(seed, 3, 2)
    big.compute_forces(dt=0.0)
    assert big.N == 6 * seed.N
    u_big = big.unbalanced_force_ratio()["ufr_mean"]
    assert u_big <= 1e-4
    # stresses on the outer walls equal the seed's
    assert np.allclose(big.wall_stress()[[0, 1]], seed.wall_stress()[[0, 1]], rtol=1e-9)
    # nothing moves when the tiled assembly is cycled
    x0 = big.x.copy()
    big.step(300)
    assert np.max(np.abs(big.x - x0)) < 1e-6 * np.median(2 * big.r)

    write_specimen(big, tmp_path)
    indep = check_equilibrium(tmp_path)
    assert indep["contact_set_matches_geometry"]
    assert indep["ufr_mean"] <= 1e-4


def test_translation_tiling_would_break_equilibrium():
    """Control: copying the seed without mirroring puts non-matching faces together."""
    from demgen.dem import Assembly
    seed = _small_seed()
    xl, xr, yb, yt = seed.box
    x2 = seed.x.copy(); x2[:, 0] += xr - xl
    a = Assembly(np.vstack([seed.x, x2]), np.concatenate([seed.r, seed.r]),
                 np.array([xl, 2 * xr - xl, yb, yt]), seed.mat)
    a.compute_forces(dt=0.0)
    assert a.unbalanced_force_ratio()["ufr_mean"] > 1e-2
