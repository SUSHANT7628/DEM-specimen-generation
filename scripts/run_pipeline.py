"""End-to-end run: PSD -> 625-disc seed in equilibrium -> 4x4 flip tiling -> 10,000-disc specimen
-> free relaxation -> independent equilibrium check -> results/ (metrics.json, figures, CSVs).

    python scripts/run_pipeline.py              # defaults reproduce results/ (about 2-3 min on a laptop)
    python scripts/run_pipeline.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from demgen.dem import Material  # noqa: E402
from demgen.generate import generate_seed  # noqa: E402
from demgen.plotting import plot_force_chains, plot_psd, plot_ufr_history  # noqa: E402
from demgen.psd import SieveCurve, grading_of, sample_diameters  # noqa: E402
from demgen.tiling import flip_tile  # noqa: E402
from demgen.validate import check_equilibrium, write_specimen  # noqa: E402


def boundary_contact_check(seed, big, nx: int) -> dict:
    """Compare every contact that crosses a tile boundary with the wall force it replaced in the seed."""
    Ns = seed.N
    c = big.contacts
    ti, tj = c.i // Ns, c.j // Ns
    cross = ti != tj
    xl, xr, yb, yt = seed.box
    kw = seed.mat.kw
    expected = []
    for i, j, fn in zip(c.i[cross], c.j[cross], c.fn[cross]):
        a_i, b_i = (i // Ns) % nx, (i // Ns) // nx
        a_j, b_j = (j // Ns) % nx, (j // Ns) // nx
        s = i % Ns
        px, py = seed.x[s]
        rr = seed.r[s]
        if a_i != a_j:
            toward_right = a_j > a_i
            if a_i % 2 == 1:              # tile i is mirrored left-right
                toward_right = not toward_right
            ov = rr - (xr - px) if toward_right else rr - (px - xl)
        else:
            toward_top = b_j > b_i
            if b_i % 2 == 1:
                toward_top = not toward_top
            ov = rr - (yt - py) if toward_top else rr - (py - yb)
        expected.append(kw * ov)
    expected = np.array(expected)
    got = c.fn[cross]
    return {
        "contacts_crossing_tile_boundaries": int(cross.sum()),
        "boundary_force_max_rel_diff_vs_seed_wall": float(np.max(np.abs(got - expected) / expected)),
        "boundary_tangential_force_max": float(np.max(np.abs(c.ft[cross]))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--psd", default=str(ROOT / "psd" / "astm_c778_graded_sand.csv"))
    ap.add_argument("--d-min", type=float, default=0.15, help="smallest diameter, mm")
    ap.add_argument("--d-max", type=float, default=0.85, help="largest diameter, mm")
    ap.add_argument("--n-seed", type=int, default=625)
    ap.add_argument("--tiles", type=int, nargs=2, default=[4, 4], metavar=("NX", "NY"))
    ap.add_argument("--sigma0", type=float, default=100e3, help="isotropic confining stress, Pa")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--kn", type=float, default=1e8)
    ap.add_argument("--ufr-target", type=float, default=1e-4)
    ap.add_argument("--relax-steps", type=int, default=6000)
    ap.add_argument("--rng-seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)
    t_start = time.time()

    # 1. particle sizes from the sieve curve
    curve = SieveCurve.from_csv(args.psd)
    d_mm = sample_diameters(curve, args.n_seed, args.d_min, args.d_max, rng)
    plot_psd(curve, d_mm, args.d_min, args.d_max, out / "figures" / "psd_fit.png")
    sieves = curve.size_mm[(curve.size_mm >= args.d_min) & (curve.size_mm <= args.d_max)]
    trunc = (curve.passing_at(sieves) - curve.passing_at(np.array([args.d_min]))) / (
        curve.passing_at(np.array([args.d_max])) - curve.passing_at(np.array([args.d_min]))) * 100

    # 2. force-equilibrated seed
    mat = Material(kn=args.kn, mu=args.mu, viscous_damping=2e-4)
    t0 = time.time()
    seed, seed_hist = generate_seed(d_mm * 1e-3 / 2, args.sigma0, mat, rng, ufr_target=args.ufr_target)
    seed_time = time.time() - t0
    seed_ufr = seed.unbalanced_force_ratio()
    write_specimen(seed, out / "seed", {"sigma0": args.sigma0})

    # 3. flip tiling
    nx, ny = args.tiles
    big = flip_tile(seed, nx, ny)
    big.compute_forces(dt=0.0)
    tiled_ufr = big.unbalanced_force_ratio()
    bcheck = boundary_contact_check(seed, big, nx)
    # control: the same 16 copies placed by plain translation (no mirroring)
    from demgen.dem import Assembly
    xl, xr, yb, yt = seed.box
    Lx, Ly = xr - xl, yt - yb
    xt = np.vstack([seed.x + np.array([a * Lx, b * Ly]) for b in range(ny) for a in range(nx)])
    ctrl = Assembly(xt, np.tile(seed.r, nx * ny), np.array([xl, xl + nx * Lx, yb, yb + ny * Ly]), seed.mat)
    ctrl.compute_forces(dt=0.0)
    ctrl_ufr = ctrl.unbalanced_force_ratio()
    f0 = {int(k): f for k, f in zip(big.contacts.key, np.hypot(big.contacts.fn, big.contacts.ft))}
    x0 = big.x.copy()

    # 4. free relaxation of the full specimen (walls fixed), checking it stays in equilibrium
    big_hist = [{"step": 0, **tiled_ufr}]
    t0 = time.time()
    block = 500
    for s in range(block, args.relax_steps + 1, block):
        big.step(block)
        big.compute_forces(dt=0.0)
        big_hist.append({"step": s, **big.unbalanced_force_ratio()})
        print(f"[tiled 10k] step {s:>5}  ufr={big_hist[-1]['ufr_mean']:.2e}")
    relax_time = time.time() - t0
    big.v[:] = 0.0
    big.w[:] = 0.0
    big.compute_forces(dt=0.0)
    final_ufr = big.unbalanced_force_ratio()
    d50 = float(np.median(2 * big.r))
    disp = np.hypot(*(big.x - x0).T)
    keys = big.contacts.key
    f1 = np.hypot(big.contacts.fn, big.contacts.ft)
    common = np.array([k in f0 for k in keys])
    df = np.abs(f1[common] - np.array([f0[int(k)] for k in keys[common]]))
    write_specimen(big, out / "specimen_10k", {"sigma0": args.sigma0, "tiles": [nx, ny], "seed_n": seed.N})

    # 5. independent re-check from the saved files
    indep_seed = check_equilibrium(out / "seed")
    indep_big = check_equilibrium(out / "specimen_10k")

    # 6. figures
    plot_force_chains(seed, out / "figures" / "seed_force_chains.png",
                      f"Seed: {seed.N} discs, sigma0 = {args.sigma0/1e3:.0f} kPa, UFR = {seed_ufr['ufr_mean']:.1e}")
    plot_force_chains(big, out / "figures" / "specimen_10k_force_chains.png",
                      f"Flip-tiled specimen: {big.N:,} discs ({nx}x{ny} tiles, dashed), "
                      f"UFR = {final_ufr['ufr_mean']:.1e}", tiles=(nx, ny), seed_size=big.seed_size,
                      show_discs=False)
    plot_ufr_history(seed_hist, big_hist, args.ufr_target, out / "figures" / "equilibrium_history.png")

    sig = big.wall_stress()
    metrics = {
        "psd": {
            "curve": Path(args.psd).name,
            "d_min_mm": float(d_mm.min()), "d_max_mm": float(d_mm.max()),
            "size_span_dmax_over_dmin": float(d_mm.max() / d_mm.min()),
            "d50_number_mm": float(np.median(d_mm)),
            "d50_mass_mm": float(np.interp(50, grading_of(d_mm, np.sort(d_mm)), np.sort(d_mm))),
            "max_abs_grading_error_pct_points_at_sieves": float(np.max(np.abs(grading_of(d_mm, sieves) - trunc))),
        },
        "seed": {
            "n_particles": seed.N, "timesteps": seed_hist[-1]["step"], "wall_clock_s": round(seed_time, 1),
            "porosity": seed.porosity(), "void_ratio": seed.porosity() / (1 - seed.porosity()),
            **seed.coordination(), "wall_stress_kPa": (seed.wall_stress() / 1e3).round(3).tolist(),
            "ufr_mean": seed_ufr["ufr_mean"], "ufr_max": seed_ufr["ufr_max"],
        },
        "tiled_specimen": {
            "n_particles": big.N, "tiles": [nx, ny],
            "ufr_mean_immediately_after_tiling": tiled_ufr["ufr_mean"],
            "ufr_mean_after_relaxation": final_ufr["ufr_mean"],
            "ufr_max_after_relaxation": final_ufr["ufr_max"],
            "relaxation_timesteps": args.relax_steps, "relaxation_wall_clock_s": round(relax_time, 1),
            "max_displacement_over_d50": float(disp.max() / d50),
            "mean_displacement_over_d50": float(disp.mean() / d50),
            "contact_force_change_mean_over_mean_force": float(df.mean() / f1.mean()),
            "contacts_kept_through_relaxation": float(common.mean()),
            "porosity": big.porosity(), **big.coordination(),
            "wall_stress_kPa": (sig / 1e3).round(3).tolist(),
            **bcheck,
        },
        "control_translation_tiling": {"ufr_mean": ctrl_ufr["ufr_mean"], "ufr_max": ctrl_ufr["ufr_max"]},
        "independent_check": {"seed": indep_seed, "specimen_10k": indep_big},
        "model": {"dimension": "2-D discs", "kn": args.kn, "kt_over_kn": mat.kt_ratio, "kw": mat.kw,
                  "mu": args.mu, "rho": mat.rho, "local_damping": mat.local_damping,
                  "dt_s": seed.dt, "dt_critical_s": seed.dt_crit, "dt_over_dt_critical": seed.dt / seed.dt_crit,
                  "sigma0_kPa": args.sigma0 / 1e3, "ufr_target": args.ufr_target},
        "total_wall_clock_s": round(time.time() - t_start, 1),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
