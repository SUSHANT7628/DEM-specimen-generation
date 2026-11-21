"""Seed generation: random placement -> radius growth -> servo-controlled isotropic compression ->
equilibration to a target unbalanced force ratio."""
from __future__ import annotations

import time

import numpy as np

from .dem import Assembly, Material


def random_placement(r: np.ndarray, box: np.ndarray, rng: np.random.Generator,
                     max_tries: int = 20000) -> np.ndarray:
    """Random sequential addition without overlap, largest discs first."""
    xl, xr, yb, yt = box
    order = np.argsort(-r)
    x = np.full((len(r), 2), np.nan)
    placed = []
    for k in order:
        for _ in range(max_tries):
            p = np.array([rng.uniform(xl + r[k], xr - r[k]), rng.uniform(yb + r[k], yt - r[k])])
            if placed:
                q = x[placed]
                if np.any(np.hypot(*(q - p).T) < r[placed] + r[k]):
                    continue
            x[k] = p
            placed.append(k)
            break
        else:
            raise RuntimeError("could not place all discs; lower the initial packing fraction")
    return x


def generate_seed(r_full: np.ndarray, sigma0: float, mat: Material, rng: np.random.Generator,
                  phi_initial: float = 0.75, grow_from: float = 0.6, mu_compaction: float = 0.1,
                  ufr_target: float = 1e-4, stress_tol: float = 0.005, max_steps: int = 600_000,
                  servo_gain: float = 0.05, servo_step_frac: float = 2e-5,
                  log_every: int = 1000, verbose: bool = True) -> tuple[Assembly, list[dict]]:
    """Build a force-equilibrated seed specimen at isotropic stress sigma0.

    Returns the assembly (at rest) and a history of diagnostics for plotting.
    """
    t0 = time.time()
    area = np.sum(np.pi * r_full**2)
    L = np.sqrt(area / phi_initial)
    box = np.array([0.0, L, 0.0, L])
    x = random_placement(grow_from * r_full, box, rng)

    mu_final = mat.mu
    mat.mu = 0.0                           # frictionless while radii grow
    a = Assembly(x, grow_from * r_full, box, mat)
    history: list[dict] = []
    step = 0

    def record(phase: str) -> dict:
        a.compute_forces(dt=0.0)
        u = a.unbalanced_force_ratio()
        rec = {"step": step, "phase": phase, **u, "sigma": a.wall_stress().tolist(),
               "porosity": a.porosity(), "elapsed_s": time.time() - t0}
        history.append(rec)
        if verbose:
            s = a.wall_stress() / 1e3
            print(f"[{phase:>11}] step {step:>7}  ufr={u['ufr_mean']:.2e}  "
                  f"sigma(kPa)=[{s[0]:6.1f} {s[1]:6.1f} {s[2]:6.1f} {s[3]:6.1f}]  n={a.porosity():.4f}")
        return rec

    # 1. grow radii in 20 increments with fixed walls
    for s in np.linspace(grow_from, 1.0, 21)[1:]:
        a.set_radii(s * r_full)
        a.step(400); step += 400
    record("grow")

    # 2. servo-controlled isotropic compression to sigma0 with low friction (dense packing)
    mat.mu = mu_compaction
    u_max = servo_step_frac * float(np.median(2 * r_full)) / a.dt   # wall travel per step <= frac * D50
    while step < max_steps:
        for _ in range(log_every // 10):
            a.servo(sigma0, gain=servo_gain, u_max=u_max); a.step(10); step += 10
        rec = record("compress")
        if np.max(np.abs(np.array(rec["sigma"]) - sigma0)) / sigma0 < 0.02 and rec["ufr_mean"] < 1e-2:
            break

    # 3. switch to the target friction and re-equilibrate under the servo
    mat.mu = mu_final
    while step < max_steps:
        for _ in range(log_every // 10):
            a.servo(sigma0, gain=servo_gain, u_max=u_max); a.step(10); step += 10
        rec = record("equilibrate")
        if np.max(np.abs(np.array(rec["sigma"]) - sigma0)) / sigma0 < stress_tol and rec["ufr_mean"] < 10 * ufr_target:
            break

    # 4. hold walls fixed and cycle until the unbalanced force ratio meets the target
    a.stop_walls()
    while step < max_steps:
        a.step(log_every); step += log_every
        rec = record("relax")
        if rec["ufr_mean"] <= ufr_target:
            break

    a.v[:] = 0.0
    a.w[:] = 0.0
    a.compute_forces(dt=0.0)
    return a, history
