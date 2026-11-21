"""Independent force-equilibrium check from saved files.

This deliberately shares no code with the solver: it reads the particle and contact CSVs, rebuilds
the contact geometry from positions with its own neighbour search, recomputes normal forces from
overlaps, assembles nodal forces with a sparse incidence matrix (F = B^T f), and reports the
unbalanced force and moment ratios. If the solver's force bookkeeping were wrong, the two would
disagree.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def load_specimen(folder: str | Path):
    folder = Path(folder)
    meta = json.loads((folder / "specimen_meta.json").read_text())
    p = np.loadtxt(folder / "particles.csv", delimiter=",", skiprows=1)
    c = np.loadtxt(folder / "contacts.csv", delimiter=",", skiprows=1, ndmin=2)
    return meta, p, c


def check_equilibrium(folder: str | Path) -> dict:
    meta, p, c = load_specimen(folder)
    x, r = p[:, 1:3], p[:, 3]
    N = len(r)
    kn, kw = meta["kn"], meta["kw"]
    xl, xr, yb, yt = meta["box"]

    # 1. contact geometry recomputed from positions (independent neighbour search)
    tree = cKDTree(x)
    pairs = tree.query_pairs(2 * r.max(), output_type="ndarray")
    d = np.linalg.norm(x[pairs[:, 1]] - x[pairs[:, 0]], axis=1)
    ov = r[pairs[:, 0]] + r[pairs[:, 1]] - d
    touching = pairs[ov > 0]
    found = {(int(a), int(b)) for a, b in touching}
    saved = {(int(a), int(b)) for a, b in c[:, :2]}
    geometry_match = found == saved

    # 2. normal forces recomputed from overlaps; tangential forces taken from the file
    i, j = c[:, 0].astype(int), c[:, 1].astype(int)
    dx = x[j] - x[i]
    dist = np.hypot(dx[:, 0], dx[:, 1])
    n = dx / dist[:, None]
    t = np.column_stack([-n[:, 1], n[:, 0]])
    fn = kn * (r[i] + r[j] - dist)
    ft = c[:, 3]
    fn_err = float(np.max(np.abs(fn - c[:, 2])) / np.mean(np.abs(fn)))

    # 3. nodal forces via a sparse incidence matrix: each contact pushes i by -fn*n+ft*t, j opposite
    m = len(i)
    rows = np.concatenate([i, j])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    vals = np.concatenate([np.ones(m), -np.ones(m)])
    B = sparse.csr_matrix((vals, (rows, cols)), shape=(N, m))
    f_vec = -fn[:, None] * n + ft[:, None] * t
    F = np.asarray(B @ f_vec)
    M = np.asarray(sparse.csr_matrix((np.concatenate([r[i] * ft, r[j] * ft]), (rows, cols)), shape=(N, m)).sum(axis=1)).ravel()

    # walls
    for k, (o, axis, sgn) in enumerate([(r - (x[:, 0] - xl), 0, +1), (r - (xr - x[:, 0]), 0, -1),
                                        (r - (x[:, 1] - yb), 1, +1), (r - (yt - x[:, 1]), 1, -1)]):
        hit = o > 0
        F[hit, axis] += sgn * kw * o[hit]

    fmag = np.hypot(fn, ft)
    unbal = np.hypot(F[:, 0], F[:, 1])
    return {
        "n_particles": N,
        "n_contacts": m,
        "contact_set_matches_geometry": bool(geometry_match),
        "normal_force_max_rel_error": fn_err,
        "ufr_mean": float(unbal.mean() / fmag.mean()),
        "ufr_max": float(unbal.max() / fmag.mean()),
        "moment_ratio_mean": float(np.mean(np.abs(M)) / (fmag.mean() * np.median(r))),
    }


def write_specimen(a, folder: str | Path, extra_meta: dict | None = None) -> None:
    """Save particles, contacts and metadata so check_equilibrium can re-verify them later."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    a.compute_forces(dt=0.0)
    with open(folder / "particles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "x_m", "y_m", "radius_m"])
        for k in range(a.N):
            w.writerow([k, repr(float(a.x[k, 0])), repr(float(a.x[k, 1])), repr(float(a.r[k]))])
    c = a.contacts
    with open(folder / "contacts.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["i", "j", "fn_N_per_m", "ft_N_per_m"])
        for k in range(len(c.i)):
            w.writerow([int(c.i[k]), int(c.j[k]), repr(float(c.fn[k])), repr(float(c.ft[k]))])
    meta = {"kn": a.mat.kn, "kw": a.mat.kw, "kt": a.mat.kt, "mu": a.mat.mu,
            "box": [float(b) for b in a.box], "n_particles": a.N}
    meta.update(extra_meta or {})
    (folder / "specimen_meta.json").write_text(json.dumps(meta, indent=2))
