"""Geometric "flip" (mirror) tiling of an equilibrated seed specimen.

Tile (a, b) of an nx-by-ny grid is the seed reflected in x when a is odd and in y when b is odd:

    +-----+-----+-----+-----+
    | F   | F'  | F   | F'  |      F  = seed
    +-----+-----+-----+-----+      F' = seed mirrored left-right
    | F'' | F'''| F'' | F'''|      F'' = seed mirrored top-bottom
    +-----+-----+-----+-----+      F'''= both

Two neighbouring tiles always meet across a mirror line, so a disc that touched a seed wall now
touches its own reflection. With frictionless walls and k_w = 2 k_n (see dem.py), that disc-image
contact carries exactly the force the wall carried, and its tangential force is zero, as the wall's
was. Every tile is itself in equilibrium (reflection maps equilibrium states to equilibrium states),
so the tiled assembly inherits the seed's force balance and its contact-force chains run straight
across the tile boundaries. Plain translation tiling has none of these properties: the right face
of one tile meets the left face of the next, which were never in contact.
"""
from __future__ import annotations

import numpy as np

from .dem import Assembly, Contacts


def flip_tile(seed: Assembly, nx: int, ny: int) -> Assembly:
    """Return a new Assembly of nx*ny mirrored copies of `seed` (seed must be at rest)."""
    xl, xr, yb, yt = seed.box
    Lx, Ly = xr - xl, yt - yb
    Ns = seed.N
    xs, rs = [], []
    v_s, w_s = [], []
    ci, cj, cft = [], [], []
    for b in range(ny):
        for a in range(nx):
            fx, fy = a % 2 == 1, b % 2 == 1
            p = seed.x.copy()
            v = seed.v.copy()
            p[:, 0] = xl + (a + 1) * Lx - (p[:, 0] - xl) if fx else xl + a * Lx + (p[:, 0] - xl)
            p[:, 1] = yb + (b + 1) * Ly - (p[:, 1] - yb) if fy else yb + b * Ly + (p[:, 1] - yb)
            if fx:
                v[:, 0] *= -1
            if fy:
                v[:, 1] *= -1
            sign = -1.0 if (fx ^ fy) else 1.0     # a single reflection reverses rotation sense
            offset = (b * nx + a) * Ns
            xs.append(p); rs.append(seed.r); v_s.append(v); w_s.append(sign * seed.w)
            c = seed.contacts
            ci.append(c.i + offset); cj.append(c.j + offset)
            # f_t is measured along t = rot90(n); under one reflection t -> -M t, so f_t flips sign
            cft.append(sign * c.ft)

    x = np.vstack(xs)
    r = np.concatenate(rs)
    box = np.array([xl, xl + nx * Lx, yb, yb + ny * Ly])
    big = Assembly(x, r, box, seed.mat, density_scaling=seed.density_scaling, dt_safety=seed.dt_safety)
    big.v = np.vstack(v_s)
    big.w = np.concatenate(w_s)
    i = np.concatenate(ci); j = np.concatenate(cj)
    key = i * big.N + j
    order = np.argsort(key)
    big.contacts = Contacts(i[order], j[order], key[order], np.zeros(len(key)), np.concatenate(cft)[order])
    big.tile_shape = (nx, ny)
    big.seed_size = (Lx, Ly)
    return big


def tile_index(big: Assembly, seed_n: int) -> np.ndarray:
    """Tile number of every disc (discs are stored tile by tile)."""
    return np.arange(big.N) // seed_n
