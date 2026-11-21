"""2-D dynamic DEM for disc assemblies in a rectangular box of rigid, frictionless walls.

Contact model (per unit thickness):
  normal      f_n = k_n * overlap                    (linear spring, repulsive only)
  tangential  f_t += k_t * v_t * dt,  |f_t| <= mu*f_n (incremental spring with Coulomb cap)
  wall        f_w = k_w * overlap,  k_w = 2 k_n,  frictionless

Integration is explicit (central difference) with Cundall's local non-viscous damping, the
usual choice for quasi-static specimen generation. Density scaling (raising the inertial mass of
the smallest discs) can be switched on to allow a larger stable timestep; it changes how fast the
assembly reaches equilibrium, not the equilibrium itself.

Why k_w = 2 k_n and frictionless walls: a disc touching a wall with overlap delta is mechanically
identical to the same disc touching its own mirror image across that wall (disc-disc overlap
2*delta, force k_n * 2*delta = k_w * delta, and zero relative tangential velocity at the contact).
That identity is what makes mirror ("flip") tiling of an equilibrated seed exact; see tiling.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 3


@dataclass
class Material:
    kn: float = 1.0e8          # normal contact stiffness, N/m per m thickness
    kt_ratio: float = 0.75     # k_t / k_n
    mu: float = 0.5            # inter-particle friction coefficient
    rho: float = 2650.0        # particle density, kg/m^3 (quartz)
    local_damping: float = 0.7
    viscous_damping: float = 0.0   # extra global velocity damping per step (calms rattlers)

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    @property
    def kw(self) -> float:
        return 2.0 * self.kn


@dataclass
class Contacts:
    """Active disc-disc contacts, sorted by key = i*N + j (i < j)."""
    i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    key: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ft: np.ndarray = field(default_factory=lambda: np.zeros(0))   # along t = (-n_y, n_x), n from i to j


class Assembly:
    def __init__(self, x: np.ndarray, r: np.ndarray, box: np.ndarray, mat: Material,
                 density_scaling: bool = True, dt_safety: float = 0.2):
        self.N = len(r)
        self.x = np.asarray(x, float).copy()
        self.r = np.asarray(r, float).copy()
        self.box = np.asarray(box, float).copy()       # [x_left, x_right, y_bottom, y_top]
        self.mat = mat
        self.v = np.zeros((self.N, 2))
        self.w = np.zeros(self.N)
        self.wall_u = np.zeros(4)                      # inward wall speeds
        self.contacts = Contacts()
        self.density_scaling = density_scaling
        self.dt_safety = dt_safety
        self.skin = 0.1 * float(np.median(2 * self.r))
        self._update_mass()
        self._build_neighbours()
        self.wall_force = np.zeros(4)
        self.F = np.zeros((self.N, 2))
        self.T = np.zeros(self.N)

    # ------------------------------------------------------------------ setup
    def _update_mass(self) -> None:
        self.mass = self.mat.rho * np.pi * self.r**2          # true mass per unit thickness
        m_in = self.mass.copy()
        if self.density_scaling:
            m_in = np.maximum(m_in, np.median(self.mass))
        self.m_in = m_in
        self.I_in = 0.5 * m_in * self.r**2
        # explicit-integration stability condition: dt < sqrt(m / k) for the stiffest spring
        self.dt_crit = float(np.sqrt(self.m_in.min() / self.mat.kw))
        self.dt = self.dt_safety * self.dt_crit

    def set_radii(self, r: np.ndarray) -> None:
        self.r = np.asarray(r, float).copy()
        self._update_mass()
        self._build_neighbours()

    def _build_neighbours(self) -> None:
        cutoff = 2 * self.r.max() + self.skin
        pairs = cKDTree(self.x).query_pairs(cutoff, output_type="ndarray")
        if len(pairs) == 0:
            self.nl_i = self.nl_j = np.zeros(0, dtype=np.int64)
        else:
            i, j = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)
            d = np.linalg.norm(self.x[j] - self.x[i], axis=1)
            keep = d < self.r[i] + self.r[j] + self.skin
            self.nl_i, self.nl_j = i[keep], j[keep]
        self.x_built = self.x.copy()

    def _maybe_rebuild(self) -> None:
        disp = np.max(np.abs(self.x - self.x_built)) if self.N else 0.0
        if disp > 0.5 * self.skin:
            self._build_neighbours()

    # ------------------------------------------------------------------ forces
    def compute_forces(self, dt: float | None = None) -> None:
        """Contact + wall forces and torques at the current state. Updates tangential history."""
        dt = self.dt if dt is None else dt
        N, x, r, v, w = self.N, self.x, self.r, self.v, self.w
        i, j = self.nl_i, self.nl_j
        dx = x[j] - x[i]
        d = np.hypot(dx[:, 0], dx[:, 1])
        ov = r[i] + r[j] - d
        c = ov > 0
        i, j, dx, d, ov = i[c], j[c], dx[c], d[c], ov[c]
        n = dx / d[:, None]
        t = np.column_stack([-n[:, 1], n[:, 0]])
        key = i * N + j
        order = np.argsort(key)
        i, j, n, t, ov, key = i[order], j[order], n[order], t[order], ov[order], key[order]

        # carry tangential force history over from the previous step
        ft_old = np.zeros(len(key))
        old = self.contacts
        if len(old.key) and len(key):
            pos = np.searchsorted(old.key, key)
            pos_c = np.minimum(pos, len(old.key) - 1)
            hit = old.key[pos_c] == key
            ft_old[hit] = old.ft[pos_c[hit]]

        vr = v[j] - v[i]
        vt = np.einsum("ij,ij->i", vr, t) - (w[j] * r[j] + w[i] * r[i])
        fn = self.mat.kn * ov
        ft = ft_old + self.mat.kt * vt * dt
        cap = self.mat.mu * fn
        ft = np.clip(ft, -cap, cap)
        self.contacts = Contacts(i, j, key, fn, ft)

        fx_i = -fn * n[:, 0] + ft * t[:, 0]
        fy_i = -fn * n[:, 1] + ft * t[:, 1]
        Fx = (np.bincount(i, fx_i, N) - np.bincount(j, fx_i, N)).astype(float)
        Fy = (np.bincount(i, fy_i, N) - np.bincount(j, fy_i, N)).astype(float)
        T = (np.bincount(i, r[i] * ft, N) + np.bincount(j, r[j] * ft, N)).astype(float)

        # frictionless rigid walls
        kw = self.mat.kw
        xl, xr, yb, yt = self.box
        wf = np.zeros(4)
        o = r - (x[:, 0] - xl); m = o > 0; Fx[m] += kw * o[m]; wf[LEFT] = kw * o[m].sum()
        o = r - (xr - x[:, 0]); m = o > 0; Fx[m] -= kw * o[m]; wf[RIGHT] = kw * o[m].sum()
        o = r - (x[:, 1] - yb); m = o > 0; Fy[m] += kw * o[m]; wf[BOTTOM] = kw * o[m].sum()
        o = r - (yt - x[:, 1]); m = o > 0; Fy[m] -= kw * o[m]; wf[TOP] = kw * o[m].sum()

        self.F = np.column_stack([Fx, Fy])
        self.T = T
        self.wall_force = wf

    # ------------------------------------------------------------------ walls
    def wall_stress(self) -> np.ndarray:
        """Normal stress on each wall (Pa) = wall force per unit thickness / wall length."""
        xl, xr, yb, yt = self.box
        lengths = np.array([yt - yb, yt - yb, xr - xl, xr - xl])
        return self.wall_force / lengths

    def wall_contact_counts(self) -> np.ndarray:
        x, r = self.x, self.r
        xl, xr, yb, yt = self.box
        return np.array([np.sum(r > x[:, 0] - xl), np.sum(r > xr - x[:, 0]),
                         np.sum(r > x[:, 1] - yb), np.sum(r > yt - x[:, 1])])

    def servo(self, sigma0: float, gain: float = 0.3, u_max: float | None = None) -> None:
        """Set inward wall speeds so each wall stress moves towards sigma0 (PFC-style servo)."""
        if u_max is None:
            u_max = 1e-4 * float(np.median(2 * self.r)) / self.dt
        sig = self.wall_stress()
        xl, xr, yb, yt = self.box
        lengths = np.array([yt - yb, yt - yb, xr - xl, xr - xl])
        nc = np.maximum(self.wall_contact_counts(), 1)
        u = gain * (sigma0 - sig) * lengths / (self.mat.kw * nc * self.dt)
        self.wall_u = np.clip(u, -u_max, u_max)

    def stop_walls(self) -> None:
        self.wall_u[:] = 0.0

    # ------------------------------------------------------------------ time stepping
    def step(self, n: int = 1) -> None:
        a = self.mat.local_damping
        vd = self.mat.viscous_damping
        for _ in range(n):
            self._maybe_rebuild()
            self.compute_forces()
            F, T = self.F, self.T
            Fd = F - a * np.abs(F) * np.sign(self.v)
            Td = T - a * np.abs(T) * np.sign(self.w)
            self.v += Fd / self.m_in[:, None] * self.dt
            self.w += Td / self.I_in * self.dt
            if vd:
                self.v *= 1.0 - vd
                self.w *= 1.0 - vd
            self.x += self.v * self.dt
            u = self.wall_u * self.dt
            self.box += np.array([u[LEFT], -u[RIGHT], u[BOTTOM], -u[TOP]])

    # ------------------------------------------------------------------ diagnostics
    def unbalanced_force_ratio(self) -> dict:
        """Force-equilibrium measures at the current state (call after compute_forces).

        ufr_mean = mean |net force on a disc| / mean |contact force|   (PFC 'average ratio')
        ufr_max  = max  |net force on a disc| / mean |contact force|
        Net force excludes damping; there is no gravity in specimen generation.
        """
        c = self.contacts
        fmag = np.hypot(c.fn, c.ft)
        mean_contact = fmag.mean() if len(fmag) else np.nan
        unbal = np.hypot(self.F[:, 0], self.F[:, 1])
        return {
            "ufr_mean": float(unbal.mean() / mean_contact),
            "ufr_max": float(unbal.max() / mean_contact),
            "mean_contact_force": float(mean_contact),
            "moment_ratio_mean": float(np.mean(np.abs(self.T)) / (mean_contact * np.median(self.r))),
        }

    def porosity(self) -> float:
        xl, xr, yb, yt = self.box
        return float(1.0 - np.sum(np.pi * self.r**2) / ((xr - xl) * (yt - yb)))

    def coordination(self) -> dict:
        """Mean coordination number, ignoring rattlers (discs with fewer than 2 contacts)."""
        c = self.contacts
        z = np.bincount(c.i, minlength=self.N) + np.bincount(c.j, minlength=self.N)
        z = z + (self.r > self.x[:, 0] - self.box[0]) + (self.r > self.box[1] - self.x[:, 0]) \
              + (self.r > self.x[:, 1] - self.box[2]) + (self.r > self.box[3] - self.x[:, 1])
        active = z >= 2
        return {"z_mean_nonrattler": float(z[active].mean()), "rattler_fraction": float(1 - active.mean())}

    def kinetic_energy(self) -> float:
        return float(0.5 * np.sum(self.m_in * np.sum(self.v**2, axis=1)) + 0.5 * np.sum(self.I_in * self.w**2))
