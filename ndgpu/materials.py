"""Multigroup macroscopic cross-section sets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Material:
    """Homogenized multigroup diffusion constants for one material region.

    All cross sections are macroscopic, in 1/cm; diffusion coefficients in cm.

    diffusion   : D_g, shape (G,)
    sigma_a     : absorption Sigma_a,g, shape (G,)
    nu_sigma_f  : production nu*Sigma_f,g, shape (G,)
    sigma_s     : scattering matrix Sigma_s[g_from, g_to], shape (G, G).
                  The diagonal (within-group scattering) is ignored: it cancels
                  identically between the removal term and the in-scatter source.
    chi         : fission emission spectrum, shape (G,); defaults to all fission
                  neutrons born in group 0. Must sum to 1 if fissile.
    total       : optional total cross section Sigma_t,g, shape (G,). Used by
                  the SP3 solver for the higher angular moments; if omitted,
                  1/(3 D_g) is used (exact for transport-corrected data with
                  isotropic scattering).
    """

    diffusion: np.ndarray
    sigma_a: np.ndarray
    nu_sigma_f: np.ndarray
    sigma_s: np.ndarray | None = None
    chi: np.ndarray | None = None
    total: np.ndarray | None = None
    name: str = ""

    def __post_init__(self):
        self.diffusion = np.atleast_1d(np.asarray(self.diffusion, dtype=np.float64))
        G = self.n_groups
        self.sigma_a = np.asarray(self.sigma_a, dtype=np.float64).reshape(G)
        self.nu_sigma_f = np.asarray(self.nu_sigma_f, dtype=np.float64).reshape(G)
        if self.sigma_s is None:
            self.sigma_s = np.zeros((G, G))
        self.sigma_s = np.asarray(self.sigma_s, dtype=np.float64).reshape(G, G)
        if self.chi is None:
            self.chi = np.zeros(G)
            self.chi[0] = 1.0
        self.chi = np.asarray(self.chi, dtype=np.float64).reshape(G)
        if np.any(self.diffusion <= 0):
            raise ValueError("diffusion coefficients must be positive")
        if np.any(self.sigma_a < 0) or np.any(self.nu_sigma_f < 0) or np.any(self.sigma_s < 0):
            raise ValueError("cross sections must be non-negative")
        if self.is_fissile and not np.isclose(self.chi.sum(), 1.0):
            raise ValueError(f"chi must sum to 1 for fissile material, got {self.chi.sum()}")
        if self.total is not None:
            self.total = np.asarray(self.total, dtype=np.float64).reshape(G)
            if np.any(self.total <= 0):
                raise ValueError("total cross sections must be positive")

    @property
    def sigma_t(self) -> np.ndarray:
        """Sigma_t,g, falling back to 1/(3 D_g) when not provided."""
        return self.total if self.total is not None else 1.0 / (3.0 * self.diffusion)

    @property
    def n_groups(self) -> int:
        return len(self.diffusion)

    @property
    def is_fissile(self) -> bool:
        return bool(np.any(self.nu_sigma_f > 0))

    @property
    def removal(self) -> np.ndarray:
        """Sigma_r,g = Sigma_a,g + sum_{g'!=g} Sigma_s,g->g' (out-scatter), shape (G,)."""
        out_scatter = self.sigma_s.sum(axis=1) - np.diag(self.sigma_s)
        return self.sigma_a + out_scatter


@dataclass
class Kinetics:
    """Point-kinetics data for transient calculations (global for the problem).

    velocities  : neutron speeds v_g in cm/s, shape (G,)
    beta        : delayed neutron fractions per precursor family, shape (I,)
    decay       : decay constants lambda_i in 1/s, shape (I,)
    chi_delayed : delayed emission spectrum, shape (G,); defaults to the
                  prompt spectrum of each material.
    """

    velocities: np.ndarray
    beta: np.ndarray
    decay: np.ndarray
    chi_delayed: np.ndarray | None = None

    def __post_init__(self):
        self.velocities = np.atleast_1d(np.asarray(self.velocities, dtype=np.float64))
        self.beta = np.atleast_1d(np.asarray(self.beta, dtype=np.float64))
        self.decay = np.atleast_1d(np.asarray(self.decay, dtype=np.float64))
        if self.beta.shape != self.decay.shape:
            raise ValueError("beta and decay must have one value per precursor family")
        if np.any(self.velocities <= 0) or np.any(self.decay <= 0) or np.any(self.beta < 0):
            raise ValueError("velocities and decay constants must be positive, beta non-negative")
        if self.chi_delayed is not None:
            self.chi_delayed = np.asarray(self.chi_delayed, dtype=np.float64)
            self.chi_delayed = self.chi_delayed / self.chi_delayed.sum()

    @property
    def n_families(self) -> int:
        return len(self.beta)

    @property
    def beta_total(self) -> float:
        return float(self.beta.sum())


# Classic homogenized two-group PWR-like constants (thermal reactor, no upscatter).
PWR_TWO_GROUP = Material(
    name="PWR two-group",
    diffusion=[1.2627, 0.3543],
    sigma_a=[0.01207, 0.1210],
    nu_sigma_f=[0.008476, 0.18514],
    sigma_s=[[0.0, 0.02619], [0.0, 0.0]],
    chi=[1.0, 0.0],
)

# Simple one-group set with k_inf ~ 1.17 (critical bare cube side ~ 87 cm).
ONE_GROUP_DEMO = Material(
    name="one-group demo",
    diffusion=[1.3],
    sigma_a=[0.030],
    nu_sigma_f=[0.035],
)
