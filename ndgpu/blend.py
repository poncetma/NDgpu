"""Per-cell material property lookup with two-material volume mixing.

Extracted from :class:`ndgpu.solver.Fields` so that every physics built on the
same mesh -- neutron cross sections, thermal conductivity, heat-pipe sink
strength -- expands its per-material tables through *identical* rules. That
matters at the mixed cells: a control-drum arc that covers 30% of a triangle
must be 30% B4C to the neutronics and 30% B4C to the conduction solve, or the
two physics disagree about where the absorber is and the coupling quietly
solves a different problem than either code alone.

The rules themselves (why linear for reaction cross sections, harmonic for the
diffusion coefficient, fission-weighted for the emission spectrum) are
documented at each method.
"""

from __future__ import annotations

import numpy as np


class MaterialBlend:
    """Expands per-material tables onto the grid, honouring a volume blend.

    material_map : integer map into the materials list, shape ``shape``. None
        is allowed only for a single material with no mixing.
    mix_material / mix_weight : the per-cell second material and its volume
        fraction w, so a mixed cell is ``(1 - w) * base + w * mix``. The
        sentinel ``mix_material < 0`` means "no blend here"; those cells stay
        bit-identical to the pure-index lookup.
    """

    def __init__(self, xp, shape, material_map, n_materials, dtype=np.float64,
                 mix_material=None, mix_weight=None):
        self.xp = xp
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.mix = mix_material is not None
        self.material_map = material_map

        if material_map is None:
            if n_materials > 1:
                raise ValueError("material_map is required with multiple materials")
            if self.mix:
                raise ValueError("mixing requires an explicit material_map")
            self.mmap = None
            return

        mmap = xp.asarray(np.asarray(material_map))
        if mmap.shape != self.shape:
            raise ValueError(f"material_map shape {mmap.shape} != grid shape "
                             f"{self.shape}")
        if int(mmap.min()) < 0 or int(mmap.max()) >= n_materials:
            raise ValueError("material_map indexes outside the materials list")
        self.mmap = mmap

        if self.mix:
            mm2 = xp.asarray(np.asarray(mix_material))
            w = xp.asarray(np.asarray(mix_weight), dtype=self.dtype)
            if mm2.shape != self.shape or w.shape != self.shape:
                raise ValueError("mix_material/mix_weight shape must match grid")
            if int(mm2.max()) >= n_materials:
                raise ValueError("mix_material indexes outside the materials list")
            self.active_mix = mm2 >= 0
            self.mm2c = xp.where(self.active_mix, mm2, 0)
            self.w = w

    def _blend(self, table, combine):
        xp = self.xp
        if self.mmap is None:
            return xp.full(self.shape, float(table[0]), dtype=self.dtype)
        dev = xp.asarray(table, dtype=self.dtype)
        base = dev[self.mmap]
        if self.mix:
            base = xp.where(self.active_mix,
                            combine(base, dev[self.mm2c], self.w), base)
        return base

    def linear(self, table):
        """Volume-linear blend -- exact reaction-rate averaging under a flat
        flux. The rule for cross sections, and for volumetric quantities like a
        heat-pipe sink coefficient or a power density."""
        return self._blend(table, lambda b, o, wt: (1.0 - wt) * b + wt * o)

    def harmonic(self, table):
        """Harmonic blend -- the rule for a diffusion coefficient (its
        transport cross section 1/(3D) volume-averages, so a trace of a strong
        absorber correctly chokes the cell) and, for the same reason, for a
        thermal conductivity in series across a cell."""
        return self._blend(table, lambda b, o, wt: 1.0 / ((1.0 - wt) / b + wt / o))

    def fission_weighted(self, table, production):
        """Blend by share of the cell's fission production, not by volume.

        For a quantity that rides on the fission source (the emission spectrum
        chi, the delayed fraction beta). Blending chi linearly would leave a
        fissile material mixed with a non-fissile one emitting a spectrum that
        sums to w -- silently losing (1 - w) of the cell's fission neutrons.
        ``production`` is the per-material total nu*Sigma_f (a flat-flux proxy;
        exact whenever at most one component is fissile, e.g. a fuel pin
        blended with moderator). Cells where neither component is fissile keep
        the base value: it multiplies a zero source.
        """
        xp = self.xp
        if not self.mix:
            return self.linear(table)
        pdev = xp.asarray(production, dtype=self.dtype)
        wb = (1.0 - self.w) * pdev[self.mmap]
        wo = self.w * pdev[self.mm2c]
        den = wb + wo
        blendable = self.active_mix & (den > 0)
        safe_den = xp.where(den > 0, den, 1.0)
        dev = xp.asarray(table, dtype=self.dtype)
        base = dev[self.mmap]
        merged = (wb * base + wo * dev[self.mm2c]) / safe_den
        return xp.where(blendable, merged, base)

    def field(self, values, harmonic=False):
        """Accept either a ready per-cell array or a per-material table.

        The convenience every physics wants: material properties are naturally
        written per material, but a caller with a computed field (a measured
        conductivity map, say) should be able to hand it over directly.
        """
        arr = np.asarray(values)
        if arr.shape == self.shape:
            return self.xp.asarray(arr, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError(
                f"expected a per-material table (1-D) or a per-cell field with "
                f"shape {self.shape}, got shape {arr.shape}")
        return self.harmonic(arr) if harmonic else self.linear(arr)
