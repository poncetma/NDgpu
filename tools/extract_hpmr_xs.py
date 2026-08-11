"""Re-extract the vendored HP-MR cross sections from the VTB Griffin library.

The two ``ndgpu/benchmarks/data/hpmr_*_xs_g11.npz`` archives are what make the
HP-MR cases self-contained -- 10 KB apiece against a 1.5 MB library that lives
outside the package. They were originally produced by hand, which is how they
came to carry only the six cross-section arrays and to silently drop the
library's ``NeutronVelocity`` and delayed-neutron blocks. This script exists so
that never happens again: run it and the extract is a faithful subset.

    python tools/extract_hpmr_xs.py [path/to/fullcore_xml_G11_endfb8_ss_tr.xml]

The library is the NEAMS Virtual Test Bed's, at
``microreactors/mrad/isoxml/fullcore_xml_G11_endfb8_ss_tr.xml`` in
https://github.com/idaholab/virtual_test_bed; a copy is kept in
``dev-refs/vtb_isoxml/`` (dev-refs is reference input, never imported).

What is stored per material: D, chi, kappaFission, nuFission, sigma_a and the
scattering matrix -- plus, once, the kinetics that belong to the library rather
than to any one material: the group speeds, the precursor decay constants, the
delayed fractions and the delayed spectrum. Also the Tfuel/Tmod tabulation
axes, because they are what make a real temperature feedback possible instead
of an analytic stand-in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ndgpu.griffin_xs import (read_kinetics, read_material,  # noqa: E402
                              read_velocity, temperature_grid)

DEFAULT_XML = ROOT / "dev-refs" / "vtb_isoxml" / "fullcore_xml_G11_endfb8_ss_tr.xml"
DATA = ROOT / "ndgpu" / "benchmarks" / "data"

#: id -> key, matching ndgpu.benchmarks.hpmr.hpmr_materials_builtin
CORE_IDS = {803: "803", 805: "805", 810: "810", 811: "811", 820: "820"}
#: id -> name, matching ndgpu.benchmarks.hpmr_assembly.PIN_MATERIAL_NAMES
PIN_IDS = {803: "graphite", 801: "fuel_compact", 802: "moderator",
           816: "mod_shell", 815: "heatpipe", 817: "hp_shell"}

#: The fuel compact is the material the kinetics are tabulated against.
KINETICS_ID = 801
GRID_INDEX = "3 3"          # Tfuel = Tmod = 800 K


def _pack(xml, ids, extra=None):
    """Build the archive contents for one material set."""
    data = {}
    G = None
    for mid, key in ids.items():
        m = read_material(xml, mid, GRID_INDEX)
        G = m.n_groups
        data[f"{key}.D"] = m.diffusion
        data[f"{key}.sa"] = m.sigma_a
        data[f"{key}.nsf"] = m.nu_sigma_f
        data[f"{key}.ss"] = m.sigma_s
        # As the dataclass carries it: the library spectrum for a fissile
        # material, and Material's default [1, 0, ...] otherwise. That default
        # is what the original hand extraction stored, and it round-trips
        # identically through hpmr_materials_builtin's `chi if sum > 0 else
        # None` -- chi multiplies a zero fission source in a non-fissile
        # material either way.
        data[f"{key}.chi"] = m.chi
        data[f"{key}.kf"] = (m.kappa_fission if m.kappa_fission is not None
                             else np.zeros(G))
    data["G"] = np.array(G)
    data["ids"] = np.array(list(ids.keys()))

    # Library-level data, the same for every material: the group speeds and the
    # delayed-neutron set, plus the tabulation axes that make a real
    # temperature feedback possible instead of an analytic stand-in.
    kin = read_kinetics(xml, KINETICS_ID, GRID_INDEX)
    data["velocity"] = read_velocity(xml, KINETICS_ID, GRID_INDEX)
    data["dnp_lambda"] = kin.decay
    data["dnp_beta"] = kin.beta
    if kin.chi_delayed is not None:
        data["dnp_chi"] = kin.chi_delayed
    for axis, values in temperature_grid(xml, KINETICS_ID).items():
        data[f"grid.{axis}"] = values
    data.update(extra or {})
    return data


def _fuel_branch(xml):
    """The homogenized fuel assembly at every temperature node.

    This is what turns the library's Tfuel/Tmod tabulation into a usable
    feedback: the assembly is re-homogenized from its pins at each node, so a
    coupled solve can interpolate real cross sections instead of scaling one
    node by an analytic law.

    Read along the **diagonal** Tfuel = Tmod. The conduction model carries a
    single solid temperature per cell -- the assembly is a graphite monolith
    with the fuel, moderator and matrix in thermal contact -- so moving fuel and
    moderator together is the branch that matches it. An independent moderator
    temperature would need a second field and a 2-D interpolation.
    """
    from ndgpu.benchmarks.hpmr import (_ASSEMBLY_VOLUME_FRACTIONS,
                                       _XS_FUEL_COMPACT)
    from ndgpu.griffin_xs import volume_homogenize

    axes = temperature_grid(xml, KINETICS_ID)
    t_fuel = axes["Tfuel"]
    n_nodes = len(t_fuel)
    if not np.allclose(t_fuel, axes.get("Tmod", t_fuel)):
        raise ValueError("Tfuel and Tmod axes differ; the diagonal read is "
                         "only meaningful when they share nodes")

    out = {}
    for i in range(n_nodes):
        gi = f"{i + 1} {i + 1}"                     # 1-based, diagonal
        lib = {mid: read_material(xml, mid, gi) for mid in PIN_IDS}
        fuel = volume_homogenize(lib, _ASSEMBLY_VOLUME_FRACTIONS,
                                 chi_from=_XS_FUEL_COMPACT, name="fuel")
        for attr, key in (("diffusion", "D"), ("sigma_a", "sa"),
                          ("nu_sigma_f", "nsf"), ("sigma_s", "ss"),
                          ("chi", "chi"), ("sigma_t", "total")):
            out.setdefault(f"fuel_branch.{key}", []).append(
                np.asarray(getattr(fuel, attr)))
        kf = fuel.kappa_fission
        out.setdefault("fuel_branch.kf", []).append(
            np.zeros(fuel.n_groups) if kf is None else np.asarray(kf))
    return {k: np.stack(v) for k, v in out.items()}


def _check_against_existing(path, data):
    """Every array the old extract had must come back bit-identical.

    The extracts feed validated benchmarks, so re-extraction has to be provably
    additive. A mismatch means this script and the hand extraction disagree
    about the library, which is a finding, not something to overwrite.
    """
    if not path.exists():
        print(f"  {path.name}: new file")
        return True
    ok = True
    with np.load(path, allow_pickle=False) as old:
        for k in old.files:
            if k not in data:
                print(f"  {path.name}: MISSING {k} in new extract")
                ok = False
                continue
            a, b = np.asarray(old[k]), np.asarray(data[k])
            if a.dtype.kind in "US" or b.dtype.kind in "US":
                if a.shape != b.shape or not np.array_equal(a, b):
                    print(f"  {path.name}: CHANGED {k} (text)")
                    ok = False
                continue
            a, b = a.astype(float), b.astype(float)
            if a.shape != b.shape or not np.array_equal(a, b):
                worst = (np.max(np.abs(a - b)) if a.shape == b.shape else float("nan"))
                print(f"  {path.name}: CHANGED {k} (max |d| = {worst:.3e})")
                ok = False
        added = sorted(set(data) - set(old.files))
        print(f"  {path.name}: {len(old.files)} existing arrays reproduced"
              f"{'' if ok else ' WITH DIFFERENCES'}; adding {added}")
    return ok


def main(xml=DEFAULT_XML, write=True):
    xml = Path(xml)
    if not xml.exists():
        raise SystemExit(
            f"library not found: {xml}\n"
            "Fetch it from the VTB repo:\n"
            "  curl -sSLo dev-refs/vtb_isoxml/fullcore_xml_G11_endfb8_ss_tr.xml \\\n"
            "    https://raw.githubusercontent.com/idaholab/virtual_test_bed/main/"
            "microreactors/mrad/isoxml/fullcore_xml_G11_endfb8_ss_tr.xml")

    branch = _fuel_branch(xml)
    targets = [
        (DATA / "hpmr_core_xs_g11.npz", _pack(xml, CORE_IDS, extra=branch)),
        (DATA / "hpmr_pin_xs_g11.npz",
         _pack(xml, PIN_IDS,
               extra={"names": np.array(list(PIN_IDS.values()))})),
    ]
    all_ok = True
    for path, data in targets:
        all_ok &= _check_against_existing(path, data)
    if not all_ok:
        raise SystemExit("\nrefusing to overwrite: the re-extraction does not "
                         "reproduce the existing arrays. Investigate before "
                         "regenerating.")
    if not write:
        return targets
    for path, data in targets:
        np.savez_compressed(path, **data)
        print(f"  wrote {path.name}: {len(data)} arrays, "
              f"{path.stat().st_size / 1024:.0f} KB")

    d = targets[0][1]
    v = d["velocity"]
    print(f"\ngroup speeds    : {v.min():.3e} .. {v.max():.3e} cm/s "
          f"({len(v)} groups)")
    print(f"delayed families: {len(d['dnp_lambda'])}, "
          f"beta_total = {1e5 * d['dnp_beta'].sum():.1f} pcm")
    print(f"Tfuel nodes     : {list(d['grid.Tfuel'])} K")
    sa = d["fuel_branch.sa"]
    print(f"fuel branch     : {sa.shape[0]} nodes x {sa.shape[1]} groups; "
          f"sum(sigma_a) {sa.sum(axis=1)[0]:.6f} -> {sa.sum(axis=1)[-1]:.6f}")
    return targets


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_XML)
