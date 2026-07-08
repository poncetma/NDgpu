"""Extract C5G7 7-group cross sections from OpenMOC's c5g7-mgxs-hdf5.py by
exec-ing it with a stubbed h5py, then emit ndgpu/benchmarks/_c5g7_data.py."""

import sys
import types
from pathlib import Path

import numpy as np

captured = {}


class StubGroup:
    def __init__(self, name=""):
        self.name = name
        self.attrs = {}

    def create_group(self, name):
        return StubGroup(name)

    def create_dataset(self, key, data=None):
        captured.setdefault(self.name, {})[key] = np.asarray(data, dtype=float)

    def close(self):
        pass


h5py_stub = types.ModuleType("h5py")
h5py_stub.File = lambda *a, **k: StubGroup("__file__")
sys.modules["h5py"] = h5py_stub

src = Path(__file__).with_name("c5g7-mgxs-hdf5.py").read_text()
exec(compile(src, "c5g7-mgxs-hdf5.py", "exec"), {"__name__": "__main__"})

G = 7
order = ["UO2", "MOX-4.3%", "MOX-7%", "MOX-8.7%", "Fission Chamber", "Guide Tube",
         "Water", "Control Rod"]
assert set(order) <= set(captured), captured.keys()

out = [
    '"""C5G7 7-group transport-corrected cross sections (NEA/NSC/DOC(2001)4).',
    "",
    "Auto-extracted from OpenMOC's sample-input/c5g7-mgxs-hdf5.py",
    "(https://github.com/mit-crpg/OpenMOC). sigma_s is indexed [g_from, g_to].",
    '"""',
    "",
    "# fmt: off",
    "C5G7_XS = {",
]
for name in order:
    d = captured[name]
    assert d["total"].shape == (G,) and d["scatter matrix"].shape == (G * G,)
    # sanity: scattering row sums must not exceed total
    st, ss = d["total"], d["scatter matrix"].reshape(G, G)
    assert np.all(ss.sum(axis=1) <= st + 1e-12), name
    out.append(f"    {name!r}: {{")
    for key, arr in [("total", st), ("fission", d["fission"]),
                     ("nu_fission", d["nu-fission"]), ("chi", d["chi"])]:
        vals = ", ".join(f"{v:.6E}" for v in arr)
        out.append(f"        {key!r}: [{vals}],")
    out.append(f"        'scatter': [  # [g_from][g_to]")
    for row in ss:
        vals = ", ".join(f"{v:.6E}" for v in row)
        out.append(f"            [{vals}],")
    out.append("        ],")
    out.append("    },")
out.append("}")
out.append("# fmt: on")

dest = Path(__file__).parents[1] / "ndgpu" / "benchmarks" / "_c5g7_data.py"
dest.parent.mkdir(exist_ok=True)
dest.write_text("\n".join(out) + "\n")
print(f"wrote {dest}")
for name in order:
    st = captured[name]["total"]
    print(f"  {name:16s} sigma_t g1={st[0]:.5f} g7={st[-1]:.5f} "
          f"nsf_sum={captured[name]['nu-fission'].sum():.4f}")
