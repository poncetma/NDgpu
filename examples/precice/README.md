# HP-MR neutronics ↔ thermal coupling through preCICE

Two processes solving the two halves of a coupled reactor steady state and
exchanging volume fields:

| participant | reads | writes |
|---|---|---|
| `neutronics.py` | `Temperature` (K) | `Power` (W/cm³) |
| `thermal.py` | `Power` (W/cm³) | `Temperature` (K) |

The same problem also runs in a single process via `ndgpu.coupling.CoupledSolver`
(`examples/hpmr_coupled.py`). That is deliberate: the internal driver is the
reference this coupling is verified against, in
`tests/validation/test_coupled_precice.py`.

## Install

The system `libprecice` on this machine **cannot be loaded** — it is an Ubuntu
24.04 build on a 22.04 box and wants GLIBC_2.38:

```
$ precice-tools version
error while loading shared libraries: libboost_log_setup.so.1.83.0
```

`pip install pyprecice` will not rescue it: pyprecice is source-only on PyPI, so
pip compiles and links against that same dead library. Use conda-forge, which
ships binaries for both halves:

```bash
conda create -n ndgpu-precice --override-channels -c conda-forge \
    python=3.13 pyprecice numpy scipy pytest
conda run -n ndgpu-precice python -m pip install -e /path/to/ndgpu --no-deps
conda run -n ndgpu-precice python -c "import precice; print(precice.get_version_information())"
```

(Use `python -m pip`, not `pip` — a bare `pip` may resolve to `~/.local/bin/pip`
and install into the wrong environment.)

## Run

```bash
conda run -n ndgpu-precice bash examples/precice/run.sh --refine 4 --groups 11
```

Any extra arguments go to **both** participants, which matters: they must build
identical problems. Useful ones are `--refine`, `--nz` (0 = the 2D radial core),
`--drum-deg`, `--groups {2,11}`, `--power`, `--config`, `--csv`.

Output lands in `examples/precice/run/` (override with
`NDGPU_PRECICE_WORKDIR`), including a per-iteration CSV trace from each side.

## The two configurations

| file | dims | acceleration | use |
|---|---|---|---|
| `precice-config.xml` | 2 | `acceleration:constant`, relaxation 0.5 | **verification** — matches `CoupledSolver(relaxation=0.5)` expression for expression, so the two couplings produce identical iterates |
| `precice-config-iqnils.xml` | 2 | `acceleration:IQN-ILS` | **production** — same fixed point in ~6 iterations instead of ~33 |
| `precice-config-3d.xml` | 3 | `acceleration:constant`, relaxation 0.5 | the extruded core (`--nz > 0`), whose centroids carry a z coordinate |

The participants assert that the config's `dimensions` matches the centroids
they built, and say which is which rather than failing obscurely.

## Things that will bite you

- **`waveform-degree="0"` requires `substeps="false"`** on the exchanges. Steady
  coupling has nothing to interpolate within a window, and preCICE refuses the
  inconsistent combination at startup.
- **`initialize="true"` goes on the Temperature exchange**, not Power.
  `Neutronics` is `first`, so it reads before `Thermal` has run in the window;
  `Thermal` is therefore the participant whose `requires_initial_data()` returns
  true. Power is written before it is read, inside the iteration.
- **Acceleration acts on the second participant's data** in a serial-implicit
  scheme, i.e. `Temperature`. That is why the internal driver relaxes `T` and
  not `q` — relaxing the other quantity would be a different fixed-point
  iteration and the lockstep comparison would be meaningless.
- **`constraint="consistent"`, never `conservative`.** Both fields are intensive
  (W/cm³ and K); a conservative constraint sums contributions and, at coincident
  vertices, silently corrupts them.
- **Identical vertex sets are load-bearing.** Both participants call
  `ndgpu.coupling.coupling_vertices`. If they ever drift — a different `refine`,
  a different drum angle — preCICE will *not* error: nearest-neighbour will map
  each vertex to a nearby wrong cell and the coupling quietly becomes a
  smoother. That is what the exact-mapping test exists to catch.
- **XML comments cannot contain `--`.**
- **`--warm-start` is off by default.** It makes the coupled map depend on its
  own history, which breaks the lockstep comparison (and preCICE's checkpoints
  restore to the start of a window, not to the previous iteration).
