# Test suite layout

Two kinds of evidence, two directories:

## `verification/` — is the *mathematics* solved correctly?

Checks against **exact references**: analytic eigenvalues (bare-box buckling,
infinite-medium k∞, the 1D slab limit), discretization *order* (error must
fall 4× per mesh doubling), exact invariants (symmetry planes, reflective ⇒
k = k∞, extruded-2D ⇒ 3D equality), cross-solver equivalence on identical
meshes, spec parsing, and exact file transcription (FEMFFUSION readers).
These tests would fail if the numerics were wrong, regardless of any physics
data. Fast; no external data needed (the FEMFFUSION reader tests skip
without a checkout).

| File | Pins down |
|---|---|
| `test_diffusion_analytic.py` | Cartesian FV vs exact bare-reactor solutions, 2nd-order convergence |
| `test_sp3_analytic.py` | SP3 vs the exact analytic SP3 eigenvalue |
| `test_boundary_conditions.py` | bc spec forms; symmetry-plane invariant |
| `test_hex_lattice.py` | hex topology, k = k∞ invariant, BC ordering |
| `test_tri_prisms.py` | tri-z extrusion: slab limit, order, extruded-2D = 3D |
| `test_unstructured_mesh.py` | unstructured FV ≡ structured FV on the same mesh |
| `test_transient_point_kinetics.py` | transient stack vs the exact point-kinetics ODE |
| `test_femffusion_io.py` | .xsec/XML readers vs the files' exact contents |

## `validation/` — does it reproduce *published reactor problems*?

Community benchmark cores solved end-to-end and compared to literature /
reference-code results. The problem definitions **and their reference
values** live together in `ndgpu/benchmarks/` (e.g.
`ndgpu.benchmarks.twigl.P_REFERENCE`); tests import them from there, so
there is a single source of truth for every published number.

| File | Benchmark | Reference |
|---|---|---|
| `test_c5g7.py` | OECD/NEA C5G7 MOX 2D | transport k = 1.18655 |
| `test_biblis_iaea.py` | BIBLIS 2D, IAEA 3D | published k's, FEMFFUSION |
| `test_vver440.py` | VVER-440 2D (tri + Gmsh mesh) | FEMFFUSION k = 1.00349 |
| `test_twigl_langenbuch.py` | TWIGL 2D, Langenbuch/LMW 3D kinetics | published power histories |
| `test_hpmr.py` | HP-MR microreactor 2D/3D | behavioural (placeholder XS): symmetry, drum worth, mesh stability |

Related directories: `ndgpu/benchmarks/` holds the benchmark *problem
builders + reference constants* (importable library code);
`examples/` holds runnable demo scripts (`speed_benchmark.py` is the
CPU-vs-GPU performance harness, not a physics benchmark); `dev-refs/` holds
third-party reference inputs used to derive data, never imported.
