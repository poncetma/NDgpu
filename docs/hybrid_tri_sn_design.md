# Hybrid tri-Sₙ / tri-diffusion solver — design & steps

> **Status: implemented** as `ndgpu.HybridTriSNDiffusionSolver`
> (`ndgpu/hybrid_tri_sn.py`), following the steps below with the SCB drum scheme
> and the interface net-current coupling. The limits are exact (empty mask =
> `TriDiffusionEigenSolver`, full mask = `TriSNTransportSolver`) and an isolated
> drum recovers ~all of the diffusion→Sₙ correction
> (`tests/verification/test_hybrid_tri_sn.py`). On the 12-drum HP-MR it captures
> the self-shielding sign but the isotropic interface reconstruction
> over-predicts the worth magnitude — the open item below (P1 incoming / buffer
> ring). Both Schwarz and the fission outer are Anderson-accelerated.

Goal: on the body-fitted HP-MR triangular mesh, run discrete-ordinates transport
(`TriSNTransportSolver`) **only in the control-drum cells** and diffusion
(`ndgpu.tri`) everywhere else, coupled at the interface — the triangular-mesh
counterpart of the Cartesian `HybridSNDiffusionSolver`. This lets the true-drum
self-shielding be captured on the real geometry at a fraction of full-core Sₙ
cost.

The two ingredients already exist and are validated:
`TriDiffusionEigenSolver` (with `active` / `mask_bc` region masking) and
`TriSNTransportSolver` (with `active` masking, vacuum boundary, step **and**
diamond differencing). What's missing is the interface coupling. The Cartesian
hybrid established the correct physics: couple by the **interface net current**
(excise the drum from the diffusion domain; the drum's outgoing current is a
source on the ring of bulk cells), *not* by pinning the drum scalar flux
(Dirichlet double-counts the absorption). The same coupling carries over to tri.

## What the triangular mesh makes easier / harder

* **Easier** — region excision is already built in. `TriGroupOperator` zeroes
  diffusion coupling across a face touching an inactive cell and applies a Robin
  `mask_bc` there; `TriSNTransportSolver` treats active-mask boundaries as
  vacuum. So "diffusion on the bulk with the drum removed" and "Sₙ on the drum"
  are one `active=` argument each.
* **Reusable** — the diamond scheme's `_build_edges` already gives every mesh
  edge a global id, its two incident cells, its outward normal, and a
  boundary/interior flag. That is exactly the structure needed to enumerate the
  **drum↔bulk interface edges** and evaluate currents on them. The diamond work
  is therefore a prerequisite that is now done.
* **Harder** — the interface is a set of irregular triangle edges, not four box
  faces. The 12 rotating drums are non-rectangular and scattered, so the coupling
  must be per-interface-edge (the Cartesian version could assume rectangular
  boxes and full box faces).

## Steps

1. **Region masks.** `drum_mask` = the drum-absorber cells (optionally dilated by
   1–2 rings of surrounding beryllium — the buffer that made the Cartesian
   coupling closure-insensitive). `bulk = active & ~drum_mask`. Using
   `_build_edges` over the full active mesh, tag every edge as bulk–bulk,
   drum–drum, or **interface** (one drum cell, one bulk cell); keep, per
   interface edge, its bulk cell, drum cell, outward normal (bulk→drum), and edge
   length.

2. **Bulk diffusion operator (fixed-source, drum excised).** Assemble a sparse
   triangular-FV diffusion matrix per group over the whole mesh with
   `active = bulk` — harmonic-mean edge D and the tri geometry couplings
   (`w = 4 D / h²`, Robin term `8 D α / (h(hα + 2√3 D))`) already in
   `ndgpu.tri.TriGroupOperator`; interface edges carry no diffusion coupling
   (the drum is not a diffusion neighbour). Factorize once per group (as the
   Cartesian hybrid does with `scipy.factorized`). The bulk cell on each
   interface edge receives, as a **source term**, the net current the drum sends
   across that edge (Step 4).

3. **Sₙ drum solve (fixed-source, incoming from the bulk).** Run
   `TriSNTransportSolver` with `active = drum_mask`. Extend it to accept a
   per-boundary-edge **incoming angular flux** instead of hard vacuum: for an
   inflow interface edge, move the known incoming to the right-hand side of the
   per-ordinate operator, `rhs += (Ω·n) h · ψ_in`, with `ψ_in` reconstructed
   isotropically from the neighbouring bulk scalar flux (`ψ_in = φ_bulk`; add the
   P1 current term only if the isotropic closure proves too coarse — on
   Cartesian it sufficed for an isolated drum). Drum-facing void edges stay
   vacuum (`ψ_in = 0`).

4. **Interface net current.** After the drum solve, on each interface edge form
   the net current `J = Σ_m w_m (Ω_m·n) ψ_{m,face}` (outgoing edge flux where the
   ordinate exits the drum, the injected `ψ_in` where it enters — the same
   construction as `HybridSNDiffusionSolver._box_net_currents`). `J` is the
   current leaving the drum into the adjacent bulk cell; add `+J·(h/A_bulk)` to
   that bulk cell's diffusion source (Step 2), consistent with current
   continuity.

5. **Schwarz fixed point, inside the power iteration.** Per group per outer:
   (a) Sₙ drum solve with incoming from the current bulk φ; (b) recompute
   interface currents; (c) bulk diffusion solve with those current sources;
   iterate (a)–(c) to an interface fixed point. Wrap in the fission power
   iteration for k (the drums are non-fissile, so the fuel drives k and the drum
   only affects the balance through its interface current = net absorption).
   Anderson-accelerate the interface state if the 12-drum coupling converges
   slowly (as the reflective-boundary fixed point needed in `ndgpu.sn`).

6. **Limits & validation.** `drum_mask` empty → pure `TriDiffusionEigenSolver`
   (bit-for-bit); `drum_mask == active` → pure `TriSNTransportSolver` (every bulk
   cell gone, whole core is the Sₙ region). Assert both exactly. Intermediate:
   the drum worth must sit between diffusion and full tri-Sₙ, closer to Sₙ, and —
   with diamond differencing and adequate refinement — recover the transport
   self-shielding (Sₙ resolves less drum worth than diffusion) at a fraction of
   full-core Sₙ cost.

## Open questions / risks

* **Interface reconstruction.** Isotropic incoming under-weights the inward
  current; on Cartesian it was near-exact for an isolated drum but overshot for
  tightly-packed drums. The buffer ring (Step 1) and, if needed, a P1 incoming
  (`ψ_in = φ + 2·J_into_drum`, damped to avoid the positive-feedback instability
  seen on Cartesian) are the mitigations.
* **Schwarz convergence with 12 drums.** Likely needs Anderson on the interface
  current vector; each drum is otherwise independent given the bulk φ, so the
  drum solves parallelize.
* **Cost.** Each drum's Sₙ solve is over a handful of cells, so the hybrid should
  be far cheaper than the full-core tri-Sₙ — the actual payoff of the method.
* **Differencing.** The drum Sₙ wants a genuinely second-order scheme so the
  transport worth is resolved without pushing refinement as high as step needs.
  Use `scheme="scb"` (simple corner balance): a second-order finite-volume scheme
  — three corner sub-volumes per triangle, cell-boundary half-edges upwinded to
  the neighbour's corner at the shared vertex, interior corner faces carrying the
  average of the two corner fluxes. It is exact for a flat flux, stays linear
  (factorizes once), and reaches the correct HP-MR drum-worth sign about two
  refinements sooner than step (`examples/hpmr_tri_sn.py`). (An earlier
  edge-average + equal-outflow "diamond" attempt was only ~1st order — the
  equal-outflow closure is not linear-consistent — and was replaced by SCB.) In
  the hybrid, the drum Sₙ region should use SCB, and its corner-boundary
  half-edges on the drum↔bulk interface carry the net current to the bulk.
