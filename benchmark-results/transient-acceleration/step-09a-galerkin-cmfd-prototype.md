# Step 09a: conservative Galerkin/CMFD prototype

Date: 2026-08-18

## Decision

The two-level algebra is correct and strongly accelerates a system whose slow
error is spatial, but it does **not** accelerate the current 11-group HP-MR
monolithic transient step. Keep the prototype and its regression tests; do not
wire it into the production transient controls or port its coarse solve to GPU.
The HP-MR bottleneck is not mesh-dependent spatial error at this stage.

## Prototype

The host-only prototype implements the conservative full-energy correction

```text
z        = M_group^-1 r
r_coarse = P^T (r - A z)
A_coarse = P^T A P
z       += P A_coarse^-1 r_coarse
```

where `P` is piecewise constant over geometric aggregates and `P.T` sums fine
balance equations. Optional labels prevent aggregates crossing active or
material boundaries. All 11 energy groups remain explicit. The exact sparse
fine block is assembled from the existing local Cartesian/triangular stencils;
the fine production solve remains matrix-free. A sparse LU factors the coarse
operator once.

Three compositions were tested: additive, fine-smoother-first multiplicative,
and coarse-first multiplicative. Factor-2 material-preserving aggregation is
the best real-core variant. Coarser factors and active-only/unlabelled
aggregates are worse.

## Simplified-system validation

A heterogeneous two-group 1-D diffusion/fission system with block Jacobi
smoothing deliberately leaves long-wavelength spatial error. Factor-4 additive
coarse correction gives:

| fine cells | plain FGMRES | Galerkin/CMFD FGMRES |
|---:|---:|---:|
| 32 | 32 | 23 |
| 64 | 65 | 30 |
| 128 | 136 | 37 |

At 128 cells the iteration reduction is 3.68x and becomes stronger with mesh
refinement, the expected multilevel signature. Independent tests verify
`A_H x = P.T A P x` and exact recovery of every piecewise-constant coarse-space
error to roundoff. FGMRES solutions agree with the unaccelerated solution.

## 2-D 11-group HP-MR result

One 20 ms bulk-insertion step uses the established three-sweep energy-group
preconditioner, `RTOL=1e-9`, and material-preserving factor-2 aggregates.

| refinement | active cells | method | outer | inner PCG | fine block applies | coarse cells |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 330 | group sweep | 26 | 1,304 | 26 | -- |
| 1 | 330 | + CMFD | 28 | 1,361 | 56 | 181 |
| 2 | 1,320 | group sweep | 25 | 2,112 | 25 | -- |
| 2 | 1,320 | + CMFD | 24 | 1,983 | 48 | 550 |
| 3 | 2,970 | group sweep | 25 | 2,995 | 25 | -- |
| 3 | 2,970 | + CMFD | 24 | 2,837 | 48 | 1,009 |

The CMFD solutions retain the same power and flux agreement with the converged
fixed-point reference, and true relative residuals remain below `1e-9`.
However:

- baseline outer iterations are already mesh-independent: 26, 25, 25;
- at refinements 2/3 CMFD saves only one outer and 5--6% inner work;
- forming the post-smoothing residual nearly doubles fine block applies;
- refinement 1 becomes worse in every work counter;
- representative interleaved CPU timings are slower with CMFD, before charging
  its roughly 0.05/0.08/0.12 s setup cost at refinements 1/2/3.

Additive correction is substantially worse (for example 25 to 45 outers at
refinement 2). Coarse-first multiplication does not change the conclusion.
Replacing the group sweep with block Jacobi requires hundreds of outer
iterations and is not competitive.

## Interpretation

The simplified ladder proves the restriction, Galerkin operator, factorization
and prolongation do remove spatial low modes. HP-MR's flat outer count proves
those modes have already been handled by the within-group PCG solves. Its
remaining FGMRES work is dominated by energy/fission coupling and the
near-critical amplitude family. A spatial CMFD correction attacks the wrong
subspace and pays an extra fine block application to do it.

The next coarse experiment should therefore be energy-aware rather than a GPU
port of this prototype: multigroup source-expansion/energy condensation, or a
small recycled Ritz/deflation space built from prior FGMRES solves. It must
avoid a second fine block application per preconditioner call and should be
tested against the existing adjoint rank-one correction.

## Reproduction

```bash
PYTHONPATH=. pytest -q tests/verification/test_multigroup_step.py

PYTHONPATH=. python examples/monolithic_hpmr_step_bench.py \
  --refine 2 --scatter-sweeps 3

PYTHONPATH=. python examples/monolithic_hpmr_step_bench.py \
  --refine 2 --scatter-sweeps 3 --spatial-cmfd \
  --cmfd-factor 2 --cmfd-labels material --cmfd-mode multiplicative
```
