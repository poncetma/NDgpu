# HPMR drum-refinement acceptance

These files preserve the one-A100, 11-group exact-volume-mixing study performed
on 2026-09-02. See
[`docs/hpmr_drum_refinement.md`](../../docs/hpmr_drum_refinement.md) for the
decision and multi-GPU consequences.

| Job | Result | Purpose |
|---:|---|---|
| 203205 | `local-exact-result.json` | r4 local levels 0-3 endpoints |
| 203206 | `global-exact-result.json` | uniform r16/r24/r32 endpoints |
| 203274 | `local-exact-curve-result.json` | local effective-r32 curve |
| 203275 | `global-exact-curve-result.json` | uniform-r32 curve |
| 203276 | `local-exact-fine-curve-result.json` | local r4 + 4 levels |
| 203279 | `local-exact-balanced-curve-result.json` | local r8 + 3 levels |
| 203282 | `local-exact-balanced16-curve-result.json` | local r16 + 2 levels |
| 203278 | `global-exact-fine-curve-result.json` | uniform-r64 reference |

All curve jobs evaluated 85, 87, 89, 90, 90.25, 90.5, 91, 93, and 95 degrees.
The accepted working mesh is r8 plus three local levels; r16 plus two local
levels is the verification mesh.
