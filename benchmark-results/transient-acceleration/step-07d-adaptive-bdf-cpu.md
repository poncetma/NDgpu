# Step 07d — adaptive BDF acceptance on CPU

Date: 2026-08-12

> The controller conclusions remain valid, but its LRA numbers predate the
> reflector-data correction in Step 07f and must not be used as reference
> benchmark values.

This checkpoint wires the previously tested predictor/corrector defect into
the monolithic diffusion transient. It is an experimental step-size controller,
not yet the final LSODE/Nordsieck-equivalent algorithm: BDF order ramps to the
requested maximum and stays there between explicit history restarts, and the
error norm currently contains multigroup flux but not precursor/temperature
components.

> Superseded norm note (2026-08-18): Step 07e adds precursor and coupled
> temperature fields to this acceptance norm. The historical rows below are
> retained as the flux-only checkpoint and must not be mixed with 07e counts.

## Implemented behavior

- scalar `dt` is the initial proposed width;
- accepted widths are bounded by `min_dt` and `max_dt`;
- failed error tests halve the width without advancing flux, precursor,
  neutron BDF, thermal BDF, or `on_step` history;
- accepted widths use the bounded Cherezov Eq.-45-style proposal;
- steps land exactly at `bdf_restart_times` and at `t_end`;
- the first step after an event restarts at BDF1;
- accepted/rejected errors, widths, orders, FGMRES work, and feedback
  constituent work are returned in `TransientResult`;
- the LRA driver restarts neutron and thermal histories after the 2 s rod-law
  corner.

## Manufactured kinetics gate

A homogeneous one-group absorption ramp followed by a constant endpoint uses
15 accepted and 9 rejected steps at `rtol=1e-2`. It lands exactly at the
0.01 s ramp corner, restarts the next step at BDF1, ends exactly at 0.02 s,
and every accepted normalized defect is at most one. The terminal power agrees
with a fine BDF3 reference within the deliberately loose test envelope.

## LRA-2D results

All rows use fully implicit temperature feedback and BDF5. Adaptive runs start
at 0.1 ms and align/restart at 2 s. Runtime includes rejected work.

| FV mesh | control | accepted | rejected | min/max h (s) | FGMRES total | runtime (s) | t1 (s) | P1 | P2 | P(3s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 cm | fixed 10 ms | 300 | 0 | .01/.01 | 12,710 | 8.66 | 1.350 | 5641 | 963 | 108.442 |
| 15 cm | adaptive `1e-2` | 184 | 8 | .0001/.0968 | — | 5.63 | 1.343 | 5884 | 963 | 108.471 |
| 15 cm | adaptive `3e-3` | 221 | 12 | .0001/.0794 | — | 6.25 | 1.344 | 5891 | 963 | 108.469 |
| 15 cm | adaptive `1e-3` | 268 | 14 | .0001/.0697 | — | 6.53 | 1.345 | 5910 | 963 | 108.465 |
| 15 cm | adaptive `1e-5` | 569 | 26 | 4.64e-6/.0438 | 17,759 | 10.95 | 1.345 | 5910 | 963 | 108.459 |
| 3.75 cm | fixed 10 ms | 300 | 0 | .01/.01 | — | 39.71 | 1.650 | 5339 | 416 | 83.831 |
| 3.75 cm | adaptive `1e-3` | 250 | 17 | .0001/.0956 | 12,228 | 28.29 | 1.648 | 5304 | 415 | 83.830 |

Power is W/cm3. Timings vary with host load; work counts are the stronger
comparison. Dashes indicate older runs made before rejected-work telemetry was
added.

## Comparison with Cherezov et al.

For the published `RTOL=1e-5` LRA run, Cherezov Table 7 reports approximately
399–403 accepted steps and 5–7 rejected steps for representative first/fourth
spatial orders. FEMCORE rapidly reaches BDF5 and predominantly stays there.
ndgpu likewise spends 561 of 569 accepted steps at BDF5 (two BDF1–4 startup
ramps, before and after 2 s), but currently takes 569 accepted and 26 rejected
steps. It is therefore more conservative and rejects more often.

This is not yet an equal-tolerance algorithm comparison: FEMCORE's Nordsieck
controller evaluates the full coupled state and compares `q-1/q/q+1` width
proposals, whereas this checkpoint controls a polynomial-extrapolation flux
defect at fixed maximum order. The nominal-tolerance row is retained because
it is reproducible, not because the two error norms are equivalent.

## Backward-Euler comparison

At 15 cm and fixed 10 ms, fully implicit backward Euler takes 13.51 s and
23,772 FGMRES applications while damping the first peak to 2810 W/cm3. At
5 ms it takes 22.96 s and reaches only 4054 W/cm3. Adaptive BDF5 resolves a
roughly 5910 W/cm3 peak in 5.6–11.0 s depending on tolerance. A final speedup
claim still requires a much finer backward-Euler ladder to match that prompt
peak/history error; equal `dt` is not an equal-accuracy comparison.

## Default-readiness decision

Adaptive stepping is **not a default yet**. Required gates are:

1. add precursor and coupled thermal state to the accepted error norm;
2. implement and verify automatic `q-1/q/q+1` order selection;
3. demonstrate tolerance convergence on LRA and exact/manufactured kinetics;
4. compare against backward Euler at matched history error;
5. pass slow, fast, asymmetric, and discontinuous HP-MR control cases without
   missed events, positivity failures, or excessive rejection;
6. show a net CPU and GPU win including rejected work and operator rebuilds.

The refined LRA result also closes one diagnostic question: adaptive and fixed
BDF give essentially the same late/weak excursion. The remaining LRA gap is
spatial/control-worth related, not an artifact of the fixed 10 ms grid.
