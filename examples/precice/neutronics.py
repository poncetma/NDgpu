"""preCICE participant: the neutronics half of the HP-MR coupling.

Reads a temperature field, solves the k-eigenvalue problem with the cross
sections evaluated at that temperature, and writes back the fission power
density normalized to rated power.

    python examples/precice/neutronics.py --refine 4 --groups 11

Run it alongside ``thermal.py`` from the same working directory (see run.sh).
This participant is `first` in the serial-implicit scheme, so it reads before
the thermal side has run in each window -- which is why the Temperature
exchange is the one declared with initialize="true".
"""

import sys

import numpy as np
import precice

from common import (Trace, build_context, check_version, gather, parse_args,
                    scatter)

from ndgpu.coupling import neutronics_step

MESH = "Neutronics-Mesh"

args = parse_args("Neutronics")
version = check_version()
problem, ctx, coords, flat_idx = build_context(args)

participant = precice.Participant("Neutronics", args.config, 0, 1)
dims = participant.get_mesh_dimensions(MESH)
if coords.shape[1] != dims:
    raise SystemExit(f"mesh is {dims}-D in the config but the problem's "
                     f"centroids are {coords.shape[1]}-D "
                     f"(a 3D run needs dimensions=\"3\" in precice-config.xml)")

vertex_ids = participant.set_mesh_vertices(MESH, coords)
print(f"[Neutronics] preCICE {version.split(';')[0]}, {len(vertex_ids):,} vertices, "
      f"{ctx.materials[1].n_groups}-group, {ctx.total_power:,.0f} W", flush=True)

# Power needs no initial data: this participant writes it before the thermal
# side reads it, inside the same iteration.
if participant.requires_initial_data():
    participant.write_data(MESH, "Power", vertex_ids,
                           np.zeros(len(vertex_ids)))

participant.initialize()

trace = Trace(args.csv, ["iteration", "k_eff", "T_max", "T_mean", "power_max",
                         "outer_iterations", "seconds"])
iteration = 0
saved = None

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        # Stateless by default (warm_start off), so there is nothing to save
        # but the warm-start cache. preCICE restores to the START of a window,
        # not to the previous iteration, so keeping this honest matters as soon
        # as warm starting is switched on.
        saved = (ctx._state, ctx._k)

    dt = participant.get_max_time_step_size()
    t_vertices = participant.read_data(MESH, "Temperature", vertex_ids, dt)
    temperature = scatter(t_vertices, flat_idx, ctx.shape, ctx.ambient_temperature)

    power, result = neutronics_step(temperature, ctx)
    iteration += 1

    participant.write_data(MESH, "Power", vertex_ids, gather(power, flat_idx))
    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        ctx._state, ctx._k = saved

    trace.add(iteration=iteration, k_eff=f"{result.k_eff:.12f}",
              T_max=f"{temperature.max():.9f}", T_mean=f"{temperature.mean():.9f}",
              power_max=f"{float(np.max(power)):.9e}",
              outer_iterations=result.outer_iterations,
              seconds=f"{result.solve_seconds:.3f}")
    print(f"[Neutronics] iter {iteration:3d}  k = {result.k_eff:.7f}  "
          f"T_max = {temperature.max():8.3f} K  "
          f"({result.outer_iterations} outers, {result.solve_seconds:.2f} s)",
          flush=True)

participant.finalize()
trace.write()
print(f"[Neutronics] done after {iteration} coupling iterations", flush=True)
