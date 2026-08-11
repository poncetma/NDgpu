"""preCICE participant: the thermal half of the HP-MR coupling.

Reads a fission power density, solves steady conduction with the heat-pipe
sink, and writes back the temperature field.

    python examples/precice/thermal.py --refine 4 --groups 11

Run it alongside ``neutronics.py`` from the same working directory (see run.sh).
This participant is `second` in the serial-implicit scheme, so its Temperature
is the accelerated quantity -- which is why the internal driver relaxes T and
not the power, or the two would be different fixed-point iterations.
"""

import numpy as np
import precice

from common import (Trace, build_context, check_version, gather, parse_args,
                    scatter)

from ndgpu.coupling import thermal_step

MESH = "Thermal-Mesh"

args = parse_args("Thermal")
version = check_version()
problem, ctx, coords, flat_idx = build_context(args)

participant = precice.Participant("Thermal", args.config, 0, 1)
dims = participant.get_mesh_dimensions(MESH)
if coords.shape[1] != dims:
    raise SystemExit(f"mesh is {dims}-D in the config but the problem's "
                     f"centroids are {coords.shape[1]}-D")

vertex_ids = participant.set_mesh_vertices(MESH, coords)
print(f"[Thermal] preCICE {version.split(';')[0]}, {len(vertex_ids):,} vertices",
      flush=True)

# Neutronics is `first` and reads Temperature before this participant has run,
# so the initial field has to come from here. It is the same guess the internal
# driver starts from -- both couplings must start at the same point for their
# iterates to be comparable.
if participant.requires_initial_data():
    participant.write_data(MESH, "Temperature", vertex_ids,
                           gather(ctx.initial_temperature(), flat_idx))

participant.initialize()

trace = Trace(args.csv, ["iteration", "T_max", "T_mean", "power_max",
                         "source_watts", "sink_watts", "leakage_watts",
                         "balance", "cg_iterations"])
iteration = 0

while participant.is_coupling_ongoing():
    if participant.requires_writing_checkpoint():
        pass                      # the conduction solve carries no history

    dt = participant.get_max_time_step_size()
    p_vertices = participant.read_data(MESH, "Power", vertex_ids, dt)
    power = scatter(p_vertices, flat_idx, ctx.shape, 0.0)

    temperature, result = thermal_step(power, ctx)
    iteration += 1

    participant.write_data(MESH, "Temperature", vertex_ids,
                           gather(temperature, flat_idx))
    participant.advance(dt)

    if participant.requires_reading_checkpoint():
        pass

    trace.add(iteration=iteration, T_max=f"{temperature.max():.9f}",
              T_mean=f"{temperature.mean():.9f}",
              power_max=f"{float(np.max(power)):.9e}",
              source_watts=f"{result.source_watts:.6f}",
              sink_watts=f"{result.sink_watts:.6f}",
              leakage_watts=f"{result.leakage_watts:.6f}",
              balance=f"{result.balance_residual:.3e}",
              cg_iterations=result.iterations)
    print(f"[Thermal]    iter {iteration:3d}  T_max = {temperature.max():8.3f} K  "
          f"balance {result.balance_residual:.1e}  "
          f"({result.iterations} CG)", flush=True)

participant.finalize()
trace.write()
print(f"[Thermal] done after {iteration} coupling iterations", flush=True)
