# wakegen

Calibrates a body-force field about one diameter upstream of a propeller so that
the nominal wake in the propeller plane matches a prescribed 3-D wake
(axial, tangential and radial components). The forces are found with an
adjoint-based optimisation in STAR-CCM+; this package is everything around the
flow solver.

## Status

| Part | State |
|---|---|
| Python core (wake reader, target field, force parametrisation, cost, L-BFGS, file protocol, report) | Written, 27 tests pass |
| `java/WakeLoop.java` (solver-independent loop driver) | Written, tested against the Python side with a plain JDK |
| STAR-CCM+ macros | `AdjointProbe` ran (phase 0 passed). `CalibrateWake` written and compile-checked, not run yet. `ApplyWakeForces` not written. API access: [docs/STAR_API_ACCESS.md](docs/STAR_API_ACCESS.md) |
| Any run in STAR-CCM+ | Phase 0 (`AdjointProbe`) passed 2026-11-19, see below |

The dry run below uses a simple stand-in for the flow solver. It shows that the
loop, the gradient chain and the files work. It says nothing about how well a
real RANS flow will follow the target.

## Method

- **Calibrate without the propeller**, against the nominal wake. The forces are
  then frozen and added to the propeller simulation.
- **Design variables:** force per unit area `F(r, theta)` (three Cartesian
  components) on a polar grid in the force disk, times a Gaussian in x. The force
  density in a cell is `f = g(x) * window(r) * interp(F)`. The table is
  mesh-independent, so it can be evaluated on the propeller simulation's mesh.
- **Cost:** volume integral over a thin cell slab at the propeller plane of
  `w_a du_a^2 + w_t du_t^2 + w_r du_r^2`, `du = u - u_target`. Normalised, `sqrt(J)`
  is an RMS error in fractions of the ship speed.
- **Gradient:** the solver's adjoint momentum in the force-disk cells, times
  cell volume, projected onto `F` with the transpose of the map `F -> f`.
- **Optimiser:** L-BFGS with a backtracking line search. One accepted iteration
  costs one flow solve and one adjoint solve; a rejected trial costs a flow solve.
- **Start:** actuator-disk estimate, `F_x = rho/2 (u_x^2 - U^2)`,
  `F_t = rho u_disk u_t`.
- **Uniform cross-flow:** the disk-mean transverse velocity of the target is set
  through the inflow direction (`crossflow = inlet`), not with forces, because a
  uniform cross-flow carries no vorticity. The force disk is offset against the
  resulting sideways drift so the disturbance arrives centred on the propeller.

What forces 1 D upstream cannot produce (mainly the in-plane divergence part of
the transverse velocity, and very deep axial deficits) shows up as residual per
component in the report.

## Wake file

`X,Y,Z,Vx,Vy,Vz`, one row per point in the propeller plane, points on
concentric rings around the shaft axis (see `examples/WakeEdStar.csv`: 6 rings
of 31 points). X is the shaft axis pointing downstream. The file's axes are used
as the simulation's laboratory axes, so there is no angle or sign convention to
set. A file made for a propeller of diameter `wake_diameter` (e.g. 1.0 m, i.e.
coordinates in units of D) is scaled by `diameter / wake_diameter`; with
`wake_diameter = 0` the coordinates are in metres. `wake_coord_scale` is an extra
factor (e.g. 0.001 for mm). Velocities are
V/Vs when `wake_velocity_normalised = true`.

## Quick start

```bash
pip install -e .[plot,test]
```

```bash
python -m pytest
```

Summary of a wake and the inflow vector it implies:

```bash
python -m wakegen info --case examples/case.properties
```

Dry run of the whole loop without STAR-CCM+ (surrogate solver, Java driver):

```bash
javac -d build java/WakeLoop.java java/DryRunLoop.java
```

```bash
mkdir -p dry && cp examples/case.properties examples/WakeEdStar.csv dry/ && java -cp build DryRunLoop dry/case.properties python3 "$PWD/src" 0.06
```

`examples/case.properties` has `ship_speed = 5.0` as a placeholder.

## File protocol between the macro and Python

All files live next to `case.properties`.

| File | Written by | Content |
|---|---|---|
| `control_cells.csv` | macro, once after meshing | `X, Y, Z, Volume` of cells around the force disk |
| `target_cells.csv` | macro, once after meshing | `X, Y, Z, Volume` of cells around the propeller plane |
| `inlet.properties` | `prepare` | `inlet_vx, inlet_vy, inlet_vz` [m/s] |
| `target_table.csv` | `prepare` | `X,Y,Z,utx,uty,utz,wxx,wyy,wzz,wyz` per target cell |
| `forces_table.csv` | `prepare`, `step` | `X,Y,Z,fx,fy,fz` [N/m3] per control cell |
| `action.properties` | `prepare`, `step` | `action = primal_adjoint \| primal \| adjoint \| stop` |
| `achieved_cells.csv` | macro, after each flow solve | `X, Y, Z, Volume, Velocity[i], Velocity[j], Velocity[k]` of the target cells |
| `adjoint_cells.csv` | macro, after each adjoint solve | `X, Y, Z, Volume`, adjoint momentum `[i],[j],[k]` of the control cells |
| `state.npz`, `history.csv` | `step` | optimiser state (restartable) and log |
| `wake_forces.json` | `step` | best force field so far; the deliverable |
| `wake_achieved.csv`, `_summary.txt`, `_cuts.png` | `report` | achieved wake in the input format plus `Vt`, `Vr` (tangential along θ from +Z towards +Y, radial outwards), errors, plots |

Headers are matched by pattern after removing quotes and units, so STAR's
`"X (m)"` and `"Velocity[i] (m/s)"` are accepted. The adjoint columns are found
with the regex `adjoint_column` in `case.properties`.

The solver's adjoint cost function must be the volume integral over the region of

```
wxx*(ux-utx)^2 + wyy*(uy-uty)^2 + wzz*(uz-utz)^2 + 2*wyz*(uy-uty)*(uz-utz)
```

with the table columns mapped onto the cells. The weights already contain the
mask of the cost region and the component weights, so the field function needs
no trigonometry and no local coordinate system.

Loop (implemented in `java/WakeLoop.java`):

| action | macro does | then |
|---|---|---|
| `primal_adjoint` | reload forces, converge flow, export achieved, converge adjoint, export adjoint | `wakegen step` |
| `primal` | reload forces, converge flow, export achieved | `wakegen step` |
| `adjoint` | converge adjoint on the current flow, export adjoint | `wakegen step` |
| `stop` | reload forces, converge flow, export achieved | `wakegen report` |

## Phase 0: what must be verified in STAR-CCM+ 21.04 first

A small test case (uniform inflow, one Gaussian force blob, cost = volume
integral of an indicator times `(u_x - c)^2`) has to answer:

1. The adjoint classes and field-function names in 21.04 (from the Javadoc).
2. Whether a volume report on a user field function can be an adjoint cost
   function, and whether the target may come from a table. Fallback: an analytic
   target expression generated by Python.
3. Whether the adjoint runs with segregated flow and SST k-omega (frozen
   turbulence). Fallback: coupled flow.
4. **Finite-difference check:** perturb the force in a patch, re-converge,
   compare the change in J with the adjoint's prediction. This fixes
   `adjoint_scale` (sign and factor) and `adjoint_times_volume`.

The test is `java/star/AdjointProbe.java` (settings in `java/star/probe.properties`).
Copy both into an empty folder and run, without `-power` if there is no Power licence:

```
"C:\Program Files\Siemens\21.04.008\STAR-CCM+21.04.008\star\bin\starccm+.bat" -batch AdjointProbe.java -new -np 8 -licpath 1999@hpcswkrilic01
```

In PowerShell, put `& ` in front of the quoted path.

It writes `probe_result.properties` (J, finite-difference dJ/dA, the adjoint prediction for
every adjoint field function and their ratios), `probe_log.csv`, `adjoint_probe_cells.csv`,
`probe_primal.sim` and `probe_adjoint.sim`. Send these plus the STAR log. Compile check
without STAR: `java\star\compile_check.bat` (needs JDK 25).

### Phase 0 result (STAR-CCM+ 21.04.008, 2026-11-19)

| Question | Answer |
|---|---|
| 1. Classes and field functions | Adjoint flow only with coupled flow. Per cost function STAR creates `Adjoint<n>::VelocityAdjoint` (vector) and `Adjoint<n>::CoordSensitivity`. |
| 2. Cost function | A `ReportCostFunction` on a `VolumeIntegralReport` of a user field function works. A target read from a table with `interpolatePositionTable(@Table(...), "utx")` gives the same J and the same adjoint (cost B = cost A to all digits). |
| 3. Turbulence | Runs with SST k-omega and frozen turbulence on coupled flow. |
| 4. Finite differences | Central difference (2 % force step) / volume integral of `VelocityAdjoint[0] * g` = **1.0006**; the plain cell sum is off by 5e5. So `adjoint_scale = 1`, `adjoint_times_volume = true`, `adjoint_column = .*velocityadjoint`. |

The response is close to linear in this test: the second-order term J'' dA^2/2 is about 1 % of the first-order change J' dA for a 2 % force step.

If 2 or 4 fails, the same Python core can be driven by a response-matrix method
in STAR-CCM+ or by an adjoint in OpenFOAM instead.

## Calibration in STAR-CCM+

Put `case.properties`, the wake file and `java/star/CalibrateWake.java` in one folder. Set
`python` and `wakegen_path` in `case.properties` (for a clone in `C:\Programs\Repo\AdjointWake`:
`wakegen_path = C:\Programs\Repo\AdjointWake\src`, and `pip install numpy matplotlib` once), then:

```
& "C:\Program Files\Siemens\21.04.008\STAR-CCM+21.04.008\star\bin\starccm+.bat" -batch CalibrateWake.java -new -np 8 -licpath 1999@hpcswkrilic01
```

The macro builds a cylinder around the propeller plane (no propeller, no hub), meshes it with a
fine zone from the force disk to the propeller plane, writes the cell files, runs `wakegen
prepare`, and then loops flow solve, adjoint solve and `wakegen step` until the optimiser stops;
`wakegen report` writes the achieved wake. `history.csv` shows the progress. `calibration.sim`
is saved after every adjoint solve; open it and run the macro again without `-new` to continue.

If the flow solve of a line-search trial diverges (e.g. "non-finite residual (Sdr)"), the macro
clears and re-initialises the solution, halves the coupled CFL (not below `coupled_cfl_min`),
and runs `wakegen step --failed`, which rejects the trial and halves the step. After 4 failures
in a row, or if the starting point diverges, it stops. To skip a trial that already crashed a
run, run `python -m wakegen step --failed --case case.properties` once before the restart.
`coupled_cfl` from case.properties is also applied on a restart.
The macro's own settings (domain, cell sizes, iterations) are at the end of
`examples/case.properties`. `java/WakeLoop.java` is embedded in the macro: after changing it,
run `python tools/embed_wakeloop.py`.

## Using the result in the propeller simulation

```bash
python -m wakegen apply --forces wake_forces.json --cells cells.csv --out forces_table.csv --prop-plane-x 0.0
```

`cells.csv` holds `X, Y, Z, Volume` of the stationary-region cells around the
force disk in the propeller simulation. The force zone must lie outside the
rotating region. The inflow must be the vector in `wake_forces.json`
(`freestream`). `--ship-speed` and `--density` rescale the forces with
`rho Vs^2`; that is exact only as far as the wake shape does not depend on
Reynolds number.

## Layout

```
src/wakegen/   config, tables, wake_io, wake_field, control, initial_guess,
               objective, optimizer, problem, report, cli, surrogate (dry run only)
java/          WakeLoop.java (driver), DryRunLoop.java (dry-run harness)
tests/         pytest suite, including the Java driver against the surrogate
examples/      WakeEdStar.csv, case.properties
```
