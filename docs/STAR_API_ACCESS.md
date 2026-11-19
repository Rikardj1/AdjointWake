# STAR-CCM+ 21.04 API access

How the STAR-CCM+ 21.04.008 Javadoc and jar files are reached when writing the
macros (`AdjointProbe`, `CalibrateWake`, `ApplyWakeForces`), and what has been
checked against them so far.

## Sources

| Source | Where | Use |
|---|---|---|
| Javadoc | `C:\Users\Rikard\StarCCM_API\21.04.008\javadoc\html` (1.3 GB, 134 packages under `star/`) | Class and method descriptions. Search `type-search-index.js` (class names) and `member-search-index.js` (methods) first, then open the class page. |
| Jar files | `C:\Users\Rikard\StarCCM_API\21.04.008\java\platform` (469 jars, e.g. `modules/ext/starbase.jar`, `coupledflow.jar`, `turbulence.jar`) | Exact signatures with `javap`; compile check of macros. |
| Proven macros | `C:\Users\Rikard\StarCCM_API\21.04.008\macros` (`RunBEMPoint.java`, `auv_crp_balance/`) and the project's other macros (`CrpMixingPlane.java`, `RootSwirlSource.java`) | Idioms for physics, mesh, reports and solve loops. |

A cloud session copies the `java` folder over as one archive
(`adjointwake_star_jars.tgz`, 353 MB) and runs `javap` on it.

## Compile check

The STAR 21.04 classes are Java class version 69 (Java 25). A JDK 21 `javap`
reads them, but `javac` older than 25 refuses them. Compile checks therefore need
a Java 25 compiler (JDK 25, or the Eclipse compiler `ecj` 3.43+), with every jar
under `java\platform` on the class path:

```
javac -proc:none -cp "<all jars under java\platform>" -d build X.java
```

Compiling is a check only; STAR compiles the macro itself at run time.

## Checked with javap (2026-11-19)

Adjoint classes that exist in 21.04.008:

- `star.common`: `AdjointModel`, `AdjointSolver` (`runAdjoint()`, `runAdjoint(int)`,
  `stepAdjoint(int)`, `clearAdjoint()`, `setFrozen(boolean)`,
  `getAdjointCostFunctionManager()`), `AdjointSteadySolver`,
  `AdjointCostFunctionManager` (`createAdjointCostFunction(Class)`,
  `getAdjointCostFunction(String)`), `ReportCostFunction`,
  `AdjointResidualStoppingCriterion`.
- `star.coupledflow`: `AdjointFlowModel`, `AdjointFlowSolver`,
  `VolumeIntegralCostFunction` (`setVolumeIntegralReport(VolumeIntegralReport)`),
  `VolumeAverageCostFunction`, `ExpressionCostFunction`, `SumCostFunction`.
- `star.turbulence`: `AdjointFrozenTurbulenceModel`. `star.saturb`: `AdjointSaTurbModel`.
- `star.flow`: `AdjointFrozenFlowModel`.

Consequences for Phase 0 (README):

1. The adjoint flow model lives only in `star.coupledflow`; `segregatedflow.jar`
   has no adjoint classes. Plan for **coupled flow** in the calibration case.
2. There is no SST k-omega adjoint model; SST runs with **frozen turbulence**
   (`AdjointFrozenTurbulenceModel`). Spalart-Allmaras has a full adjoint.
3. A `VolumeIntegralCostFunction` wraps a `VolumeIntegralReport`, so a volume
   integral of a user field function can be the cost function as far as the API
   goes. Whether the field function may read the target from a table, and the
   name of the adjoint momentum field function, are not in the Javadoc and must
   be checked in a STAR run.
