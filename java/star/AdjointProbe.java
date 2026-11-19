import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import star.common.*;
import star.base.neo.*;
import star.base.report.*;
import star.material.*;
import star.flow.*;
import star.meshing.*;

/**
 * Phase 0 of wakegen: checks the STAR-CCM+ adjoint before the real calibration is built.
 *
 * Test case, built from -new: uniform inflow U along +x in a cylinder, one Gaussian
 * momentum source ("force blob") at x = blob_x_D*D, off the axis, and a cost function
 *
 *   J = volume integral over a slab at x = 0 of  mask * (u_x - c)^2,   c = U.
 *
 * The macro answers the README's phase-0 questions:
 *   1. Which adjoint models and field functions exist (all adjoint field functions are logged).
 *   2. Whether a volume integral of a user field function works as an adjoint cost
 *      function (cost A, analytic target), and whether the target may come from a
 *      table (cost B, same target read from target_probe.csv with interpolatePositionTable).
 *   3. Whether the adjoint runs with SST k-omega (frozen turbulence) on coupled flow.
 *      The 21.04 API has no segregated-flow adjoint, so coupled flow is used.
 *   4. Finite-difference check: J(A0 + dA) and J(A0 - dA) against the adjoint prediction
 *      dJ/dA = s * integral( adjoint_x * g ) with g the blob shape. The ratio fixes
 *      adjoint_scale (sign and factor); comparing the volume integral with the plain
 *      cell sum fixes adjoint_times_volume.
 *
 * Inputs next to the macro: probe.properties (optional; defaults below).
 * Outputs next to the macro: probe_result.properties, probe_log.csv, adjoint_probe_cells.csv,
 * probe_primal.sim, probe_adjoint.sim. Log markers: PROBE_MESH, PROBE_J, PROBE_ADJOINT_FF,
 * PROBE_FD, PROBE_COMPLETE, PROBE_FAILED.
 *
 * Written against the STAR-CCM+ 21.04.008 jars (compile-checked); not yet run in STAR.
 */
public class AdjointProbe extends StarMacro {

    private Simulation sim;
    private final Properties cfg = new Properties();
    private PrintWriter log;

    private double num(String key, double def) {
        return Double.parseDouble(cfg.getProperty(key, Double.toString(def)).trim());
    }

    private int integer(String key, int def) {
        return Integer.parseInt(cfg.getProperty(key, Integer.toString(def)).trim());
    }

    private String file(String name) {
        return resolvePath(name);
    }

    private static String f(double v) {
        return String.format(Locale.ROOT, "%.12g", v);
    }

    public void execute() {
        sim = getActiveSimulation();
        try {
            File props = new File(file("probe.properties"));
            if (props.exists()) {
                try (Reader in = new FileReader(props)) {
                    cfg.load(in);
                }
            }
            try (PrintWriter out = new PrintWriter(new FileWriter(file("probe_log.csv")))) {
                log = out;
                log.println("stage,amplitude,iterations,J_A,J_B");
                run();
            }
        } catch (Exception e) {
            sim.println("PROBE_FAILED: " + e);
            throw new RuntimeException(e);
        }
    }

    // ------------------------------------------------------------------ main sequence

    private double D, U, rho, a0;
    private Region region;
    private UserFieldFunction forceFunction;
    private VolumeIntegralReport costA, costB;
    private VolumeIntegralCostFunction costFunctionA, costFunctionB;
    private String blobShape;

    private void run() throws Exception {
        if (!sim.getRegionManager().getRegions().isEmpty()) {
            throw new IllegalStateException("Start with a new simulation (-new)");
        }
        D = num("diameter", 1.0);
        U = num("speed", 2.0);
        rho = num("density", 998.0);
        // Force amplitude [N/m3]; a negative value retards the flow.
        a0 = num("blob_amplitude_nd", -0.5) * rho * U * U / D;

        PhysicsContinuum physics = buildPhysics();
        buildMeshAndBoundaries(physics);
        buildForce();
        buildCosts();
        Map<String, String> result = new LinkedHashMap<String, String>();
        result.put("cells_note", "see PROBE_MESH in the log");
        result.put("amplitude_N_m3", f(a0));

        // 1. Primal at A0
        int maxIter = integer("primal_iterations", 3000);
        setAmplitude(a0);
        double[] j0 = solvePrimal("primal_A0", a0, maxIter);
        result.put("J_A", f(j0[0]));
        result.put("J_B", f(j0[1]));
        sim.saveState(file("probe_primal.sim"));

        // 2. Adjoint of cost A (and cost B if it was created)
        AdjointSolver adjoint = sim.getSolverManager().getSolver(AdjointSolver.class);
        int adjIter = integer("adjoint_iterations", 2000);
        boolean adjointA = runAdjoint(adjoint, costFunctionA, adjIter, "A", result);
        boolean adjointB = costFunctionB != null && runAdjoint(adjoint, costFunctionB, adjIter, "B", result);
        result.put("adjoint_A_ran", Boolean.toString(adjointA));
        result.put("adjoint_B_ran", Boolean.toString(adjointB));

        // 3. Adjoint field functions and the predicted dJ/dA for each
        Pattern pattern = Pattern.compile(cfg.getProperty("adjoint_ff_regex", "(?i).*adjoint.*"));
        List<FieldFunction> adjointFunctions = new ArrayList<FieldFunction>();
        for (FieldFunction ff : sim.getFieldFunctionManager().getObjects()) {
            String name = ff.getFunctionName() + " | " + ff.getPresentationName();
            if (pattern.matcher(name).matches()) {
                adjointFunctions.add(ff);
                sim.println("PROBE_ADJOINT_FF " + name + " | " + ff.getClass().getSimpleName());
            }
        }
        result.put("adjoint_field_function_count", Integer.toString(adjointFunctions.size()));
        List<String[]> predictions = new ArrayList<String[]>();
        int k = 0;
        for (FieldFunction ff : adjointFunctions) {
            String[] p = predictDerivative(ff, k++);
            if (p != null) {
                predictions.add(p);
            }
        }
        exportAdjointCells(adjointFunctions);
        sim.saveState(file("probe_adjoint.sim"));

        // 4. Finite differences, central, continued from the converged A0 flow
        double dA = num("fd_relative_step", 0.02) * Math.abs(a0);
        setAmplitude(a0 + dA);
        double[] jp = solvePrimal("primal_A0+dA", a0 + dA, maxIter);
        setAmplitude(a0 - dA);
        double[] jm = solvePrimal("primal_A0-dA", a0 - dA, maxIter);
        double fdA = (jp[0] - jm[0]) / (2 * dA);
        double fdB = (jp[1] - jm[1]) / (2 * dA);
        result.put("dA_N_m3", f(dA));
        result.put("fd_dJA_dA", f(fdA));
        result.put("fd_dJB_dA", f(fdB));
        result.put("fd_curvature_check", f((jp[0] - 2 * j0[0] + jm[0]) / (dA * dA)));
        sim.println("PROBE_FD dJA/dA=" + f(fdA) + " dJB/dA=" + f(fdB));
        for (String[] p : predictions) {
            // p = {name, volumeIntegral, cellSum}
            double vol = Double.parseDouble(p[1]);
            double sum = Double.parseDouble(p[2]);
            String key = "ff_" + p[0].replaceAll("[^A-Za-z0-9_]", "_");
            result.put(key + "_volume_integral", p[1]);
            result.put(key + "_cell_sum", p[2]);
            result.put(key + "_fd_over_volume_integral", vol != 0 ? f(fdA / vol) : "nan");
            result.put(key + "_fd_over_cell_sum", sum != 0 ? f(fdA / sum) : "nan");
        }
        setAmplitude(a0);
        writeResult(result);
        sim.println("PROBE_COMPLETE");
    }

    // ------------------------------------------------------------------ physics

    private PhysicsContinuum buildPhysics() {
        PhysicsContinuum physics = sim.getContinuumManager().createContinuum(PhysicsContinuum.class);
        physics.setPresentationName("Water - steady coupled SST, adjoint");
        physics.enable(star.metrics.ThreeDimensionalModel.class);
        physics.enable(SteadyModel.class);
        physics.enable(SingleComponentLiquidModel.class);
        physics.enable(star.coupledflow.CoupledFlowModel.class);
        physics.enable(ConstantDensityModel.class);
        physics.enable(star.turbulence.TurbulentModel.class);
        physics.enable(star.turbulence.RansTurbulenceModel.class);
        physics.enable(star.kwturb.KOmegaTurbulence.class);
        physics.enable(star.kwturb.SstKwTurbModel.class);
        physics.enable(star.kwturb.KwAllYplusWallTreatment.class);
        physics.enable(AdjointModel.class);
        physics.enable(star.coupledflow.AdjointFlowModel.class);
        physics.enable(star.turbulence.AdjointFrozenTurbulenceModel.class);
        Liquid water = (Liquid) physics.getModelManager().getModel(SingleComponentLiquidModel.class).getMaterial();
        ((ConstantMaterialPropertyMethod) water.getMaterialProperties().getMaterialProperty(ConstantDensityProperty.class).getMethod())
                .getQuantity().setValue(rho);
        ((ConstantMaterialPropertyMethod) water.getMaterialProperties().getMaterialProperty(DynamicViscosityProperty.class).getMethod())
                .getQuantity().setValue(num("viscosity", 1.0e-3));
        physics.getInitialConditions().get(VelocityProfile.class).getMethod(ConstantVectorProfileMethod.class)
                .getQuantity().setComponents(U, 0, 0);
        return physics;
    }

    private void buildMeshAndBoundaries(PhysicsContinuum physics) {
        double xIn = -4 * D, xOut = 4 * D;
        SimpleCylinderPart domain = sim.get(MeshPartFactory.class).createNewCylinderPart(sim.get(SimulationPartManager.class));
        domain.setPresentationName("Probe domain");
        domain.setStartCoordinate(new DoubleVector(new double[] {xIn, 0, 0}));
        domain.setEndCoordinate(new DoubleVector(new double[] {xOut, 0, 0}));
        domain.setRadius(2 * D);
        domain.rebuildSimpleShapePart();
        domain.getPartSurfaceManager().splitPartSurfacesByAngle(domain.getPartSurfaceManager().getPartSurfaces(), 45);
        sim.getRegionManager().newRegionsFromParts(Collections.<GeometryPart>singletonList(domain), "OneRegion", null,
                "OneBoundaryPerPartSurface", null, "OneFeatureCurve", null, RegionManager.CreateInterfaceMode.BOUNDARY);
        region = sim.getRegionManager().getRegions().iterator().next();
        region.setPresentationName("Probe fluid");
        region.setPhysicsContinuum(physics);

        double cell = num("base_size_D", 0.1) * D;
        double fine = num("fine_size_D", 0.02) * D;
        AutoMeshOperation mesh = sim.get(MeshOperationManager.class).createAutoMeshOperation(
                Arrays.asList("star.resurfacer.ResurfacerAutoMesher", "star.trimmer.TrimmerAutoMesher"),
                Collections.<GeometryPart>singletonList(domain));
        mesh.getDefaultValues().get(BaseSize.class).setValue(cell);
        SimpleCylinderPart refinement = sim.get(MeshPartFactory.class).createNewCylinderPart(sim.get(SimulationPartManager.class));
        refinement.setPresentationName("Blob to cost slab refinement");
        refinement.setStartCoordinate(new DoubleVector(new double[] {(num("blob_x_D", -1.0) - 0.4) * D, 0, 0}));
        refinement.setEndCoordinate(new DoubleVector(new double[] {0.3 * D, 0, 0}));
        refinement.setRadius(0.7 * D);
        refinement.rebuildSimpleShapePart();
        VolumeCustomMeshControl control = mesh.getCustomMeshControls().createVolumeControl();
        control.setPresentationName("Fine cells");
        control.getGeometryObjects().setObjects(refinement);
        control.getCustomConditions().get(star.trimmer.VolumeControlTrimmerSizeOption.class).setVolumeSourceTrimmerSizeOption(true);
        control.getCustomValues().get(VolumeControlSize.class).setAbsoluteSize(fine, sim.getUnitsManager().getObject("m"));
        mesh.execute();

        AreaAverageReport position = sim.getReportManager().createReport(AreaAverageReport.class);
        position.setPresentationName("Boundary x position");
        position.setFieldFunction(sim.getFieldFunctionManager().getFunction("Position").getComponentFunction(0));
        int inlets = 0, outlets = 0;
        for (Boundary b : region.getBoundaryManager().getBoundaries()) {
            position.getParts().setObjects(b);
            double x = position.getReportMonitorValue();
            if (Math.abs(x - xIn) < 0.01 * D) {
                b.setPresentationName("Inlet");
                b.setBoundaryType(InletBoundary.class);
                b.getValues().get(VelocityMagnitudeProfile.class).getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(U);
                inlets++;
            } else if (Math.abs(x - xOut) < 0.01 * D) {
                b.setPresentationName("Outlet");
                b.setBoundaryType(PressureBoundary.class);
                outlets++;
            } else {
                b.setPresentationName("Far field slip");
                b.setBoundaryType(SymmetryBoundary.class);
            }
        }
        if (inlets != 1 || outlets != 1) {
            throw new IllegalStateException("Could not identify inlet/outlet (" + inlets + ", " + outlets + ")");
        }
        double cfl = num("coupled_cfl", 20.0);
        sim.getSolverManager().getSolver(star.coupledflow.CoupledImplicitSolver.class).setCFL(cfl);
        sim.println("PROBE_MESH base_m=" + f(cell) + " fine_m=" + f(fine) + " cfl=" + f(cfl));
    }

    // ------------------------------------------------------------------ force blob and costs

    private UserFieldFunction fieldFunction(String name, FieldFunctionTypeOption.Type type, String definition) {
        FieldFunctionManager mgr = sim.getFieldFunctionManager();
        UserFieldFunction ff = mgr.has(name) ? (UserFieldFunction) mgr.getFunction(name) : mgr.createFieldFunction();
        ff.setPresentationName(name);
        ff.setFunctionName(name);
        ff.getTypeOption().setSelected(type);
        ff.setDefinition(definition);
        return ff;
    }

    private void buildForce() {
        double x0 = num("blob_x_D", -1.0) * D;
        double y0 = num("blob_y_D", 0.25) * D;
        double s = num("blob_sigma_D", 0.1) * D;
        blobShape = "exp(-(pow($$Position[0]-(" + f(x0) + "),2)+pow($$Position[1]-(" + f(y0) + "),2)+pow($$Position[2],2))/"
                + f(s * s) + ")";
        fieldFunction("ProbeBlobShape", FieldFunctionTypeOption.Type.SCALAR, blobShape);
        forceFunction = fieldFunction("ProbeForce", FieldFunctionTypeOption.Type.VECTOR, "[0, 0, 0]");
        region.getConditions().get(MomentumUserSourceOption.class).setSelected(MomentumUserSourceOption.Type.SPECIFIED);
        MomentumUserSource source = region.getValues().get(MomentumUserSource.class);
        source.setMethod(FunctionVectorProfileMethod.class);
        source.getMethod(FunctionVectorProfileMethod.class).setFieldFunction(forceFunction);
    }

    private void setAmplitude(double a) {
        forceFunction.setDefinition("[" + f(a) + "*$ProbeBlobShape, 0, 0]");
    }

    private void buildCosts() throws IOException {
        double h = num("cost_slab_thickness_D", 0.04) * D;
        double rMax = num("cost_radius_D", 0.6) * D;
        String mask = "((abs($$Position[0]) < " + f(h / 2) + " && sqrt(pow($$Position[1],2)+pow($$Position[2],2)) < "
                + f(rMax) + ") ? 1 : 0)";
        fieldFunction("ProbeCostMask", FieldFunctionTypeOption.Type.SCALAR, mask);
        fieldFunction("ProbeCostA", FieldFunctionTypeOption.Type.SCALAR,
                "$ProbeCostMask*pow($$Velocity[0]-(" + f(U) + "),2)");
        costA = volumeIntegral("Probe cost A", sim.getFieldFunctionManager().getFunction("ProbeCostA"));
        AdjointSolver adjoint = sim.getSolverManager().getSolver(AdjointSolver.class);
        costFunctionA = adjoint.getAdjointCostFunctionManager().createAdjointCostFunction(VolumeIntegralCostFunction.class);
        costFunctionA.setPresentationName("Probe cost A");
        costFunctionA.setVolumeIntegralReport(costA);

        // Cost B: the same target, read from a table (tests a table-driven target).
        try {
            writeTargetTable(h, rMax);
            Table table = sim.getTableManager().createFromFile(file("target_probe.csv"));
            table.setPresentationName("ProbeTarget");
            String lookup = cfg.getProperty("table_lookup", "interpolatePositionTable(@Table(\"ProbeTarget\"), \"utx\")");
            fieldFunction("ProbeCostB", FieldFunctionTypeOption.Type.SCALAR,
                    "$ProbeCostMask*pow($$Velocity[0]-" + lookup + ",2)");
            costB = volumeIntegral("Probe cost B", sim.getFieldFunctionManager().getFunction("ProbeCostB"));
            costFunctionB = adjoint.getAdjointCostFunctionManager().createAdjointCostFunction(VolumeIntegralCostFunction.class);
            costFunctionB.setPresentationName("Probe cost B");
            costFunctionB.setVolumeIntegralReport(costB);
        } catch (Exception e) {
            sim.println("PROBE_COST_B_FAILED " + e);
            costB = null;
            costFunctionB = null;
        }
    }

    private VolumeIntegralReport volumeIntegral(String name, FieldFunction ff) {
        VolumeIntegralReport r = sim.getReportManager().createReport(VolumeIntegralReport.class);
        r.setPresentationName(name);
        r.setFieldFunction(ff);
        r.getParts().setObjects(region);
        return r;
    }

    /** Target u_x = U on a point grid covering the cost slab (X,Y,Z,utx). */
    private void writeTargetTable(double h, double rMax) throws IOException {
        int nx = 5, ny = 41;
        try (PrintWriter out = new PrintWriter(new FileWriter(file("target_probe.csv")))) {
            out.println("X,Y,Z,utx");
            for (int i = 0; i < nx; i++) {
                double x = -h + 2 * h * i / (nx - 1);
                for (int j = 0; j < ny; j++) {
                    double y = -1.2 * rMax + 2.4 * rMax * j / (ny - 1);
                    for (int k = 0; k < ny; k++) {
                        double z = -1.2 * rMax + 2.4 * rMax * k / (ny - 1);
                        out.println(f(x) + "," + f(y) + "," + f(z) + "," + f(U));
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------ solves

    /** Converge the flow; returns {J_A, J_B}. Stops when J_A changes less than primal_tolerance over 100 iterations. */
    private double[] solvePrimal(String stage, double amplitude, int maxIter) {
        for (SolverStoppingCriterion c : sim.getSolverStoppingCriterionManager().getObjects()) {
            c.setIsUsed(false);
        }
        double tol = num("primal_tolerance", 1.0e-6);
        int block = 100, done = 0;
        double previous = Double.NaN, jA = Double.NaN;
        while (done < maxIter) {
            sim.getSimulationIterator().step(block);
            done += block;
            jA = costA.getReportMonitorValue();
            if (!Double.isNaN(previous) && Math.abs(jA - previous) <= tol * Math.max(Math.abs(jA), 1e-30)) {
                break;
            }
            previous = jA;
        }
        double jB = costB != null ? costB.getReportMonitorValue() : Double.NaN;
        log.println(stage + "," + f(amplitude) + "," + done + "," + f(jA) + "," + f(jB));
        log.flush();
        sim.println("PROBE_J " + stage + " iterations=" + done + " J_A=" + f(jA) + " J_B=" + f(jB));
        return new double[] {jA, jB};
    }

    private boolean runAdjoint(AdjointSolver adjoint, AdjointCostFunction cost, int iterations, String tag,
            Map<String, String> result) {
        try {
            adjoint.runAdjoint(Collections.singletonList(cost), iterations);
            sim.println("PROBE_ADJOINT_" + tag + " done");
            return true;
        } catch (Exception e) {
            sim.println("PROBE_ADJOINT_" + tag + "_FAILED " + e);
            result.put("adjoint_" + tag + "_error", e.toString().replace('\n', ' '));
            return false;
        }
    }

    /** {name, integral(adj_x*g dV), sum(adj_x*g)} for one vector adjoint field function, or null. */
    private String[] predictDerivative(FieldFunction ff, int k) {
        try {
            String name = ff.getFunctionName();
            UserFieldFunction product = fieldFunction("ProbeAdjDotG" + k, FieldFunctionTypeOption.Type.SCALAR,
                    "$${" + name + "}[0]*$ProbeBlobShape");
            VolumeIntegralReport vol = volumeIntegral("Probe adj.g volume " + k, product);
            SumReport sum = sim.getReportManager().createReport(SumReport.class);
            sum.setPresentationName("Probe adj.g sum " + k);
            sum.setFieldFunction(product);
            sum.getParts().setObjects(region);
            String[] p = {name, f(vol.getReportMonitorValue()), f(sum.getReportMonitorValue())};
            sim.println("PROBE_ADJOINT_FF_PREDICTION " + name + " volume_integral=" + p[1] + " cell_sum=" + p[2]);
            return p;
        } catch (Exception e) {
            // scalar adjoint functions (or ones without a [0] component) end up here
            sim.println("PROBE_ADJOINT_FF_SKIPPED " + ff.getFunctionName() + " " + e);
            return null;
        }
    }

    /** Cell table X, Y, Z, Volume and every adjoint field function, in the format wakegen will read. */
    private void exportAdjointCells(List<FieldFunction> adjointFunctions) {
        try {
            XyzInternalTable table = sim.getTableManager().createTable(XyzInternalTable.class);
            table.setPresentationName("Probe adjoint cells");
            List<FieldFunction> columns = new ArrayList<FieldFunction>();
            columns.add(sim.getFieldFunctionManager().getFunction("Volume"));
            columns.add(sim.getFieldFunctionManager().getFunction("ProbeBlobShape"));
            columns.addAll(adjointFunctions);
            table.setFieldFunctions(columns);
            table.getParts().setObjects(region);
            table.extract();
            table.export(file("adjoint_probe_cells.csv"), ",");
        } catch (Exception e) {
            sim.println("PROBE_EXPORT_FAILED " + e);
        }
    }

    private void writeResult(Map<String, String> result) throws IOException {
        try (PrintWriter out = new PrintWriter(new FileWriter(file("probe_result.properties")))) {
            out.println("# AdjointProbe result (phase 0). fd_* are central differences in J per N/m3 of amplitude.");
            out.println("# adjoint_scale = fd_dJA_dA / volume_integral (or / cell_sum if that ratio is the clean one).");
            for (Map.Entry<String, String> e : result.entrySet()) {
                out.println(e.getKey() + " = " + e.getValue());
            }
        }
    }
}
