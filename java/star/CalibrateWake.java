import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import star.common.*;
import star.base.neo.*;
import star.base.report.*;
import star.material.*;
import star.flow.*;
import star.meshing.*;
import star.vis.ThresholdPart;
import star.vis.ThresholdMode;

/**
 * Calibrates the wake body-force field in STAR-CCM+ (no propeller, nominal wake).
 *
 * Run from the folder that holds case.properties and the wake file:
 *   starccm+ -batch CalibrateWake.java -new -np N [-licpath ...]
 *
 * What it does:
 *   1. Builds a cylindrical domain along x (flow +x) around the propeller plane, coupled flow,
 *      SST k-omega, adjoint with frozen turbulence (the only adjoint STAR 21.04 has for SST).
 *      Meshes it with a fine zone from the force disk to the propeller plane.
 *   2. Writes control_cells.csv and target_cells.csv (cells around the force disk and around
 *      the propeller plane) and runs "python -m wakegen prepare".
 *   3. Sets the inflow from inlet.properties, reads forces_table.csv as the momentum source and
 *      target_table.csv as the target of the cost function
 *        J = integral( wxx dux^2 + wyy duy^2 + wzz duz^2 + 2 wyz duy duz ) dV,  du = u - ut.
 *   4. Runs the loop of WakeLoop (nested below, a copy of java/WakeLoop.java): flow solve,
 *      achieved_cells.csv, adjoint solve, adjoint_cells.csv, "wakegen step", until the
 *      optimiser says stop; then "wakegen report".
 *
 * Restart: open calibration.sim (saved after every solve) and run the macro again without
 * -new; it then skips steps 1-3 and continues from state.npz / action.properties.
 *
 * Keys read from case.properties besides the wakegen ones (all optional):
 *   python (python), wakegen_path (folder holding the wakegen package; empty = installed),
 *   domain_inlet_D (-4), domain_outlet_D (4), domain_radius_D (3), side_boundary (symmetry | inlet),
 *   base_size_D (0.1), fine_size_D (0.02), coupled_cfl (20), viscosity (1.0e-3),
 *   primal_min_iterations (300), primal_max_iterations (3000), primal_block (50),
 *   primal_tolerance (1e-5, relative change of J per block), primal_stable_blocks (3),
 *   adjoint_iterations (1000), max_solves (200), adjoint_ff_regex ((?i).*VelocityAdjoint.*).
 *
 * Log markers: WAKE_MESH, WAKE_PREPARED (from Python), WAKE_SOLVE, WAKE_ITER (from Python),
 * WAKE_ADJOINT_FF, WAKE_COMPLETE, WAKE_FAILED.
 *
 * Phase 0 (AdjointProbe, 2026-11-19) fixed: ReportCostFunction on a VolumeIntegralReport works,
 * table targets work, dJ/df_cell = VelocityAdjoint * cell volume.
 */
public class CalibrateWake extends StarMacro {

    private static final String REGION = "Wake calibration fluid";

    private Simulation sim;
    private final Properties cfg = new Properties();
    private File caseFile;

    private double D, R, U, rho, propX, forceX, sigma;
    private Region region;
    private PhysicsContinuum physics;
    private ThresholdPart controlPart, targetPart;
    private VolumeIntegralReport costReport;
    private ReportCostFunction costFunction;
    private FieldFunction adjointField;
    private int tableCount = 0;
    private Table forcesTable;

    private String text(String key, String def) {
        return cfg.getProperty(key, def).trim();
    }

    private double num(String key, double def) {
        return Double.parseDouble(text(key, Double.toString(def)));
    }

    private int integer(String key, int def) {
        return (int) Math.round(num(key, def));
    }

    private static String f(double v) {
        return String.format(Locale.ROOT, "%.12g", v);
    }

    public void execute() {
        sim = getActiveSimulation();
        try {
            caseFile = new File(resolvePath("case.properties")).getAbsoluteFile();
            try (Reader in = new FileReader(caseFile)) {
                cfg.load(in);
            }
            run();
        } catch (Exception e) {
            sim.println("WAKE_FAILED: " + e);
            throw new RuntimeException(e);
        }
    }

    private File caseFile(String name) {
        return new File(caseFile.getParentFile(), name);
    }

    // ------------------------------------------------------------------ main sequence

    private void run() throws Exception {
        D = num("diameter", 1.0);
        R = D / 2;
        U = num("ship_speed", 1.0);
        rho = num("density", 998.0);
        propX = num("prop_plane_x", 0.0);
        forceX = propX + num("force_plane_x_D", -1.0) * D;
        sigma = num("ctrl_sigma_D", 0.03) * D;

        StarLoop loop = new StarLoop(caseFile, text("python", "python"), text("wakegen_path", ""));
        boolean resume = !sim.getRegionManager().getRegions().isEmpty();
        if (resume) {
            attach();
            sim.println("WAKE_RESUME continuing from " + caseFile("action.properties"));
        } else {
            buildPhysics();
            buildMeshAndBoundaries();
            buildZones();
            exportCells(controlPart, "control_cells.csv", new ArrayList<FieldFunction>());
            exportCells(targetPart, "target_cells.csv", new ArrayList<FieldFunction>());
            loop.wakegen("prepare");
            setInflow();
            loadForces();
            buildCost();
            sim.saveState(caseFile("calibration.sim").getPath());
        }
        String message = loop.run(integer("max_solves", 200));
        sim.saveState(caseFile("calibration.sim").getPath());
        sim.println("WAKE_COMPLETE " + message);
    }

    /** Find the objects of a simulation saved by an earlier run of this macro. */
    private void attach() {
        region = sim.getRegionManager().getRegion(REGION);
        physics = (PhysicsContinuum) region.getPhysicsContinuum();
        controlPart = (ThresholdPart) sim.getPartManager().getPart("Wake control cells");
        targetPart = (ThresholdPart) sim.getPartManager().getPart("Wake target cells");
        costReport = (VolumeIntegralReport) sim.getReportManager().getReport("Wake cost");
        AdjointSolver adjoint = sim.getSolverManager().getSolver(AdjointSolver.class);
        costFunction = (ReportCostFunction) adjoint.getAdjointCostFunctionManager().getAdjointCostFunction("Wake cost");
        for (Table t : sim.getTableManager().getObjects()) {
            if (t.getPresentationName().startsWith("WakeForces_")) {
                forcesTable = t;
                tableCount = Math.max(tableCount, Integer.parseInt(t.getPresentationName().substring(11)));
            }
        }
    }

    // ------------------------------------------------------------------ set-up

    private void buildPhysics() {
        physics = sim.getContinuumManager().createContinuum(PhysicsContinuum.class);
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
    }

    private double xIn, xOut;

    private void buildMeshAndBoundaries() {
        xIn = propX + num("domain_inlet_D", -4.0) * D;
        xOut = propX + num("domain_outlet_D", 4.0) * D;
        SimpleCylinderPart domain = sim.get(MeshPartFactory.class).createNewCylinderPart(sim.get(SimulationPartManager.class));
        domain.setPresentationName("Wake calibration domain");
        domain.setStartCoordinate(new DoubleVector(new double[] {xIn, 0, 0}));
        domain.setEndCoordinate(new DoubleVector(new double[] {xOut, 0, 0}));
        domain.setRadius(num("domain_radius_D", 3.0) * D);
        domain.rebuildSimpleShapePart();
        domain.getPartSurfaceManager().splitPartSurfacesByAngle(domain.getPartSurfaceManager().getPartSurfaces(), 45);
        sim.getRegionManager().newRegionsFromParts(Collections.<GeometryPart>singletonList(domain), "OneRegion", null,
                "OneBoundaryPerPartSurface", null, "OneFeatureCurve", null, RegionManager.CreateInterfaceMode.BOUNDARY);
        region = sim.getRegionManager().getRegions().iterator().next();
        region.setPresentationName(REGION);
        region.setPhysicsContinuum(physics);

        double cell = num("base_size_D", 0.1) * D;
        double fine = num("fine_size_D", 0.02) * D;
        AutoMeshOperation mesh = sim.get(MeshOperationManager.class).createAutoMeshOperation(
                Arrays.asList("star.resurfacer.ResurfacerAutoMesher", "star.trimmer.TrimmerAutoMesher"),
                Collections.<GeometryPart>singletonList(domain));
        mesh.getDefaultValues().get(BaseSize.class).setValue(cell);
        SimpleCylinderPart refinement = sim.get(MeshPartFactory.class).createNewCylinderPart(sim.get(SimulationPartManager.class));
        refinement.setPresentationName("Force disk to propeller plane refinement");
        refinement.setStartCoordinate(new DoubleVector(new double[] {forceX - 6 * sigma - 2 * fine, 0, 0}));
        refinement.setEndCoordinate(new DoubleVector(new double[] {propX + 0.1 * D, 0, 0}));
        refinement.setRadius(zoneRadius() + 2 * fine);
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
        boolean sideInlet = text("side_boundary", "symmetry").equalsIgnoreCase("inlet");
        int inlets = 0, outlets = 0;
        for (Boundary b : region.getBoundaryManager().getBoundaries()) {
            position.getParts().setObjects(b);
            double x = position.getReportMonitorValue();
            if (Math.abs(x - xIn) < 0.01 * D) {
                b.setPresentationName("Inlet");
                b.setBoundaryType(InletBoundary.class);
                inlets++;
            } else if (Math.abs(x - xOut) < 0.01 * D) {
                b.setPresentationName("Outlet");
                b.setBoundaryType(PressureBoundary.class);
                outlets++;
            } else if (sideInlet) {
                b.setPresentationName("Side inlet");
                b.setBoundaryType(InletBoundary.class);
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
        sim.println("WAKE_MESH base_m=" + f(cell) + " fine_m=" + f(fine) + " cfl=" + f(cfl));
    }

    /** Radius around the shaft axis that holds the force disk (also when it is offset) and the cost annulus. */
    private double zoneRadius() {
        return num("ctrl_r_max_R", 1.3) * R + num("zone_margin_D", 0.25) * D;
    }

    private UserFieldFunction fieldFunction(String name, FieldFunctionTypeOption.Type type, String definition) {
        FieldFunctionManager mgr = sim.getFieldFunctionManager();
        UserFieldFunction ff = mgr.has(name) ? (UserFieldFunction) mgr.getFunction(name) : mgr.createFieldFunction();
        ff.setPresentationName(name);
        ff.setFunctionName(name);
        ff.getTypeOption().setSelected(type);
        ff.setDefinition(definition);
        return ff;
    }

    private String zoneMask(double x0, double halfLength, double radius) {
        return "((abs($$Position[0]-(" + f(x0) + ")) <= " + f(halfLength) + " && sqrt(pow($$Position[1],2)+pow($$Position[2],2)) <= "
                + f(radius) + ") ? 1 : 0)";
    }

    private ThresholdPart threshold(String name, FieldFunction mask) {
        ThresholdPart part = sim.getPartManager().createThresholdPart();
        part.setPresentationName(name);
        part.getInputParts().setObjects(region);
        part.setFieldFunction(mask);
        part.setMode(ThresholdMode.BETWEEN_TAG);
        part.setRange(new DoubleVector(new double[] {0.5, 1.5}));
        return part;
    }

    private void buildZones() {
        double fine = num("fine_size_D", 0.02) * D;
        // force disk: Gaussian in x with std sigma; 5 sigma holds all but 1e-5 of it
        FieldFunction control = fieldFunction("WakeControlMask", FieldFunctionTypeOption.Type.SCALAR,
                zoneMask(forceX, 5 * sigma, zoneRadius()));
        // propeller plane: the cost slab plus one cell on each side
        double half = 0.5 * num("cost_slab_thickness_D", 0.02) * D + fine;
        FieldFunction target = fieldFunction("WakeTargetMask", FieldFunctionTypeOption.Type.SCALAR,
                zoneMask(propX, half, zoneRadius()));
        controlPart = threshold("Wake control cells", control);
        targetPart = threshold("Wake target cells", target);
    }

    /** Write X, Y, Z, Volume and the extra field functions of the cells of a threshold part. */
    private void exportCells(ThresholdPart part, String name, List<FieldFunction> extra) {
        String tableName = "Export " + name;
        XyzInternalTable table = null;
        for (Table t : sim.getTableManager().getObjects()) {
            if (t.getPresentationName().equals(tableName) && t instanceof XyzInternalTable) {
                table = (XyzInternalTable) t;
            }
        }
        if (table == null) {
            table = sim.getTableManager().createTable(XyzInternalTable.class);
            table.setPresentationName(tableName);
            table.getParts().setObjects(part);
        }
        List<FieldFunction> columns = new ArrayList<FieldFunction>();
        columns.add(sim.getFieldFunctionManager().getFunction("Volume"));
        columns.addAll(extra);
        table.setFieldFunctions(columns);
        table.extract();
        File out = caseFile(name);
        File tmp = caseFile(name + ".tmp");
        table.export(tmp.getPath(), ",");
        if (out.exists() && !out.delete()) {
            throw new IllegalStateException("cannot replace " + out);
        }
        if (!tmp.renameTo(out)) {
            throw new IllegalStateException("cannot rename " + tmp + " to " + out);
        }
    }

    private void setInflow() throws IOException {
        Properties inlet = new Properties();
        try (Reader in = new FileReader(caseFile("inlet.properties"))) {
            inlet.load(in);
        }
        double vx = Double.parseDouble(inlet.getProperty("inlet_vx").trim());
        double vy = Double.parseDouble(inlet.getProperty("inlet_vy").trim());
        double vz = Double.parseDouble(inlet.getProperty("inlet_vz").trim());
        for (Boundary b : region.getBoundaryManager().getBoundaries()) {
            if (b.getBoundaryType() instanceof InletBoundary) {
                b.getConditions().get(InletVelocityOption.class).setSelected(InletVelocityOption.Type.COMPONENTS);
                b.getValues().get(VelocityProfile.class).getMethod(ConstantVectorProfileMethod.class).getQuantity().setComponents(vx, vy, vz);
            }
        }
        physics.getInitialConditions().get(VelocityProfile.class).getMethod(ConstantVectorProfileMethod.class)
                .getQuantity().setComponents(vx, vy, vz);
        sim.println("WAKE_INFLOW " + f(vx) + " " + f(vy) + " " + f(vz));
    }

    /** Read forces_table.csv into a new table and point the momentum source at it; drops the old table. */
    private void loadForces() {
        Table old = forcesTable;
        tableCount++;
        String name = "WakeForces_" + tableCount;
        forcesTable = sim.getTableManager().createFromFile(caseFile("forces_table.csv").getPath());
        forcesTable.setPresentationName(name);
        String look = "interpolatePositionTable(@Table(\"" + name + "\"), \"%s\")";
        UserFieldFunction force = fieldFunction("WakeForce", FieldFunctionTypeOption.Type.VECTOR,
                "$WakeControlMask*[" + String.format(look, "fx") + ", " + String.format(look, "fy") + ", "
                        + String.format(look, "fz") + "]");
        if (old == null) {
            region.getConditions().get(MomentumUserSourceOption.class).setSelected(MomentumUserSourceOption.Type.SPECIFIED);
            MomentumUserSource source = region.getValues().get(MomentumUserSource.class);
            source.setMethod(FunctionVectorProfileMethod.class);
            source.getMethod(FunctionVectorProfileMethod.class).setFieldFunction(force);
        } else {
            sim.getTableManager().remove(old);
        }
    }

    private void buildCost() {
        Table target = sim.getTableManager().createFromFile(caseFile("target_table.csv").getPath());
        target.setPresentationName("WakeTarget");
        String[] cols = {"utx", "uty", "utz", "wxx", "wyy", "wzz", "wyz"};
        for (String c : cols) {
            fieldFunction("WakeT_" + c, FieldFunctionTypeOption.Type.SCALAR,
                    "interpolatePositionTable(@Table(\"WakeTarget\"), \"" + c + "\")");
        }
        fieldFunction("WakeDux", FieldFunctionTypeOption.Type.SCALAR, "$$Velocity[0]-$WakeT_utx");
        fieldFunction("WakeDuy", FieldFunctionTypeOption.Type.SCALAR, "$$Velocity[1]-$WakeT_uty");
        fieldFunction("WakeDuz", FieldFunctionTypeOption.Type.SCALAR, "$$Velocity[2]-$WakeT_utz");
        FieldFunction integrand = fieldFunction("WakeCostIntegrand", FieldFunctionTypeOption.Type.SCALAR,
                "$WakeTargetMask*($WakeT_wxx*pow($WakeDux,2)+$WakeT_wyy*pow($WakeDuy,2)+$WakeT_wzz*pow($WakeDuz,2)"
                        + "+2*$WakeT_wyz*$WakeDuy*$WakeDuz)");
        costReport = sim.getReportManager().createReport(VolumeIntegralReport.class);
        costReport.setPresentationName("Wake cost");
        costReport.setFieldFunction(integrand);
        costReport.getParts().setObjects(region);
        costReport.createMonitor();
        AdjointSolver adjoint = sim.getSolverManager().getSolver(AdjointSolver.class);
        costFunction = adjoint.getAdjointCostFunctionManager().createAdjointCostFunction(ReportCostFunction.class);
        costFunction.setPresentationName("Wake cost");
        costFunction.setReport(costReport);
    }

    // ------------------------------------------------------------------ solves

    private void solvePrimal() {
        for (SolverStoppingCriterion c : sim.getSolverStoppingCriterionManager().getObjects()) {
            c.setIsUsed(false);
        }
        int minIter = integer("primal_min_iterations", 300);
        int maxIter = integer("primal_max_iterations", 3000);
        int block = integer("primal_block", 50);
        int needStable = integer("primal_stable_blocks", 3);
        double tol = num("primal_tolerance", 1.0e-5);
        int done = 0, stable = 0;
        double previous = Double.NaN, j = Double.NaN;
        while (done < maxIter) {
            sim.getSimulationIterator().step(block);
            done += block;
            j = costReport.getReportMonitorValue();
            boolean small = !Double.isNaN(previous) && Math.abs(j - previous) <= tol * Math.max(Math.abs(j), 1e-30);
            stable = small ? stable + 1 : 0;
            previous = j;
            if (done >= minIter && stable >= needStable) {
                break;
            }
        }
        sim.println("WAKE_SOLVE primal iterations=" + done + " J_solver=" + f(j) + (stable >= needStable ? "" : " NOT_CONVERGED"));
    }

    private void solveAdjoint() {
        AdjointSolver adjoint = sim.getSolverManager().getSolver(AdjointSolver.class);
        adjoint.runAdjoint(Collections.<AdjointCostFunction>singletonList(costFunction), integer("adjoint_iterations", 1000));
        if (adjointField == null) {
            Pattern pattern = Pattern.compile(text("adjoint_ff_regex", "(?i).*VelocityAdjoint.*"));
            for (FieldFunction ff : sim.getFieldFunctionManager().getObjects()) {
                if (pattern.matcher(ff.getFunctionName()).matches()) {
                    sim.println("WAKE_ADJOINT_FF " + ff.getFunctionName());
                    if (adjointField == null) {
                        adjointField = ff;
                    }
                }
            }
            if (adjointField == null) {
                throw new IllegalStateException("no adjoint velocity field function matches adjoint_ff_regex");
            }
        }
        sim.println("WAKE_SOLVE adjoint " + adjointField.getFunctionName());
    }

    /** WakeLoop with the STAR-CCM+ hooks. */
    private class StarLoop extends WakeLoop {
        StarLoop(File caseFile, String python, String pythonPath) {
            super(caseFile, python, pythonPath);
        }

        @Override protected void reloadForces() {
            loadForces();
        }

        @Override protected void solvePrimal() {
            CalibrateWake.this.solvePrimal();
        }

        @Override protected void exportAchieved() {
            exportCells(targetPart, "achieved_cells.csv",
                    Collections.<FieldFunction>singletonList(sim.getFieldFunctionManager().getFunction("Velocity")));
        }

        @Override protected void solveAdjoint() {
            CalibrateWake.this.solveAdjoint();
        }

        @Override protected void exportAdjoint() {
            exportCells(controlPart, "adjoint_cells.csv", Collections.<FieldFunction>singletonList(adjointField));
            sim.saveState(caseFile("calibration.sim").getPath());
        }

        @Override protected void log(String line) {
            sim.println(line);
        }
    }

    // ------------------------------------------------------------------ WakeLoop
    // BEGIN WakeLoop (generated from java/WakeLoop.java by tools/embed_wakeloop.py; do not edit here)
    /**
     * Solver-independent driver of the wake calibration loop.
     *
     * It reads action.properties, asks the flow solver (through the abstract
     * hooks) for the solve the optimiser needs, and calls "python -m wakegen step"
     * to get the next force table. It contains no STAR-CCM+ calls, so it compiles
     * and runs with a plain JDK (see DryRunLoop.java). A STAR-CCM+ macro uses it
     * by pasting this class in as a nested static class and implementing the hooks.
     *
     * Hooks, in the order they are called for each action:
     *   primal_adjoint: reloadForces, solvePrimal, exportAchieved, solveAdjoint, exportAdjoint
     *   primal:         reloadForces, solvePrimal, exportAchieved
     *   adjoint:        solveAdjoint, exportAdjoint   (the flow solution is already current)
     *   stop:           reloadForces, solvePrimal, exportAchieved, then "report"
     */
    abstract static class WakeLoop {

        protected final File caseDir;
        protected final File caseFile;
        private final String python;
        private final String pythonPath;

        /**
         * @param caseFile   case.properties; its folder holds all exchanged files
         * @param python     Python executable, e.g. "python3"
         * @param pythonPath folder that contains the wakegen package, or null if it is installed
         */
        protected WakeLoop(File caseFile, String python, String pythonPath) {
            this.caseFile = caseFile.getAbsoluteFile();
            this.caseDir = this.caseFile.getParentFile();
            this.python = python;
            this.pythonPath = pythonPath;
        }

        /** Make the solver read forces_table.csv again. */
        protected abstract void reloadForces() throws Exception;

        /** Converge the flow with the current forces. */
        protected abstract void solvePrimal() throws Exception;

        /** Write achieved_cells.csv: X, Y, Z, Volume, Velocity[i,j,k] of the target cells. */
        protected abstract void exportAchieved() throws Exception;

        /** Converge the adjoint of the cost function on the current flow solution. */
        protected abstract void solveAdjoint() throws Exception;

        /** Write adjoint_cells.csv: X, Y, Z, Volume, adjoint momentum [i,j,k] of the control cells. */
        protected abstract void exportAdjoint() throws Exception;

        protected void log(String line) {
            System.out.println(line);
        }

        /** Run until the optimiser says stop. Returns the optimiser's closing message. */
        public String run(int maxSolves) throws Exception {
            for (int solves = 0; ; solves++) {
                Properties next = readProperties("action.properties");
                String action = next.getProperty("action", "");
                if (action.equals("stop")) {
                    reloadForces();
                    solvePrimal();
                    exportAchieved();
                    wakegen("report");
                    return next.getProperty("message", "");
                }
                if (solves >= maxSolves) {
                    throw new IllegalStateException("WAKE_FAILED more than " + maxSolves + " solves without a stop");
                }
                boolean primal = action.equals("primal_adjoint") || action.equals("primal");
                boolean adjoint = action.equals("primal_adjoint") || action.equals("adjoint");
                if (!primal && !adjoint) {
                    throw new IllegalStateException("WAKE_FAILED unknown action '" + action + "'");
                }
                if (primal) {
                    reloadForces();
                    solvePrimal();
                    exportAchieved();
                }
                if (adjoint) {
                    solveAdjoint();
                    exportAdjoint();
                }
                wakegen("step");
            }
        }

        /** Run "python -m wakegen <command> --case <caseFile> [extra...]"; its output goes to the log. */
        public void wakegen(String command, String... extra) throws Exception {
            List<String> cmd = new ArrayList<String>(Arrays.asList(python, "-m", "wakegen", command, "--case", caseFile.getPath()));
            cmd.addAll(Arrays.asList(extra));
            ProcessBuilder builder = new ProcessBuilder(cmd);
            builder.directory(caseDir);
            builder.redirectErrorStream(true);
            if (pythonPath != null && !pythonPath.isEmpty()) {
                builder.environment().put("PYTHONPATH", pythonPath);
            }
            Process process = builder.start();
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream(), "UTF-8"))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log(line);
                }
            }
            int code = process.waitFor();
            if (code != 0) {
                throw new IllegalStateException("WAKE_FAILED python -m wakegen " + command + " exited with " + code);
            }
        }

        public Properties readProperties(String name) throws Exception {
            Properties properties = new Properties();
            try (InputStream in = new FileInputStream(new File(caseDir, name))) {
                properties.load(in);
            }
            return properties;
        }
    }
    // END WakeLoop
}
