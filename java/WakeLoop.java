import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

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
public abstract class WakeLoop {

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
