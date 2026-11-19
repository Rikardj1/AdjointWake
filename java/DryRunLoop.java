import java.io.File;

/**
 * Runs WakeLoop with the Python surrogate in place of STAR-CCM+, to check the
 * Java/Python hand-over without a licence. Not a flow prediction.
 *
 *   javac -d build java/WakeLoop.java java/DryRunLoop.java
 *   java -cp build DryRunLoop <case.properties> <python> <folder with wakegen, or ""> [blur/R]
 */
public class DryRunLoop extends WakeLoop {

    private final String blur;
    private int primalSolves = 0;
    private int adjointSolves = 0;

    DryRunLoop(File caseFile, String python, String pythonPath, String blur) {
        super(caseFile, python, pythonPath);
        this.blur = blur;
    }

    @Override protected void reloadForces() { }

    @Override protected void solvePrimal() throws Exception {
        primalSolves++;
        wakegen("surrogate-solve", "--blur-R", blur);
    }

    @Override protected void exportAchieved() { }

    @Override protected void solveAdjoint() throws Exception {
        adjointSolves++;
        wakegen("surrogate-solve", "--blur-R", blur, "--adjoint");
    }

    @Override protected void exportAdjoint() { }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: DryRunLoop <case.properties> <python> <pythonpath or \"\"> [blur/R]");
            System.exit(2);
        }
        DryRunLoop loop = new DryRunLoop(new File(args[0]), args[1], args[2], args.length > 3 ? args[3] : "0.06");
        loop.wakegen("surrogate-cells");
        loop.wakegen("prepare");
        String message = loop.run(500);
        System.out.println("WAKE_DONE primal=" + loop.primalSolves + " adjoint=" + loop.adjointSolves + " message=" + message);
    }
}
