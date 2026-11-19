import numpy as np

from wakegen.optimizer import Lbfgs, Options


def rosenbrock(x):
    a, b = x[:-1], x[1:]
    value = np.sum(100 * (b - a**2) ** 2 + (1 - a) ** 2)
    grad = np.zeros_like(x)
    grad[:-1] += -400 * a * (b - a**2) - 2 * (1 - a)
    grad[1:] += 200 * (b - a**2)
    return value, grad


def drive(opt, func, on_step=None):
    """Play the solver's role: answer whatever the optimiser asks for."""
    solves = {"primal": 0, "adjoint": 0}
    while opt.action != "stop":
        action = opt.action
        value, grad = func(opt.x_trial)
        if action == "primal_adjoint":
            solves["primal"] += 1
            solves["adjoint"] += 1
            opt.tell(value, grad)
        elif action == "primal":
            solves["primal"] += 1
            opt.tell(cost=value)
        else:
            solves["adjoint"] += 1
            opt.tell(grad=grad)
        if on_step:
            opt = on_step(opt)
    return opt, solves


def test_minimises_rosenbrock():
    opts = Options(max_iterations=300, initial_step=0.1, max_step=1.0, tol_rel_decrease=1e-12, noise_tol=0.0, tol_cost=1e-14)
    opt, solves = drive(Lbfgs(np.full(6, -1.0), opts), rosenbrock)
    np.testing.assert_allclose(opt.x, 1.0, atol=1e-5)
    np.testing.assert_allclose(opt.x_trial, opt.x)  # at stop the trial is the best point
    assert solves["adjoint"] == opt.iteration + 1  # one adjoint per accepted point


def test_cost_never_increases_between_accepted_points():
    costs = []

    def record(opt):
        costs.append(opt.cost)
        return opt

    opts = Options(max_iterations=60, initial_step=0.1, max_step=1.0, tol_rel_decrease=1e-12, noise_tol=0.0)
    drive(Lbfgs(np.array([-1.2, 1.0, 0.5]), opts), rosenbrock, record)
    assert np.all(np.diff(costs) <= 1e-15)


def test_state_survives_save_and_load_between_every_solve(tmp_path):
    path = tmp_path / "state.npz"

    def reload(opt):
        opt.save(path)
        return Lbfgs.load(path)

    opts = Options(max_iterations=40, initial_step=0.1, max_step=1.0, tol_rel_decrease=1e-12, noise_tol=0.0)
    direct, _ = drive(Lbfgs(np.full(4, -1.0), opts), rosenbrock)
    restarted, _ = drive(Lbfgs(np.full(4, -1.0), opts), rosenbrock, reload)
    np.testing.assert_allclose(restarted.x, direct.x, rtol=1e-12)
    assert restarted.iteration == direct.iteration and restarted.message == direct.message


def test_stops_with_a_message_when_the_gradient_is_wrong():
    def wrong_sign(x):
        value, grad = rosenbrock(x)
        return value, -grad

    opt, _ = drive(Lbfgs(np.full(3, -1.0), Options(noise_tol=0.0)), wrong_sign)
    assert opt.iteration == 0 and "line search failed" in opt.message
    np.testing.assert_allclose(opt.x_trial, -1.0)  # forces left at the starting point


def test_step_cap_limits_the_physical_change():
    opts = Options(initial_step=0.05, max_step=0.2)
    scale = np.array([1.0, 4.0, 1.0])
    opt = Lbfgs(np.zeros(3), opts, step_scale=scale)
    opt.tell(1.0, np.array([1.0, 1.0, 1.0]))
    assert np.isclose(np.max(np.abs((opt.x_trial - opt.x) * scale)), 0.05)
