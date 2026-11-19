import numpy as np

from wakegen.control import ControlGrid, regularisation


def _grid():
    return ControlGrid(np.linspace(0.05, 0.6, 12), 36, x0=-1.0, sigma=0.03, edge_taper=0.07)


def _cells(rng, n=4000):
    return np.column_stack([rng.uniform(-1.12, -0.88, n), rng.uniform(-0.7, 0.7, n), rng.uniform(-0.7, 0.7, n)])


def test_apply_T_is_the_transpose_of_apply():
    rng = np.random.default_rng(0)
    grid, xyz = _grid(), _cells(rng)
    force = rng.normal(size=grid.shape)
    sens = rng.normal(size=(len(xyz), 3))
    lhs = np.sum(grid.apply(force, xyz) * sens)
    rhs = np.sum(force * grid.apply_T(sens, xyz))
    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_uniform_force_gives_profile_times_window():
    grid = _grid()
    xyz = np.array([[-1.0, 0.0, 0.3], [-1.03, 0.2, -0.2], [-1.0, 0.0, 0.0], [-1.0, 0.6, 0.0], [-0.8, 0.0, 0.3]])
    force = np.ones(grid.shape) * [1.0, 2.0, 3.0]
    f = grid.apply(force, xyz)
    peak = 1.0 / (0.03 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(f[0], peak * np.array([1, 2, 3]))
    np.testing.assert_allclose(f[1], peak * np.exp(-0.5) * np.array([1, 2, 3]))
    np.testing.assert_allclose(f[2], 0, atol=1e-12)  # shaft axis
    np.testing.assert_allclose(f[3], 0, atol=1e-12)  # outer edge
    np.testing.assert_allclose(f[4], 0, atol=1e-12)  # outside the disk in x
    assert list(grid.in_zone(xyz)) == [True, True, True, False, False]


def test_axial_profile_integrates_to_one():
    grid = _grid()
    x = np.linspace(-1.2, -0.8, 4001)
    assert abs(np.sum(grid.axial_profile(x)) * (x[1] - x[0]) - 1.0) < 5e-3  # cut at 3 sigma


def test_node_areas_sum_to_the_disk():
    grid = _grid()
    assert np.isclose(grid.node_area().sum(), np.pi * 0.6**2)


def test_bilinear_reproduces_a_field_linear_in_r():
    grid = _grid()
    force = np.zeros(grid.shape)
    force[..., 0] = grid.r_nodes[:, None]
    xyz = np.array([[-1.0, 0.25 * np.sin(1.234), 0.25 * np.cos(1.234)]])
    expect = 0.25 * grid.axial_profile(xyz[:, 0])[0]
    np.testing.assert_allclose(grid.apply(force, xyz)[0, 0], expect, rtol=1e-12)


def test_regularisation_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    fhat = rng.normal(size=(6, 8, 3))
    value, grad = regularisation(fhat, alpha=0.3, beta=0.7)
    direction = rng.normal(size=fhat.shape)
    eps = 1e-6
    fd = (regularisation(fhat + eps * direction, 0.3, 0.7)[0] - regularisation(fhat - eps * direction, 0.3, 0.7)[0]) / (2 * eps)
    assert np.isclose(fd, np.sum(grad * direction), rtol=1e-7)
    assert value > 0
