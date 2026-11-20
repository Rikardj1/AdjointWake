import numpy as np
import pytest

from wakegen.problem import Problem
from wakegen.wake_field import WakeField, polar_unit_vectors
from wakegen.wake_io import read_wake

from conftest import EXAMPLES


@pytest.fixture(scope="module")
def table():
    return read_wake(EXAMPLES / "WakeEdStar.csv")


def test_example_file_has_six_rings_of_31_points(table):
    assert len(table.rings) == 6
    assert [len(r.theta) for r in table.rings] == [31] * 6
    np.testing.assert_allclose(table.radii / 0.472, [0.35, 0.5, 0.65, 0.85, 1.0, 1.15], atol=2e-3)
    for ring in table.rings:
        assert np.all(np.diff(ring.theta) > 0) and ring.theta[0] >= 0 and ring.theta[-1] < 2 * np.pi


@pytest.mark.parametrize("kind", ["pchip", "linear"])
def test_interpolation_reproduces_the_table_points(table, kind):
    field = WakeField(table, [1.0, 0.0, 0.0], r_blend=0.7, kind=kind)
    y, z, u = table.points()
    np.testing.assert_allclose(field.evaluate(y, z), u, atol=1e-9)


def test_pchip_stays_within_neighbouring_data(table):
    field = WakeField(table, [1.0, 0.0, 0.0], r_blend=0.7)
    ring = table.rings[3]
    theta = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    u = field.evaluate(ring.radius * np.sin(theta), ring.radius * np.cos(theta))
    assert u[:, 0].min() >= ring.velocity[:, 0].min() - 1e-12
    assert u[:, 0].max() <= ring.velocity[:, 0].max() + 1e-12


def test_field_is_periodic_and_blends_to_the_free_stream(table):
    free = np.array([1.0, -0.04, 0.03])
    field = WakeField(table, free, r_blend=0.7)
    r = 0.4
    a = field.evaluate([r * np.sin(1e-9)], [r * np.cos(1e-9)])
    b = field.evaluate([r * np.sin(-1e-9)], [r * np.cos(-1e-9)])
    np.testing.assert_allclose(a, b, atol=1e-6)
    np.testing.assert_allclose(field.evaluate([0.0, 0.9], [0.7, 0.0]), [free, free], atol=1e-12)
    inside = field.evaluate([0.0], [0.05])  # held from the innermost ring, towards +Z
    np.testing.assert_allclose(inside, field.evaluate([0.0], [table.r_min]), atol=1e-12)


def test_units_and_scaling():
    table = read_wake(EXAMPLES / "WakeEdStar.csv", coord_scale=2.0, velocity_scale=5.0)
    assert table.r_max == pytest.approx(2 * 0.543, rel=1e-4)
    assert table.rings[0].velocity[:, 0].max() == pytest.approx(5 * 0.9492, rel=1e-6)


def test_wake_file_for_another_diameter_is_scaled_to_the_propeller(case_dir):
    from wakegen.config import Case

    text = (case_dir / "case.properties").read_text()
    for diameter in (0.944, 6.0):
        (case_dir / "case.properties").write_text(text.replace("diameter = 0.944", f"diameter = {diameter}"))
        case = Case.from_properties(case_dir / "case.properties")
        assert case.wake_diameter == 1.0 and case.wake_scale == pytest.approx(diameter)
        problem = Problem(case, case_dir)
        assert problem.table.r_max / case.radius == pytest.approx(0.543 / 0.5, rel=1e-4)


def test_polar_unit_vectors_are_orthonormal():
    y, z = np.array([0.0, 0.3, -0.2]), np.array([0.5, 0.1, -0.4])
    e_r, e_t = polar_unit_vectors(y, z)
    np.testing.assert_allclose(np.einsum("nc,nc->n", e_r, e_t), 0, atol=1e-15)
    np.testing.assert_allclose(e_r[0], [0, 0, 1])  # theta = 0 is +Z
    np.testing.assert_allclose(e_t[0], [0, 1, 0])  # increasing theta turns towards +Y


def test_crossflow_goes_into_the_inflow_vector(case, case_dir):
    problem = Problem(case, case_dir)
    mean = problem.wake_mean / case.ship_speed
    assert 0.7 < mean[0] < 0.9
    assert mean[1] < 0 < mean[2]  # this wake has side flow towards -Y and up-flow towards +Z
    np.testing.assert_allclose(problem.freestream, [case.ship_speed, problem.wake_mean[1], problem.wake_mean[2]])
    case.crossflow = "forces"
    np.testing.assert_allclose(Problem(case, case_dir).freestream, [case.ship_speed, 0, 0])
