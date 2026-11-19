"""L-BFGS with a backtracking line search, written as a restartable state machine.

Each cost evaluation is a flow solution and each gradient an adjoint solution,
both run by the flow solver in another process. The optimiser therefore never
calls a function: it says which solve it needs next (``action``) and is told
the result (``tell``). The whole state fits in one ``.npz`` file.

Actions:
    primal_adjoint  solve the flow and the adjoint at x_trial (first point only)
    primal          solve the flow at x_trial (line-search trial)
    adjoint         solve the adjoint at x_trial, whose flow solution is current
    stop            finished; x_trial holds the best accepted point
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class Options:
    memory: int = 10
    c1: float = 1.0e-4
    noise_tol: float = 0.0  # relative slack on the cost in the line search; 0 = strict decrease
    max_backtracks: int = 6
    max_iterations: int = 40
    initial_step: float = 0.02
    max_step: float = 0.20
    tol_cost: float = 0.0
    tol_rel_decrease: float = 1.0e-3


class Lbfgs:
    def __init__(self, x0: np.ndarray, options: Options, step_scale: np.ndarray | None = None):
        self.options = options
        self.step_scale = np.ones_like(x0) if step_scale is None else step_scale.astype(float)
        self.x = x0.astype(float).copy()  # last accepted point
        self.cost = np.nan
        self.grad = np.zeros_like(self.x)
        self.x_trial = self.x.copy()
        self.cost_trial = np.nan
        self.direction = np.zeros_like(self.x)
        self.step = 0.0
        self.slope = 0.0
        self.backtracks = 0
        self.pairs_s: list[np.ndarray] = []
        self.pairs_y: list[np.ndarray] = []
        self.iteration = 0
        self.evaluations = 0
        self.small_decreases = 0
        self.phase = "first"
        self.message = ""

    # ------------------------------------------------------------------
    @property
    def action(self) -> str:
        return {"first": "primal_adjoint", "line_search": "primal", "gradient": "adjoint", "done": "stop"}[self.phase]

    def tell(self, cost: float | None = None, grad: np.ndarray | None = None) -> str:
        """Report the result of the requested solve; returns the next action."""
        if self.phase == "first":
            if cost is None or grad is None:
                raise ValueError("the first evaluation needs both cost and gradient")
            self.evaluations += 1
            self.cost, self.grad = float(cost), grad.copy()
            if self.cost <= self.options.tol_cost:
                self._finish("cost below tol_cost at the starting point")
            else:
                self._start_search()
        elif self.phase == "line_search":
            if cost is None:
                raise ValueError("a line-search trial needs the cost")
            self.evaluations += 1
            self._line_search(float(cost))
        elif self.phase == "gradient":
            if grad is None:
                raise ValueError("the accepted point needs its gradient")
            self._accept(grad)
        return self.action

    # ------------------------------------------------------------------
    def _two_loop(self, grad: np.ndarray) -> np.ndarray:
        q = grad.copy()
        alphas = []
        for s, y in zip(reversed(self.pairs_s), reversed(self.pairs_y)):
            rho = 1.0 / np.dot(y, s)
            a = rho * np.dot(s, q)
            alphas.append((a, rho))
            q -= a * y
        if self.pairs_s:
            s, y = self.pairs_s[-1], self.pairs_y[-1]
            q *= np.dot(s, y) / np.dot(y, y)
        for (a, rho), s, y in zip(reversed(alphas), self.pairs_s, self.pairs_y):
            b = rho * np.dot(y, q)
            q += (a - b) * s
        return q

    def _start_search(self) -> None:
        if not np.any(self.grad):
            self._finish("zero gradient")
            return
        direction = -self._two_loop(self.grad)
        slope = float(np.dot(self.grad, direction))
        if slope >= 0.0:  # not a descent direction: drop the memory
            self.pairs_s.clear()
            self.pairs_y.clear()
            direction = -self.grad
            slope = float(np.dot(self.grad, direction))
        largest = float(np.max(np.abs(direction * self.step_scale)))
        if self.pairs_s:
            step = min(1.0, self.options.max_step / largest)
        else:
            step = self.options.initial_step / largest
        self.direction, self.slope, self.step, self.backtracks = direction, slope, step, 0
        self.x_trial = self.x + step * direction
        self.phase = "line_search"

    def _line_search(self, cost: float) -> None:
        opt = self.options
        allowed = self.cost + opt.c1 * self.step * self.slope + opt.noise_tol * abs(self.cost)
        if np.isfinite(cost) and cost <= allowed:
            self.cost_trial = cost
            self.phase = "gradient"
            return
        self.backtracks += 1
        if self.backtracks > opt.max_backtracks:
            if self.pairs_s:  # retry once from the accepted point with steepest descent
                self.pairs_s.clear()
                self.pairs_y.clear()
                self._start_search()
            else:
                hint = "check the sign and scaling of the adjoint (adjoint_scale)" if self.iteration == 0 else \
                    "converged to within the noise of the flow solution, or the gradient is inaccurate"
                self._finish(f"line search failed: no decrease along the search direction; {hint}")
            return
        shrink = 0.5
        if np.isfinite(cost):
            curvature = cost - self.cost - self.slope * self.step
            if curvature > 0.0:  # minimiser of the quadratic through the two points
                shrink = float(np.clip(-self.slope * self.step / (2.0 * curvature), 0.1, 0.5))
        self.step *= shrink
        self.x_trial = self.x + self.step * self.direction

    def _accept(self, grad: np.ndarray) -> None:
        opt = self.options
        s = self.x_trial - self.x
        y = grad - self.grad
        if np.dot(s, y) > 1.0e-10 * np.linalg.norm(s) * np.linalg.norm(y):
            self.pairs_s.append(s)
            self.pairs_y.append(y)
            del self.pairs_s[: -opt.memory], self.pairs_y[: -opt.memory]
        decrease = (self.cost - self.cost_trial) / max(abs(self.cost), 1.0e-300)
        self.small_decreases = self.small_decreases + 1 if decrease < opt.tol_rel_decrease else 0
        self.x, self.cost, self.grad = self.x_trial.copy(), self.cost_trial, grad.copy()
        self.iteration += 1
        if self.cost <= opt.tol_cost:
            self._finish("cost below tol_cost")
        elif self.iteration >= opt.max_iterations:
            self._finish("max_iterations reached")
        elif self.small_decreases >= 2:
            self._finish("cost no longer decreasing")
        else:
            self._start_search()

    def _finish(self, message: str) -> None:
        self.x_trial = self.x.copy()
        self.phase = "done"
        self.message = message

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        n = self.x.size
        np.savez(
            tmp,
            options=np.array([list(asdict(self.options).values())], dtype=float),
            step_scale=self.step_scale,
            x=self.x,
            grad=self.grad,
            x_trial=self.x_trial,
            direction=self.direction,
            pairs_s=np.array(self.pairs_s).reshape(-1, n),
            pairs_y=np.array(self.pairs_y).reshape(-1, n),
            scalars=np.array([self.cost, self.cost_trial, self.step, self.slope]),
            counters=np.array([self.backtracks, self.iteration, self.evaluations, self.small_decreases]),
            text=np.array([self.phase, self.message]),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "Lbfgs":
        with np.load(path) as data:
            names = list(asdict(Options()).keys())
            kinds = [type(v) for v in asdict(Options()).values()]
            options = Options(**{k: t(v) for k, t, v in zip(names, kinds, data["options"][0])})
            self = cls(data["x"], options, data["step_scale"])
            self.grad = data["grad"].copy()
            self.x_trial = data["x_trial"].copy()
            self.direction = data["direction"].copy()
            self.pairs_s = [row.copy() for row in data["pairs_s"]]
            self.pairs_y = [row.copy() for row in data["pairs_y"]]
            self.cost, self.cost_trial, self.step, self.slope = (float(v) for v in data["scalars"])
            self.backtracks, self.iteration, self.evaluations, self.small_decreases = (int(v) for v in data["counters"])
            self.phase, self.message = (str(v) for v in data["text"])
        return self
