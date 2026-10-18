import cvxpy as cp
import numpy as np

class PortfolioOptimizer:
  def __init__(self, params, debug=False, **kwargs):
    super().__init__(
      params=params,
      debug=debug,
      **kwargs
    )
    self.p = params
    self.debug = debug
    self.lam = 1
    self.solver_chain = (cp.CLARABEL, cp.SCS)
    self.use_regularization = False

  def _smooth_covariance(self, cov):
    cov = 0.5 * (cov + cov.T)

    eigval, eigvec = np.linalg.eigh(cov)

    if eigval.min() < -1e-8:
      raise ValueError(
        f"Covariance materially non-PSD: "
        f"min eigenvalue={eigval.min()}"
      )

    eigval = np.maximum(eigval, 1e-10)

    return (
      eigvec
      @ np.diag(eigval)
      @ eigvec.T
    )

  def _solve_cluster_w(self, cluster_return_series, mean, covariance, w_anchor=None, scenario_prob=None):
    R = np.asarray(cluster_return_series, dtype=float)
    S, N = R.shape
    w = cp.Variable(N)
    u = cp.Variable(S)
    zeta = cp.Variable()
    gamma_var = self.p.gamma_var
    gamma_cvar = self.p.gamma_cvar
    lambda_anchor = self.p.lambda_anchor
    anchor_penalty = 0.0

    mu = np.asarray(mean, dtype=float)
    cov = np.asarray(
        self._smooth_covariance(covariance),
        dtype=float
    )

    if scenario_prob is None:
      p = np.ones(S) / S
    else:
      p = np.asarray(
        scenario_prob,
        dtype=float
      ).ravel()

      p /= p.sum()

    es = (
      zeta + cp.sum(cp.multiply(p, u))
      / (1.0  - self.p.cvar_q)
    )

    constraints = [
      u >= -R @ w - zeta,
      u >= 0,
      cp.sum(w) == 1,
      w >= 0.0,
    ]

    risk_term = gamma_var * cp.quad_form(w, cov) + gamma_cvar * es
    if w_anchor is not None:
      anchor_penalty = (
        lambda_anchor
        * cp.sum_squares(w - w_anchor)
      )

    obj_fn = cp.Maximize(
        mu @w
        - gamma_var * cp.quad_form(w, cov)
        - gamma_cvar * es
        - anchor_penalty
    )

    prob = cp.Problem(obj_fn, constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal"]:
        raise ValueError(f"Cluster MVO+CVaR optimization failed: {prob.status}")

    return w.value

  def _calculate_w(self, x):
    x_opt = np.asarray(x.value).ravel()
    w = x_opt/x_opt.sum()
    return w

  def _ru_cvar(self, R, scenario_prob=None):
    R = np.asarray(R, dtype=float)
    S, N = R.shape
    w_probe = np.ones(N) / N
    probe_R = R @ w_probe
    probe_L = -probe_R
    u = cp.Variable(S)
    zeta = cp.Variable()

    if scenario_prob is None:
      p = np.ones(S) / S
    else:
      p = np.asarray(scenario_prob, dtype=float).ravel()
      if p.shape[0] != S:
        raise ValueError(
          f"scenario_prob length {p.shape[0]} does not match "
          f"number of scenarios {S}."
        )
      p = p / p.sum()

    constraints = [
      u >= probe_L - zeta,
      u >= 0
    ]

    cvar = zeta + cp.sum(cp.multiply(p, u)) / (1.0 - self.p.cvar_q)
    obj_fn = cp.Minimize(cvar)
    problem = cp.Problem(obj_fn, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(
            "Risk scaling CVaR failed"
        )

    risk = float(problem.value)

    if not np.isfinite(risk) or risk <= 1e-10:
        raise ValueError(f"Invalid numerical risk scale: {risk}")

    return risk

  def _get_scaled_risk(self, R, scenario_prob=None):
    R = np.asarray(R, dtype=float)
    risk_scale = self._ru_cvar(R, scenario_prob=scenario_prob)

    return R / risk_scale, risk_scale

  def _inspect_sol(self, x_raw, R_num, zeta, used_solver, problem, risk_scale, scenario_prob=None):
    if (
        not np.all(np.isfinite(x_raw))
        or np.any(x_raw <= 0)
    ):
      raise RuntimeError(
        f"Invalid optimizer solution: {x_raw}"
      )

    raw_L = -(R_num @ x_raw)

    zeta_opt = float(zeta.value)

    if scenario_prob is None:
      tail_expectation = np.mean(np.maximum(raw_L - zeta_opt, 0.0))
    else:
      p = np.asarray(scenario_prob, dtype=float).ravel()
      p = p / p.sum()
      tail_expectation = np.sum(p * np.maximum(raw_L - zeta_opt, 0.0))

    raw_cvar_ru = (
        zeta_opt
        + tail_expectation
        / (1.0 - self.p.cvar_q)
    )

    kkt_scale_error = abs(
        raw_cvar_ru - self.lam
    )

    if self.debug:
      print("\n--- RB optimizer diagnostics ---")
      print("solver:          ", used_solver)
      print("status:          ", problem.status)
      print("risk scale:      ", risk_scale)
      print("raw x:           ", x_raw)
      print("raw zeta:        ", zeta_opt)
      print("scaled raw CVaR: ", raw_cvar_ru)
      print("lambda:          ", self.lam)
      print("|CVaR-lambda|:   ", kkt_scale_error)

    if not np.isclose(
      raw_cvar_ru,
      self.lam,
      rtol=1e-2,
      atol=1e-4,
    ):
      raise RuntimeError(
          "Risk-budget KKT scale condition failed: "
          f"CVaR(x)={raw_cvar_ru:.6g}, "
          f"lambda={self.lam:.6g}"
      )

  def _solve_cvar_risk_budget(
    self,
    horizon_returns,
    risk_budget=None,
    scenario_prob=None,
  ):
    R = np.asarray(horizon_returns, dtype=float)

    S, N = R.shape

    if risk_budget is None:
      b = np.ones(N) / N
    else:
      b = np.asarray(risk_budget, dtype=float).ravel()
      b /= b.sum()

    if scenario_prob is None:
      p = np.ones(S) / S
    else:
      p = np.asarray(scenario_prob, dtype=float).ravel()
      if p.shape[0] != S:
        raise ValueError(
          f"scenario_prob length {p.shape[0]} does not match "
          f"number of scenarios {S}."
        )
      p = p / p.sum()

    R_num, risk_scale = self._get_scaled_risk(R, p)

    x = cp.Variable(N)
    zeta = cp.Variable()
    u = cp.Variable(S, nonneg=True)

    losses = -(R_num @ x)

    constraints = [
      x >= 1e-7,
      u >= losses - zeta,
    ]

    cvar = zeta + cp.sum(cp.multiply(p, u)) / (1.0 - self.p.cvar_q)

    barrier = -self.lam * cp.sum(cp.multiply(b, cp.log(x)))

    problem = cp.Problem(
        cp.Minimize(cvar + barrier),
        constraints
    )
    used_solver = None
    last_status = None

    for solver in self.solver_chain:
      try:
        if solver == cp.SCS:
          problem.solve(
            solver=solver,
            eps=1e-6,
            max_iters=200_000,
            normalize=True,
            warm_start=True,
            verbose=False,
          )
        else:
          problem.solve(
            solver=solver,
            warm_start=True,
            verbose=False,
          )

      except cp.error.SolverError as exc:
        if self.debug:
          print(f"{solver} failed: {exc}")
        continue

      last_status = problem.status

      if self.debug:
        print(f"{solver}: {problem.status}")

      if problem.status == cp.OPTIMAL:
        used_solver = solver
        break

    if used_solver is None:
      raise RuntimeError(
        "All solvers in the fallback chain "
        f"{[s for s in self.solver_chain]} failed to reach optimality "
        f"(last status: {last_status})."
      )

    x_raw = np.asarray(
        x.value,
        dtype=float
    ).ravel()

    self._inspect_sol(x_raw, R_num, zeta, used_solver, problem, risk_scale, scenario_prob=scenario_prob)

    return self._calculate_w(x)

  def _get_intra_cluster_w(
      self,
      paths,
      idx,
      euler_risk_check_fn=None,
      risk_budget=None,
      scenario_prob=None,
  ):
    if self.debug:
      print(f"--------- Fitting w for cluster:{idx:^2} ---------")

    asset_R = (
      np.prod(1 + paths, axis=2) - 1
    )

    if risk_budget is None:
      b = np.ones(asset_R.shape[1]) / asset_R.shape[1]

    else:
      b = np.asarray(risk_budget, dtype=float).ravel()

      if len(b) != asset_R.shape[1]:
          raise ValueError("risk_budget dimension mismatch")
      if np.any(b <= 0):
          raise ValueError("risk budgets must be positive")

      b /= b.sum()

    w = self._solve_cvar_risk_budget(
      asset_R,
      risk_budget=b,
      scenario_prob=scenario_prob,
    )

    if euler_risk_check_fn is not None:
      euler_risk_check_fn(asset_R, w=w, b=b)

    return w, asset_R

class LambdaOptimizer:
  def __init__(self, params, debug=False, **kwargs):
    super().__init__(
      params=params,
      debug=debug,
      **kwargs
    )
    self.p = params
    self.debug = debug

  def optimize_lambda(self, p, A, b, k_eq, k_ineq):
    constraints = []
    K = k_eq + k_ineq
    lambda_var = cp.Variable(K)

    if k_ineq > 0:
        constraints.append(lambda_var[k_eq:] >= 0)

    exp_terms = -lambda_var @ A + np.log(p)
    obj_fn = cp.log_sum_exp(exp_terms) + lambda_var @ b

    prob = cp.Problem(cp.Minimize(obj_fn), constraints)
    prob.solve(solver=cp.CLARABEL)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Optimization failed. Status: {prob.status}")

    return lambda_var.value
