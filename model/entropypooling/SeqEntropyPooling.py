from model.optimizers.Optimizers import LambdaOptimizer
from model.entropypooling.utils import Handlers
from scipy.special import logsumexp
import numpy as np


class EntropyPooling(Handlers, LambdaOptimizer):
  def __init__(self, params, debug:bool=False, heuristic="H1", **kwargs):
    self.debug = debug
    self.heuristic = heuristic
    super().__init__(
      params=params,
      debug=debug,
      **kwargs
    )

  def _check_views(q, R_prod, columns, views, tol=1e-6):
    th = {"mu": q @ R_prod, "var": q @ (R_prod - q @ R_prod) ** 2}
    for v in views:
        if v["view"] == "mean":
            i = columns.get_loc(v["ticker"]); val = th["mu"][i]
        elif v["view"] == "volatility":
            i = columns.get_loc(v["ticker"]); val = np.sqrt(th["var"][i])
        elif v["view"] == "correlation":
            i, j = columns.get_loc(v["ticker1"]), columns.get_loc(v["ticker2"])
            val = (q * (R_prod[:, i] - th["mu"][i]) * (R_prod[:, j] - th["mu"][j])).sum() \
                  / np.sqrt(th["var"][i] * th["var"][j])
        elif v["view"] == "relative":
            i, j = columns.get_loc(v["ticker1"]), columns.get_loc(v["ticker2"])
            val = th["mu"][i] - th["mu"][j]
        ok = abs(val - v["value"]) < tol if v["type"] == "eq" else \
             (val >= v["value"] - tol if v.get("direction") == "ge" else val <= v["value"] + tol)
        assert ok, (v, val)

  def _get_view_matrices(self, views, R, theta_q, columns):
    A_eq, b_eq, A_ineq, b_ineq = [], [], [], []
    fix_mean, fix_vol = {}, {}
    view_handlers = {
      "mean": lambda view: self._view_mean(view, R, columns),
      "relative": lambda view: self._view_relative(view, R, columns),
      "volatility": lambda view: self._view_volatility(view, R, columns, theta_q),
      "correlation": lambda view: self._view_correlation(view, R, columns, theta_q),
    }
    eq_mean = {v["ticker"] for v in views if v["view"] == "mean" and v["type"] == "eq"}
    eq_vol  = {v["ticker"] for v in views if v["view"] == "volatility" and v["type"] == "eq"}


    for view in views:
      handler = view_handlers.get(view["view"])

      if handler is None:
        raise ValueError(f"Unknown view type: {view['view']!r}")

      A, b, vtype, direction = handler(view)
      if vtype == "eq":
        A_eq.append(A)
        b_eq.append(b)
      elif direction == "ge":
        A_ineq.append(-A)
        b_ineq.append(-b)
      else:
        A_ineq.append(A)
        b_ineq.append(b)

      if view["view"] == "volatility":
          fix_mean[view["ticker"]] = None
      elif view["view"] == "correlation":
        for t in (view["ticker1"], view["ticker2"]):
          fix_mean[t] = None
          fix_vol[t] = None

    for t in fix_mean:
      if t in eq_mean:
          continue
      i = columns.get_loc(t)
      A_eq.append(R[:, i]); b_eq.append(theta_q["mu"][i])

    for t in fix_vol:
      if t in eq_vol:
          continue
      i = columns.get_loc(t)
      A_eq.append((R[:, i] - theta_q["mu"][i]) ** 2)
      b_eq.append(theta_q["var"][i])

    return A_eq, b_eq, A_ineq, b_ineq

  def _build_views(self, A_eq, b_eq, A_ineq, b_ineq):
    A_list, b_list = [], []

    k_eq = 0
    if A_eq is not None and b_eq is not None and len(A_eq) > 0:
      A_eq_clean = np.atleast_2d(A_eq)
      b_eq_clean = np.atleast_1d(b_eq)
      A_list.append(A_eq_clean)
      b_list.append(b_eq_clean)
      k_eq = A_eq_clean.shape[0]

    k_ineq = 0
    if A_ineq is not None and b_ineq is not None and len(A_ineq) > 0:
      A_ineq_clean = np.atleast_2d(A_ineq)
      b_ineq_clean = np.atleast_1d(b_ineq)
      A_list.append(A_ineq_clean)
      b_list.append(b_ineq_clean)
      k_ineq = A_ineq_clean.shape[0]

    if not A_list:
      return None, None, 0, 0

    A = np.vstack(A_list)
    b = np.concatenate(b_list)

    return A, b, k_eq, k_ineq


  def _solve_entropy_pooling(
    self,
    R,
    columns,
    views=None,
    p=None,
  ):
    if self.debug:
      print("------------- Entropy Pooling --------------")

    R_prod = np.prod(1 + R, axis=2) - 1
    S, N = R_prod.shape

    if p is None:
      p = np.ones(S) / S
    else:
      p = np.asarray(p)
      p = p / p.sum()

    if views is None:
      return p

    sorted_views = self._sort_views(views)
    applied = []
    q = p
    stages = [
      sorted_views["mean"] + sorted_views["relative"],
      sorted_views["volatility"],
      sorted_views["correlation"],
    ]
    theta = self._compute_theta(R_prod, q)
    for stage in stages:
      if not stage: continue
      applied += stage

      prior = p if self.heuristic == "H1" else q
      prior = np.clip(prior, 1e-300, 1 - 1e-300)

      A_eq, b_eq, A_ineq, b_ineq = self._get_view_matrices(applied, R_prod, theta, columns)
      A, b, k_eq, k_ineq = self._build_views(A_eq, b_eq, A_ineq, b_ineq)

      opt_lambda = self.optimize_lambda(prior, A, b, k_eq, k_ineq)
      log_q = np.log(prior) - opt_lambda @ A
      q = np.exp(log_q - logsumexp(log_q))
      theta = self._compute_theta(R_prod, q)


    q = np.asarray(q, dtype=float).ravel()
    self._posterior_diagnostics(p, q)
    self._check_views(q, R_prod, columns, views)

    return q
