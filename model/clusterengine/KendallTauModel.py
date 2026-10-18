import numpy as np
from scipy.stats import kendalltau, rankdata
import pandas as pd

class TailWeightedKendall:
  def __init__(
    self,
    params,
    **kwargs
  ):
    super().__init__(params=params, **kwargs)
    if not 0.0 < params.tail_threshold < 1.0:
      raise ValueError(
        "tail_threshold must lie between 0 and 1."
      )

    if params.tail_power <= 0:
      raise ValueError(
        "tail_power must be positive."
      )

    self.tail_threshold = params.tail_threshold
    self.tail_power = params.tail_power

    self.min_effective_sample = params.min_effective_sample

    self.shrink = params.shrink
    self.shrink_target_ess = params.shrink_target_ess
    self.tau_shrinkage_params = params.tau_shrinkage_params

    if self.shrink_target_ess <= self.min_effective_sample:
      raise ValueError(
        "shrink_target_ess must exceed "
        "min_effective_sample"
      )

    if not 0 <= self.tau_shrinkage_params <= 1:
      raise ValueError(
        "tau_shrinkage_params must lie in [0,1]"
      )

  @staticmethod
  def compound_returns(sim_returns):
    sim_returns = np.asarray(
      sim_returns,
      dtype=float
    )
    if sim_returns.ndim != 3:
      raise ValueError(
        "sim_returns must have shape "
        "(n_paths, n_assets, n_steps)."
      )

    return (
        np.prod(1.0 + sim_returns, axis=2) - 1.0
    )

  @staticmethod
  def empirical_uniforms(losses):
    losses = np.asarray(
      losses,
      dtype=float
    )

    if losses.ndim != 2:
      raise ValueError(
        "losses must be two-dimensional."
      )

    n_paths, n_assets = losses.shape

    U = np.empty_like(
      losses,
      dtype=float
    )

    for i in range(n_assets):
      U[:, i] = (
        rankdata(
          losses[:, i],
          method="average"
        ) / (n_paths + 1.0)
      )

    return U

  def tail_weight(self, u):
    u = np.asarray(u, dtype=float)

    z = np.maximum(
      (u - self.tail_threshold) / (1.0 - self.tail_threshold),
      0.0
    )

    return z ** self.tail_power

  @staticmethod
  def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)

    s1 = weights.sum()
    s2 = np.sum(weights ** 2)

    if s2 <= 0:
        return 0.0

    return (s1 ** 2) / s2

  @staticmethod
  def _weighted_kendall(x, y, obs_weights):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(obs_weights, dtype=float)

    n = len(x)
    if n < 2:
      return np.nan

    numerator = 0.0
    denominator = 0.0

    for p in range(n - 1):
      dx = x[p] - x[p + 1:]
      dy = y[p] - y[p + 1:]

      pair_weights = (w[p] * w[p + 1:])

      signs = np.sign(dx * dy)

      numerator += np.sum(pair_weights * signs)

      denominator += np.sum(pair_weights)

    if denominator <= 0:
      return np.nan

    return numerator / denominator

  def pair_tau(self, losses_i, losses_j, U_i, U_j):
    wi = self.tail_weight(U_i)
    wj = self.tail_weight(U_j)
    pair_weights = wi * wj
    mask = pair_weights > 0
    n_tail = int(mask.sum())

    tau_global, _ = kendalltau(losses_i, losses_j)
    if not np.isfinite(tau_global):
      tau_global = 0.0

    if n_tail < 2:
      return {
        "tau_tail": np.nan,
        "tau_global": tau_global,
        "tau_final": np.clip(tau_global, -1.0, 1.0),
        "ess": 0.0,
        "n_tail": n_tail,
      }

    x = losses_i[mask]
    y = losses_j[mask]
    w = pair_weights[mask]

    ess = self.effective_sample_size(w)
    tau_tail = self._weighted_kendall(x, y, w)

    if self.shrink and np.isfinite(tau_tail):
      if ess < self.min_effective_sample:
        tau_final = tau_global
      else:
        gamma = np.clip(
            (ess - self.min_effective_sample)
            / (self.shrink_target_ess - self.min_effective_sample),
            0.0,
            1.0,
          )

        tau_final = (
          gamma * tau_tail
          + (1.0 - gamma) * tau_global
        )

    else:
        tau_final = tau_tail if np.isfinite(tau_tail) else tau_global

    if not np.isfinite(tau_final):
        tau_final = 0.0

    tau_final = np.clip(tau_final, -1.0, 1.0)

    return {
        "tau_tail": tau_tail,
        "tau_global": tau_global,
        "tau_final": tau_final,
        "ess": ess,
        "n_tail": n_tail,
    }

  def fit_kendall(
    self,
    returns,
    assets=None,
    tau_prev=None,
    return_tau=False
  ):
    horizon_returns = self.compound_returns(returns)

    losses = -horizon_returns

    n_paths, n_assets = losses.shape

    if assets is None:
      assets = [
        f"asset_{i}"
        for i in range(n_assets)
      ]

    if len(assets) != n_assets:
      raise ValueError(
        "len(assets) must equal "
        "returns.shape[1]."
      )

    U = self.empirical_uniforms(losses)

    tau = np.eye(n_assets, dtype=float)
    tau_tail = np.eye(n_assets, dtype=float)
    tau_global = np.eye(n_assets, dtype=float)
    ess = np.zeros((n_assets, n_assets), dtype=float)
    n_tail = np.zeros((n_assets, n_assets), dtype=int)

    for i in range(n_assets):
      ess[i, i] = n_paths
      n_tail[i, i] = n_paths

      for j in range(i + 1, n_assets):
        result = self.pair_tau(losses[:, i], losses[:, j], U[:, i], U[:, j])

        tau[i, j] = result[ "tau_final"]
        tau[j, i] = tau[i, j]
        tau_tail[i, j] = result["tau_tail"]
        tau_tail[j, i] = tau_tail[i, j]
        tau_global[i, j] = result["tau_global"]
        tau_global[j, i] = tau_global[i, j]

        ess[i, j] = result["ess"]
        ess[j, i] = ess[i, j]

        n_tail[i, j] = result[ "n_tail"]
        n_tail[j, i] = n_tail[i, j]

    if tau_prev is not None:
      if tau_prev.shape != tau.shape:
        raise ValueError("tau_prev shape mismatch")

      rho = self.tau_shrinkage_params
      tau_out = tau * (1.0 - rho) + tau_prev * rho

    else:
      tau_out = tau

    distance = np.sqrt(
      0.5 * (1.0 - np.clip(tau_out, -1.0, 1.0))
    )

    np.fill_diagonal(distance, 0.0)
    dist_out = pd.DataFrame(distance, index=assets, columns=assets)

    if return_tau:
      tau_out_df = pd.DataFrame(tau_out, index=assets, columns=assets)
      output = (dist_out, tau_out_df)
    else:
      output = dist_out

    return output

  def get_tau_dist(
      self,
      returns,
      assets=None,
      tau_prev=None,
      return_tau=False
  ):
    dist_mtx = self.fit_kendall(
      returns=returns,
      assets=assets,
      tau_prev=tau_prev,
      return_tau=return_tau
    )
    return dist_mtx