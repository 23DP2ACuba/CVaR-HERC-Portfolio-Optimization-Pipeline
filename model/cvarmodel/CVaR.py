from model.cvarmodel.GARCHEVTCOPULA import GARCHEVTCOPULA
from model.DataStore import ModelParams
import pandas as pd
import numpy as np
import os

class CVaR(GARCHEVTCOPULA):
  def __init__(
      self,
      params: ModelParams,
      debug: bool = False,
      **kwargs,
  ):
    super().__init__(
      debug=debug,
      params=params,
      **kwargs,
    )

    self.debug = debug
    self.cvar_p = params

    self.sobol_seed = params.seed
    self.sim_path = None

  def _make_sim_path(self):
    seed_label = (
      "random"
      if self.sobol_seed is None
      else str(self.sobol_seed)
    )
    start_s = pd.Timestamp(self.start).strftime("%Y-%m-%d")
    end_s   = pd.Timestamp(self.end).strftime("%Y-%m-%d")

    return (
      f"simulation_{start_s}_{end_s}"
      f"_steps{self.model_p.n_steps}"
      f"_sobol2p{self.model_p.sobol_pow}"
      f"_ql{self.model_p.cvar_q:g}"
      f"_qu{self.model_p.cvar_q:g}"
      f"_seed{seed_label}.npy"
    )

  def fit_garch_evt_copula(self, returns, fit_params):
    self.start = returns.index[0]
    self.end = returns.index[-1]
    self.universe = returns.columns

    if not bool(fit_params.fit):
      fit_params = self.fit(returns, fit_params)
      self.sim_path = self._make_sim_path()
    else:
      self.model_fit = fit_params
      self.cols = returns.columns

    return fit_params

  def generate_r_sample(self, force_resimulate=False):
      if self.sim_path is None:
        raise ValueError("Fit the model first!")

      may_use_cache = (
        self.sobol_seed is not None
        and not force_resimulate
        and os.path.exists(self.sim_path)
      )

      if may_use_cache:
        return np.load(self.sim_path)

      sim_returns = self.generate_sample()

      if self.sobol_seed is not None:
        np.save(self.sim_path, sim_returns)

      return sim_returns

  def verify_cvar(
    self,
    horizon_returns,
    w,
    b,
    scenario_prob=None
  ):
    R = np.asarray(horizon_returns, dtype=float)

    w = np.asarray(w, dtype=float).ravel()

    b = np.asarray(b, dtype=float ).ravel()

    b /= b.sum()

    asset_L = -R
    portfolio_L = asset_L @ w

    S = len(portfolio_L)
    tail_mass =  (1.0 - self.p.cvar_q) * S

    order = np.argsort(portfolio_L)[::-1]

    L_sorted = portfolio_L[order]
    asset_L_sorted = asset_L[order]

    k = int(np.floor(tail_mass))
    frac = tail_mass - k

    if k == 0:
      raise ValueError(
        "Not enough scenarios for CVaR level."
      )

    weighted_loss_sum = np.sum(
      L_sorted[:k]
    )

    weighted_asset_sum = np.sum(
      asset_L_sorted[:k],
      axis=0
    )

    if frac > 0:
      weighted_loss_sum += frac * L_sorted[k]
      weighted_asset_sum += frac * asset_L_sorted[k]

    cvar = weighted_loss_sum / tail_mass

    mcvar = weighted_asset_sum / tail_mass

    RC = w * mcvar
    realized_budget = RC / cvar

    print("w:               ", w)
    print("MCVaR:           ", mcvar)
    print("RC:              ", RC)
    print("CVaR:            ", cvar)
    print("Euler sum:       ", RC.sum())
    print("target budget:   ", b)
    print("realized budget: ", realized_budget)
    print(
        "Max budget error:",
        np.max(
            np.abs(realized_budget - b)
        )
    )

    return mcvar, cvar