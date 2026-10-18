from model.DataStore import TailFit
from scipy.stats import genpareto
import numpy as np
import warnings
import pandas as pd

class MarginalTailModel:
  def __init__(self, params, ticker):
    self.dist_p = params
    self.ticker = ticker
    self.tail_fit = None

  def fit_dist(self, r):
    full_idx = r.index
    r = r.dropna()
    valid_idx = r.index
    r = r.to_numpy(dtype=float)

    q_upper = self.dist_p.cvar_q * 100
    q_lower = (1 - self.dist_p.cvar_q) * 100

    u_upper = np.percentile(r, q_upper)
    u_lower = np.percentile(r, q_lower)

    upper_tail = r[r > u_upper]
    lower_tail = r[r < u_lower]

    if len(upper_tail) == 0 or len(lower_tail) == 0:
      raise ValueError(f"{self.ticker}: insufficient observations in EVT tail.")

    c_U, _, scale_U = genpareto.fit(upper_tail - u_upper, floc=0)
    c_L, _, scale_L = genpareto.fit(u_lower - lower_tail, floc=0)

    if c_L >= 1.0:
      warnings.warn(
        f"{self.ticker}: lower-tail GPD xi={c_L:.4f} >= 1. "
        "The fitted tail has infinite mean, so theoretical CVaR/ES is not finite.",
        RuntimeWarning,
      )
    elif c_L >= 0.5:
      warnings.warn(
        f"{self.ticker}: lower-tail GPD xi={c_L:.4f} is very heavy. "
        "CVaR can converge slowly even with QMC.",
        RuntimeWarning,
      )

    p_l = np.mean(r < u_lower)
    p_u = np.mean(r > u_upper)

    lower_mask = r < u_lower
    upper_mask = r > u_upper
    body_mask = (~lower_mask) & (~upper_mask)

    u_resid = np.zeros_like(r, dtype=float)

    if np.any(lower_mask):
      cdf_l = 1 - genpareto.cdf(
        u_lower - r[lower_mask], c_L, loc=0, scale=scale_L
      )
      u_resid[lower_mask] = p_l * cdf_l

    if np.any(body_mask):
      body = r[body_mask]
      ranks = pd.Series(body).rank(method="average").to_numpy()
      cdf_b = p_l + (1 - p_l - p_u) * (ranks / (len(body) + 1))
      u_resid[body_mask] = cdf_b

    if np.any(upper_mask):
      cdf_u = (1 - p_u) + p_u * genpareto.cdf(
        r[upper_mask] - u_upper, c_U, loc=0, scale=scale_U
      )
      u_resid[upper_mask] = cdf_u

    self.tail_fit = TailFit(
      c_L=c_L,
      scale_L=scale_L,
      c_U=c_U,
      scale_U=scale_U,
      p_l=p_l,
      p_u=p_u,
      u_lower=u_lower,
      u_upper=u_upper,
      ecdf_data=r[(r >= u_lower) & (r <= u_upper)],
    )

    u_resid_full = pd.Series(np.nan, index=full_idx, dtype=float)
    u_resid_full.loc[valid_idx] = u_resid
    return u_resid_full

  def inverse_cdf(self, u_resid):
    z_resid_final = np.zeros_like(u_resid, dtype=float)

    for i in range(u_resid.shape[1]):
      u_resid_i = u_resid[:, i]
      real_col = np.zeros_like(u_resid_i)

      p_u = self.tail_fit.p_u
      p_l = self.tail_fit.p_l
      u_upper = self.tail_fit.u_upper
      u_lower = self.tail_fit.u_lower

      lower_mask = u_resid_i < p_l
      upper_mask = u_resid_i > (1 - p_u)
      body_mask = (~lower_mask) & (~upper_mask)

      if np.any(upper_mask):
        q = (u_resid_i[upper_mask] - (1 - p_u)) / p_u
        real_col[upper_mask] = (
          u_upper
          + genpareto.ppf(
            q,
            self.tail_fit.c_U,
            loc=0,
            scale=self.tail_fit.scale_U,
          )
        )

      if np.any(body_mask):
        body_q = (
          (u_resid_i[body_mask] - p_l)
          / (1 - p_l - p_u)
          * 100
        )
        real_col[body_mask] = np.percentile(
          self.tail_fit.ecdf_data,
          np.clip(body_q, 0, 100),
        )

      if np.any(lower_mask):
        q = 1 - u_resid_i[lower_mask] / p_l
        real_col[lower_mask] = (
          u_lower
          - genpareto.ppf(
            q,
            self.tail_fit.c_L,
            loc=0,
            scale=self.tail_fit.scale_L,
          )
      )

      z_resid_final[:, i] = real_col

    return z_resid_final