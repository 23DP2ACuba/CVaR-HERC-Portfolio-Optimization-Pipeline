from model.cvarmodel.MarginalTailModel import MarginalTailModel
from scipy.stats import qmc
import pyvinecopulib as pv
import numpy as np
from arch import arch_model
import pandas as pd

class GARCHEVTCOPULA:
  def __init__(self, params, debug=False, **kwargs):
    self.debug = debug
    self.model_p = params
    self.sobol_seed = params.seed

  @staticmethod
  def _fit_arch(model, warm_start_params):
    fit_kwargs = dict(disp="off", show_warning=False)
    if warm_start_params is not None:
      try:
        return model.fit(
          starting_values=np.asarray(warm_start_params, dtype=float),
          **fit_kwargs,
        )
      except (ValueError, TypeError, np.linalg.LinAlgError):
        pass
    return model.fit(**fit_kwargs)

  def _fit_model(
      self,
      r,
      ticker=None,
      warm_start_params=None,
      warm_start_lags=None,
  ):
    best_bic = np.inf
    best_model = best_fit = best_lags = None

    lag_candidates = (
      [warm_start_lags]
      if warm_start_lags is not None
      else range(1, self.model_p.lags + 1)
    )

    for lags in lag_candidates:
      model = arch_model(
        r,
        mean="AR",
        lags=lags,
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="studentst",
      )

      fit = self._fit_arch(model, warm_start_params)

      if fit.convergence_flag:
        continue

      alpha = fit.params["alpha[1]"]
      gamma = fit.params["gamma[1]"]
      beta = fit.params["beta[1]"]

      if alpha + beta + 0.5 * gamma < 1 and fit.bic < best_bic:
        best_bic = fit.bic
        best_model, best_fit, best_lags = model, fit, lags

    if best_model is None:
      for lags in lag_candidates:
        model = arch_model(
          r,
          mean="AR",
          lags=lags,
          vol="GARCH",
          p=1,
          o=0,
          q=1,
          dist="studentst",
        )
        fit = self._fit_arch(model, None)

        if fit.convergence_flag:
          continue

        alpha = fit.params["alpha[1]"]
        beta = fit.params["beta[1]"]

        if self.debug:
          print("GARCH persistence:", alpha + beta)

        if alpha + beta < 1 and fit.bic < best_bic:
            best_bic = fit.bic
            best_model, best_fit, best_lags = model, fit, lags

    if best_model is None:
      raise ValueError("No stationary GARCH fit found")

    return best_model, best_fit, best_lags

  def _fit_arma_garch(self, returns):
    if self.debug:
      print("------------- Fitting AR-GARCH -------------")

    log_returns = np.log1p(returns)
    filtered_resid = pd.DataFrame(
      index=returns.index,
      columns=returns.columns,
      dtype=float,
    )
    cols = returns.columns

    warm_start = bool(self.model_fit.fit)

    if not hasattr(self.model_fit, "ar_history"):
      self.model_fit.ar_history = {}

    for i, col in enumerate(cols):
      if self.debug:
        print(f"Fitting {col} | {i + 1}/{len(cols)}")

      r_scaled = (log_returns[col] * self.model_p.scale_factor).dropna()

      warm_params, warm_lags = None, None
      if warm_start and col in self.model_fit.best_fits:
        warm_params = self.model_fit.best_fits[col].params
        warm_lags = self.model_fit.best_lags[col]

      best_model, best_fit, best_lags = self._fit_model(
        r_scaled,
        ticker=col,
        warm_start_params=warm_params,
        warm_start_lags=warm_lags,
      )

      self.model_fit.best_models[col] = best_model
      self.model_fit.best_fits[col] = best_fit
      self.model_fit.best_lags[col] = best_lags

      hist = r_scaled.to_numpy(dtype=float)
      if hist.size < best_lags:
        raise ValueError(
          f"{col}: need at least {best_lags} observations "
          "to initialize AR recursion."
        )
      self.model_fit.ar_history[col] = hist[-best_lags:].copy()
      self.model_fit.vol_init[col] = np.asarray(
        best_fit.conditional_volatility,
        dtype=float
      )[-1]

      self.model_fit.r_init[col] = hist[-1]

      resid = pd.Series(best_fit.std_resid, index=r_scaled.index)
      filtered_resid.loc[resid.index, col] = resid

    return filtered_resid

  def _get_uniform_residuals(self, residuals):
      if self.debug:
        print("------------ Uniform Residuals -------------")

      u_resid = pd.DataFrame(
        index=residuals.index,
        columns=residuals.columns,
        dtype=float,
      )

      for ticker in residuals.columns:
        model = MarginalTailModel(self.model_p, ticker)
        u_resid_i = model.fit_dist(residuals[ticker])
        u_resid[ticker] = u_resid_i
        self.model_fit.marginal_distributions[ticker] = model

      return u_resid

  def _inverse_semi_parametric_cdf(self, uniform_samples):
    if self.debug:
      print("-------- Inverse Semi Parametric CDF -------")

    n_paths, n_steps, n_assets = uniform_samples.shape
    z_resid = np.zeros(
      (n_paths, n_steps, n_assets),
      dtype=float,
    )

    eps = 1e-6

    for asset_idx, ticker in enumerate(self.cols):
      model = self.model_fit.marginal_distributions[ticker]

      u_slice = uniform_samples[:, :, asset_idx].T
      u_slice = np.clip(u_slice, eps, 1 - eps)

      z_slice = model.inverse_cdf(u_slice)
      z_resid[:, :, asset_idx] = z_slice.T

    return z_resid

  def _fit_vine_copula(self, u_resid, structure=None):
    if self.debug:
      print("----------- Fitting Vine Copula ------------")

    clean = u_resid.dropna(how="any")

    if clean.empty:
      raise ValueError("No complete residual rows available for vine fit.")

    eps = 1e-10
    np_resid = np.asfortranarray(
      np.clip(clean.to_numpy(dtype=float), eps, 1 - eps),
      dtype=np.float64,
    )

    controls = pv.FitControlsVinecop()
    self.copula = pv.Vinecop.from_data(
      u=np_resid,
      controls=controls,
      structure=structure,
    )

  def fit(self, returns, fit_params):
    self.model_fit = fit_params

    z_resid = self._fit_arma_garch(returns)
    self.cols = z_resid.columns
    self.start = z_resid.index[0]
    self.end = z_resid.index[-1]

    u_resid = self._get_uniform_residuals(z_resid)

    structure = (
      self.model_fit.vine_structure
      if self.model_fit.fit
      else None
    )
    self._fit_vine_copula(u_resid, structure=structure)

    self.model_fit.vine_structure = self.copula.structure
    self.model_fit.fit = True

    return self.model_fit

  def _ar_coefficients(self, params, ticker, p):
    coeffs = np.empty(p, dtype=float)

    for lag in range(1, p + 1):
      key = f"{ticker}[{lag}]"
      if key not in params.index:
        raise KeyError(
          f"Expected AR parameter '{key}' not found. "
          f"Available parameters: {list(params.index)}"
        )
      coeffs[lag - 1] = float(params[key])

    return coeffs

  def _convert_to_returns(self, sim_r, model, params, ticker):
    if self.debug:
      print(f"------ Converting {ticker} to Returns -----")

    n_paths, n_steps = sim_r.shape
    p = int(self.model_fit.best_lags[ticker])

    const = float(params.get("Const", 0.0))
    phi = self._ar_coefficients(params, ticker, p)

    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params.get("gamma[1]", 0.0))
    beta = float(params["beta[1]"])

    fit = self.model_fit.best_fits[ticker]
    sigma_prev = fit.conditional_volatility.iloc[-1]
    eps_prev_scalar = fit.resid.iloc[-1]

    sigma2_prev = np.full(
      n_paths,
      sigma_prev ** 2,
      dtype=float,
    )
    eps_prev = np.full(
      n_paths,
      eps_prev_scalar,
      dtype=float,
    )

    hist_1d = np.asarray(
      self.model_fit.ar_history[ticker],
      dtype=float,
    )
    if hist_1d.size != p:
      raise ValueError(
          f"{ticker}: AR history size {hist_1d.size} != selected p={p}"
      )

    r_hist = np.tile(hist_1d, (n_paths, 1))

    r_t = np.zeros_like(sim_r, dtype=float)

    for t in range(n_steps):
      I_prev = (eps_prev < 0.0).astype(float)

      sigma2_t = (
        omega
        + (alpha + gamma * I_prev) * eps_prev ** 2
        + beta * sigma2_prev
      )

      sigma2_t = np.maximum(sigma2_t, 1e-14)

      eps_t = np.sqrt(sigma2_t) * sim_r[:, t]

      ar_term = np.zeros(n_paths, dtype=float)
      for lag in range(1, p + 1):
        ar_term += phi[lag - 1] * r_hist[:, -lag]

      mu_t = const + ar_term
      r_current = mu_t + eps_t
      r_t[:, t] = r_current

      if p > 1:
        r_hist[:, :-1] = r_hist[:, 1:]
      r_hist[:, -1] = r_current

      sigma2_prev = sigma2_t
      eps_prev = eps_t

    return r_t

  def _simulate(self, n_assets):
    n_steps = self.model_p.n_steps
    m = self.model_p.sobol_pow
    n_scenarios = 2 ** m
    d = n_steps * n_assets

    sobol_engine = qmc.Sobol(
      d=d,
      scramble=True,
      seed=self.sobol_seed,
    )

    base_uniforms = sobol_engine.random_base2(m=m)
    base_uniforms = base_uniforms.reshape(
      n_scenarios,
      n_steps,
      n_assets,
    )

    eps = 1e-12
    base_uniforms = np.clip(base_uniforms, eps, 1 - eps)

    copula_uniforms = np.empty_like(base_uniforms)

    for t in range(n_steps):
      u_t = np.asfortranarray(
        base_uniforms[:, t, :],
        dtype=np.float64,
      )
      copula_uniforms[:, t, :] = (
        self.copula.inverse_rosenblatt(u_t)
      )

    return copula_uniforms

  def generate_sample(self):
    if self.debug:
      print("------------ Generating Samples ------------")

    n_assets = len(self.cols)
    simulated_uniforms = self._simulate(n_assets)

    sim_resid = self._inverse_semi_parametric_cdf(
        simulated_uniforms
    )
    sim_resid_T = np.transpose(sim_resid, (0, 2, 1))

    sim_r_log = np.zeros_like(sim_resid_T)

    for i, ticker in enumerate(self.cols):
      params = self.model_fit.best_fits[ticker].params
      model = self.model_fit.best_models[ticker]

      log_paths = (
        self._convert_to_returns(
          sim_resid_T[:, i, :],
          model,
          params,
          ticker,
        )
        / self.model_p.scale_factor
      )

      sim_r_log[:, i, :] = log_paths

    sim_r = np.expm1(sim_r_log)

    if not np.all(np.isfinite(sim_r)):
      raise FloatingPointError(
        "Simulation produced NaN/inf returns. "
        "Inspect EVT xi and GARCH path diagnostics."
      )

    return sim_r