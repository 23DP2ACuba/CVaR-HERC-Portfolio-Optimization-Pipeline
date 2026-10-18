import pandas as pd
import numpy as np

class ForwardTest:
  def __init__(
      self,
      model,
      leverage: float = 1.0,
      risk_free_rate: float = 0.0,
      risk_tolerance_threshold: float = 0.0,
      periods_per_year: int = 252,
      n_forward_batches: int = 1,
      debug: bool = False,
  ):
    self.model = model
    self.leverage = leverage
    self.risk_free_rate = risk_free_rate
    self.risk_tolerance_threshold = risk_tolerance_threshold
    self.periods_per_year = periods_per_year
    self.n_forward_batches = n_forward_batches
    self.debug = debug

    self.weights_ = None
    self.cluster_tree_str_ = None
    self.results_ = None

  def _generate_weights(self, returns):
    _, clusters_df, Z, sim_returns = self.model.fit_cvar_model(returns)
    w = self.model.assign_weights_cvar_herc(clusters_df, sim_returns, Z)

    tree_str = self._render_cluster_tree(
      self.model.last_cluster_tree,
      clusters_df,
    )

    return w, clusters_df, tree_str

  def _render_cluster_tree(self, tree, clusters_df):
    members = (
      clusters_df.groupby("Cluster")["Asset"]
      .apply(list)
      .to_dict()
    )

    def render(node):
      if isinstance(node, tuple):
        left, right = node
        return f"({render(left)}, {render(right)})"

      cid = node
      return "{" + ",".join(str(a) for a in members.get(cid, [cid])) + "}"

    return render(tree)

  def _draw_forward_scenarios(self):
    batches = [
      self.model.generate_r_sample(force_resimulate=True)
      for _ in range(self.n_forward_batches)
    ]
    return np.concatenate(batches, axis=0)

  def _apply_weights(self, w, columns, sim_returns_fwd):
    asset_order = list(columns)


    fwd_cols = pd.Index(self.model.cols)
    missing = set(asset_order) - set(fwd_cols)

    if missing:
      raise ValueError(
        f"Forward simulation is missing assets: {missing}"
      )

    idx = fwd_cols.get_indexer(asset_order)
    sim_returns_fwd = sim_returns_fwd[:, idx, :]

    step_returns = np.einsum("snt,n->st", sim_returns_fwd, w)

    rf_period = (
      (1.0 + self.risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
    )

    port_r = (
      self.leverage * step_returns
      - (self.leverage - 1.0) * rf_period
    )

    return port_r

  def _portfolio_metrics(self, port_r):
    S, T = port_r.shape

    equity = np.cumprod(1.0 + port_r, axis=1)
    equity = np.concatenate([np.ones((S, 1)), equity], axis=1)

    roi = equity[:, -1] - 1.0
    cagr = equity[:, -1] ** (self.periods_per_year / T) - 1.0
    vol = np.std(port_r, axis=1, ddof=1) * np.sqrt(self.periods_per_year)

    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown = equity / running_max - 1.0
    max_dd = -drawdown.min(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
      calmar = np.where(max_dd > 1e-12, cagr / max_dd, np.nan)

    roi_mean = float(np.mean(roi))
    forward_cvar = self.model._scenario_cvar(roi, scenario_prob=None)
    prob_above = float(np.mean(roi > self.risk_tolerance_threshold))

    return {
      "n_forward_scenarios": S,
      "expected_return": roi_mean,
      "forward_cvar": forward_cvar,
      "prob_above_threshold": prob_above,
      "avg_volatility": float(np.mean(vol)),
      "cagr": float(np.mean(cagr)),
      "calmar_ratio": float(np.nanmean(calmar)),
      "max_drawdown": float(np.mean(max_dd)),
    }

  def run(self, returns, label=None):
    w, clusters_df, tree_str = self._generate_weights(returns)

    sim_returns_fwd = self._draw_forward_scenarios()
    port_r = self._apply_weights(
      w,
      self.model.last_columns,
      sim_returns_fwd,
    )

    metrics = self._portfolio_metrics(port_r)
    metrics["leverage"] = self.leverage
    metrics["cluster_tree"] = tree_str

    self.weights_ = pd.Series(w, index=self.model.last_columns)
    self.cluster_tree_str_ = tree_str

    column_order = [
      "n_forward_scenarios",
      "leverage",
      "expected_return",
      "forward_cvar",
      "prob_above_threshold",
      "avg_volatility",
      "cagr",
      "calmar_ratio",
      "max_drawdown",
      "cluster_tree",
    ]

    row_label = (
      label
      if label is not None
      else pd.Timestamp.now().isoformat(timespec="seconds")
    )

    row = pd.DataFrame(
      [metrics],
      index=pd.Index([row_label], name="run"),
    )[column_order]

    self.results_ = (
      row
      if self.results_ is None
      else pd.concat([self.results_, row])
    )

    return self.results_, w