from model.entropypooling.SeqEntropyPooling import EntropyPooling
from model.clusterengine.ClusterEngine import ClusterEngine
from model.optimizers.Optimizers import PortfolioOptimizer
from model.DataStore import DataStore, CVaRHERCParams
from model.cvarmodel.CVaR import CVaR
from model.BootstrapEngine  import BootstrapEngine
from sklearn.metrics import adjusted_rand_score
import numpy as np
import os

class CVaRHERCPortfolio(
  DataStore, 
  EntropyPooling, 
  ClusterEngine, 
  PortfolioOptimizer, 
  BootstrapEngine, 
  CVaR
):
  def __init__(self, params, debug=False, **kwargs):
    super().__init__(
        params=params,
        debug=debug,
        **kwargs
    )
    self.p = params
    self.debug = debug
    self.stored_params = CVaRHERCParams()
    self.Z_prev = None

  def get_data(self, universe, start, end, benchmark_ticker="^GSPC"):
    returns, benchmark = self._get_data(
      universe=universe,
      start=start,
      end=end,
      benchmark=benchmark_ticker
    )

    return returns, benchmark

  def _scenario_cvar(self, horizon_returns, scenario_prob, floor=1e-8):
    R = np.asarray(horizon_returns, dtype=float).ravel()

    if R.size == 0:
      raise ValueError("horizon_returns cannot be empty.")

    if not np.all(np.isfinite(R)):
      raise ValueError(
        "horizon_returns contains NaN or inf values."
      )

    L = -R

    alpha = self.p.cvar_q

    if not 0.0 < alpha < 1.0:
      raise ValueError(
        "cvar_q must lie strictly between 0 and 1."
      )

    var = np.quantile(L, alpha)

    tail_losses = L[L >= var]

    if tail_losses.size == 0:
      cvar = var
    else:
      cvar = np.mean(tail_losses)

    if not np.isfinite(cvar):
      raise ValueError(
        "Computed CVaR is non-finite."
      )

    return max(float(cvar), floor)

  def _get_herc_weights(self, R_left_sim, R_right_sim, scenario_prob=None):
    risk_L = self._ru_cvar(
        np.asarray(R_left_sim)[:, None],
        scenario_prob=scenario_prob,
    )

    risk_R = self._ru_cvar(
        np.asarray(R_right_sim)[:, None],
        scenario_prob=scenario_prob,
    )

    alpha_L = risk_R / (risk_L + risk_R)
    alpha_R = risk_L / (risk_L + risk_R)

    return alpha_L, alpha_R

  def _recursive_tree_allocate(
      self,
      tree,
      cluster_sim_return_series,
      scenario_prob=None,
  ):
    S = next(
        iter(
            cluster_sim_return_series.values()
        )
    ).shape[0]

    if scenario_prob is None:
        q = np.ones(S) / S
    else:
      q = np.asarray(
          scenario_prob,
          dtype=float,
      ).ravel()

      q /= q.sum()

    def solve(node):
      if not isinstance(node, tuple):
        cid = node

        return (
          cluster_sim_return_series[cid],
          {cid: 1.0},
        )

      left_node, right_node = node

      R_left, W_left = solve(
          left_node
      )

      R_right, W_right = solve(
          right_node
      )

      alpha_L_herc, alpha_R_herc = (
        self._get_herc_weights(
          R_left,
          R_right,
          q,
        )
      )


      if self.p.use_cluster_mvo:
        w_anchor = np.array([
          alpha_L_herc,
          alpha_R_herc,
        ])

        node_R = np.column_stack([
          R_left,
          R_right,
        ])

        mu_post = np.array([
          q @ R_left,
          q @ R_right,
        ])

        R_dev = node_R - mu_post

        cov = (
          R_dev.T * q
        ) @ R_dev

        node_w = self._solve_cluster_w(
          cluster_return_series=node_R,
          mean=mu_post,
          covariance=cov,
          w_anchor=w_anchor,
          scenario_prob=q,
        )

        alpha_L, alpha_R = node_w

      else:
        alpha_L = alpha_L_herc
        alpha_R = alpha_R_herc

      weights = {}

      for cid, w in W_left.items():
        weights[cid] = (
          alpha_L * w
        )

      for cid, w in W_right.items():
        weights[cid] = (
          alpha_R * w
        )

      R_parent = (
          alpha_L * R_left
          + alpha_R * R_right
      )

      return R_parent, weights

    _, weights = solve(tree)

    return weights


  def fit_cvar_model(self, returns):
    if self.stored_params.fit and self.stored_params.tau_prev is not None:
      self.tau_prev = self.stored_params.tau_prev

    self.stored_params = self.fit_garch_evt_copula(returns, self.stored_params)
    if os.path.exists(self.sim_path):
      sim_returns = np.load(self.sim_path)

    else:
      sim_returns = self.generate_r_sample()
      np.save(self.sim_path, sim_returns)

    try:
      tau_prev = self.stored_params.tau_prev.values
    except:
      tau_prev = None

    dist_mtx, tau_prev = self.get_dist_mtx(
        returns=sim_returns,
        assets=returns.columns,
        tau_prev=tau_prev,
        return_tau=True,
    )

    self.dist_mtx = dist_mtx
    self.stored_params.tau_prev = tau_prev

    clusters_df, Z = self.get_clusters(dist_mtx, sim_returns, self.p.n_steps, returns.columns)
    clusters_df, new_z = self._apply_sticky_clusters(clusters_df, sim_returns, returns.columns)

    self.clusters = clusters_df
    self.Z_prev = Z if new_z else self.Z_prev

    return self.stored_params, clusters_df, self.Z_prev, sim_returns

  def _apply_sticky_clusters(self, new_clusters_df, sim_returns, assets):
    prev = getattr(self, "clusters", None)
    if prev is None or set(prev["Asset"]) != set(new_clusters_df["Asset"]):
      return new_clusters_df, True

    prev_labels = prev.set_index("Asset").loc[new_clusters_df["Asset"], "Cluster"].values
    new_labels = new_clusters_df["Cluster"].values
    observed_ari = adjusted_rand_score(prev_labels, new_labels)

    k = len(np.unique(new_labels))
    threshold, _ = self.bootstrap_cluster_stability(
      sim_returns=sim_returns,
      assets=list(assets),
      baseline_labels=new_labels,
      k=k
    )

    if self.debug:
      print(f"Sticky cluster check: observed ARI={observed_ari:.4f} vs threshold={threshold:.4f}")

    if observed_ari >= threshold:
      return prev.set_index("Asset").loc[new_clusters_df["Asset"]].reset_index(), False

    return new_clusters_df, True

  def _semantic_tree_splits(
    self,
    tree,
    clusters_df,
  ):
    cluster_members = {
      int(cid): tuple(sorted(g["Asset"].tolist()))
      for cid, g in clusters_df.groupby("Cluster")
    }

    splits = []

    def walk(node):
      if not isinstance(node, tuple):
        return frozenset(
          cluster_members[int(node)]
        )

      left_assets = walk(node[0])
      right_assets = walk(node[1])

      L = tuple(sorted(left_assets))
      R = tuple(sorted(right_assets))

      split = tuple(
        sorted(
          (L, R),
          key=repr,
        )
      )

      splits.append(split)

      return left_assets | right_assets

    walk(tree)

    return tuple(
      sorted(
        splits,
        key=repr,
      )
    )

  def assign_weights_cvar_herc(self, clusters_df, sim_returns, Z):
    columns = clusters_df["Asset"].values

    q = self._solve_entropy_pooling(
      R=sim_returns,
      columns=columns,
      views=getattr(self.p, "views", None),
    )

    unique = np.unique(clusters_df.Cluster.values)
    w_by_cluster = {}
    cluster_return_series = {}

    for c_id in unique:
      cols_i = clusters_df.Cluster == c_id
      sim_returns_i = sim_returns[:, cols_i]

      if sim_returns_i.shape[1] > 1:
        w_i, _ = self._get_intra_cluster_w(
          paths=sim_returns_i,
          idx=c_id,
          euler_risk_check_fn=self.verify_cvar,
          scenario_prob=q,
        )
      else:
        w_i = np.ones(1)

      w_by_cluster[c_id] = w_i

      cluster_R_i = np.prod(1 + sim_returns_i, axis=2) - 1

      cluster_return_series[c_id] = cluster_R_i @ w_i

    cluster_tree = self._get_cluster_tree(
      Z,
      clusters_df,
    )

    print(
      "SEMANTIC TREE:",
      self._semantic_tree_splits(
          cluster_tree,
          clusters_df,
      )
    )

    cluster_weights = (
      self._recursive_tree_allocate(
        cluster_tree,
        cluster_return_series,
        scenario_prob=q,
      )
    )

    w_final = np.zeros(
      sim_returns.shape[1],
      dtype=float
    )

    labels = clusters_df["Cluster"].to_numpy()

    for c_id, w_intra in w_by_cluster.items():
      idx = np.flatnonzero(labels == c_id)

      w_final[idx] = (
        cluster_weights[c_id]
        * w_intra
      )

    w_final /= w_final.sum()

    self.last_scenario_prob = q
    self.last_cluster_tree = cluster_tree
    self.last_columns = columns

    return w_final


  def get_w(self, returns):
    _, clusters_df, Z, sim_returns = self.fit_cvar_model(returns)
    return self.assign_weights_cvar_herc(clusters_df, sim_returns, Z)