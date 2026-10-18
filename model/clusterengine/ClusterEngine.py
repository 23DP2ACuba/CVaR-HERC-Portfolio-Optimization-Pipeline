from model.DataStore import ModelParams, ClusterParams
from model.clusterengine.KendallTauModel import TailWeightedKendall
from scipy.cluster.hierarchy import fcluster, to_tree
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
import pandas as pd 
import numpy as np

class ClusterEngine(TailWeightedKendall):
  def __init__(self, params: ModelParams, debug=False, **kwargs):
    super().__init__(
      params=params,
      debug=debug,
      **kwargs
    )
    self.cp = ClusterParams()
    self.p = params

  def get_dist_mtx(
      self,
      returns:np.ndarray|pd.DataFrame,
      assets=None,
      tau_prev=None,
      return_tau=False
  ):
    if self.p.distance_method == "kendalltau":
      assert assets is not None, "Assets must be provided for Kendall Tau"
      dist_mtx = self.get_tau_dist(returns, assets, tau_prev, return_tau)

    elif self.p.distance_method == "pearsoncorr":
      assert returns.ndim == 2, "Returns must be two-dimensional"
      corr_mtx = np.clip(np.corrcoef(returns, rowvar=False), -1.0, 1.0)
      dist_mtx = np.sqrt(0.5 * (1 - corr_mtx))

    return dist_mtx

  def _generate_null_dists(self, sim_returns, risk_horizon, assets):
    n_paths, n_assets, n_steps = sim_returns.shape
    n_refs = self.p.gap_bootstrap_samples
    rng = np.random.default_rng()

    if sim_returns.ndim != 3:
      raise ValueError(
        "sim_returns must have shape "
        "(n_paths, n_assets, n_steps)"
      )

    null_dists, null_links = [], []

    for i in range(n_refs):
      null_returns = np.empty_like(sim_returns)

      for j in range(n_assets):
        perm = rng.permutation(n_paths)

        null_returns[:, j, :] = (
            sim_returns[perm, j, :]
        )

      null_dist = self.get_dist_mtx(
        null_returns,
        assets,
      )

      condensed_dist = squareform(null_dist.to_numpy(), checks=False)
      Z_null = linkage(condensed_dist, method="average")

      null_dists.append(null_dist)
      null_links.append(Z_null)

    return np.array(null_dists), np.array(null_links)

  def _compute_cl_dispersion(self, dist_mtx, cluster_labels):
    if isinstance(dist_mtx, pd.DataFrame):
      dist_mtx = dist_mtx.to_numpy()

    unique_clusters = np.unique(cluster_labels)
    W_k = 0.0

    for c_id in unique_clusters:
      cluster_indices = np.where(cluster_labels == c_id)[0]
      cluster_dist = dist_mtx[np.ix_(cluster_indices, cluster_indices)]

      norm = 2*len(cluster_indices)

      D_r = np.sum(cluster_dist**2)

      W_k += D_r / norm

    return W_k

  def _compute_ref_log_disp(self, null_dists, null_links, k):
    n_refs = len(null_dists)
    W_k_log = []

    for ref_i in range(n_refs):
      clusters = fcluster(null_links[ref_i], t=k, criterion="maxclust")

      W_k = self._compute_cl_dispersion(null_dists[ref_i], clusters)
      W_k_log.append(np.log(max(W_k, 1e-300)))

    W_k_log = np.array(W_k_log)
    E_W_k = np.mean(W_k_log)

    sdk = np.std(W_k_log, ddof=1) if n_refs > 1 else 0.0
    s_k = sdk * np.sqrt(1+1/n_refs)

    return E_W_k, s_k

  def _get_k_clusters(self, dist_mtx, Z, sim_returns, risk_horizon, assets):
    if self.debug:
      print("---------- Calculating K Clusters ----------")

    k_max = min(self.p.k_max, dist_mtx.shape[0]-1)
    Gap_k, s_k_list = [], []
    null_dists, null_links = self._generate_null_dists(sim_returns, risk_horizon, assets)
    k_range = list(range(1, k_max+1))

    for k in k_range:
      clusters = fcluster(Z, t=k, criterion="maxclust")

      W_k_real = self._compute_cl_dispersion(dist_mtx, clusters)
      log_disp_k = np.log(max(W_k_real, 1e-300))

      E_W_k, s_k = self._compute_ref_log_disp(null_dists, null_links, k)

      Gap_k.append(E_W_k - log_disp_k)
      s_k_list.append(s_k)

    gaps = np.array(Gap_k)
    s_k_list = np.array(s_k_list)

    optimal_k = k_range[np.argmax(gaps)]
    for idx in range(len(k_range)-1):
      if gaps[idx] >= gaps[idx+1] - s_k_list[idx+1]:
        optimal_k = k_range[idx]
        break

    if self.debug:
      print(f"Optimal k: {optimal_k}")
    return int(optimal_k)


  def get_clusters(
      self,
      dist_mtx:pd.DataFrame,
      sim_returns:np.ndarray,
      risk_horizon:int,
      assets:list
  ) -> pd.DataFrame:

    comp_disp = squareform(dist_mtx, checks=False)
    Z = linkage(comp_disp, method="average")

    optimal_k = self._get_k_clusters(dist_mtx, Z, sim_returns, risk_horizon, assets)
    labels = fcluster(Z, t=optimal_k, criterion="maxclust")

    return pd.DataFrame({'Asset': assets, 'Cluster': labels}), Z

  def _cluster_at_k(self, dist_mtx, k):
    comp_disp = squareform(dist_mtx, checks=False)
    Z = linkage(comp_disp, method="average")
    return fcluster(Z, t=k, criterion="maxclust")


  def _get_cluster_tree(
      self,
      Z,
      clusters_df,
  ):
    labels = (
      clusters_df["Cluster"]
      .to_numpy()
    )

    root = to_tree(
      Z,
      rd=False,
    )

    def contract(node):
      if node.is_leaf():
        cid = labels[node.id]
        return cid, {cid}

      left_tree, left_ids = contract(
        node.left
      )
      right_tree, right_ids = contract(
        node.right
      )

      ids = left_ids | right_ids

      if len(ids) == 1:
        cid = next(iter(ids))
        return cid, ids

      overlap = left_ids & right_ids

      if overlap:
        raise RuntimeError(
          "Flat clusters are not "
          "subtrees of linkage tree: "
          f"{overlap}"
        )

      return (
        (left_tree, right_tree),
        ids,
      )

    tree, _ = contract(root)

    return tree