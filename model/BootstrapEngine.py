import numpy as np 
from sklearn.metrics import adjusted_rand_score

class BootstrapEngine:
  def __init__(self, params, debug=False, **kwargs):
    super().__init__(
        params=params,
        debug=debug,
        **kwargs
    )
    self.bootstrap_p = params
    self.debug = debug

  def run_bootstrap(self, returns):
    if self.debug:
      print("------ Running a Bootstrap Simulation ------")

    if returns is None:
      raise RuntimeError("Call .run(returns) before .bootstrap_metrics().")

    r = returns.values
    T = len(r)
    block_size = self.bootstrap_p.block_size or max(int(np.sqrt(T)), 2)
    rng = np.random.default_rng(self.bootstrap_p.seed)


    for i in range(self.bootstrap_p.n_bootstrap):
      idx = self._resample_indices(T, block_size, rng)

  def _resample_indices(self, T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
      return self._stationary_bootstrap_indices(T, block_size, rng)

  @staticmethod
  def _stationary_bootstrap_indices(T: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / block_size
    indices = np.empty(T, dtype=int)
    idx = rng.integers(0, T)

    for t in range(T):
      indices[t] = idx
      idx = rng.integers(0, T) if rng.random() < p else (idx + 1) % T

    return indices

  def bootstrap_cluster_stability(self, sim_returns, assets, baseline_labels, k, alpha=0.05):
    n_paths = sim_returns.shape[0]
    block_size = self.bootstrap_p.block_size or max(int(np.sqrt(n_paths)), 2)
    rng = np.random.default_rng(self.bootstrap_p.seed)

    tau_prev_snapshot = getattr(self, "tau_prev", None)
    aris = np.empty(self.bootstrap_p.n_bootstrap, dtype=float)

    try:
      for b in range(self.bootstrap_p.n_bootstrap):
        self.tau_prev = tau_prev_snapshot
        idx = self._resample_indices(n_paths, block_size, rng)
        resampled_paths = sim_returns[idx]

        dist_mtx_b = self.get_dist_mtx(resampled_paths, assets)
        labels_b = self._cluster_at_k(dist_mtx_b, k)

        aris[b] = adjusted_rand_score(baseline_labels, labels_b)
    finally:
      self.tau_prev = tau_prev_snapshot

    threshold = float(np.quantile(aris, alpha))
    if self.debug:
      print(f"Sticky-cluster ARI threshold ({alpha:.0%} quantile): {threshold:.4f}")

    return threshold, aris