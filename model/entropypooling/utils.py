import numpy as np
class Handlers:
  def __init__(self, params, debug=False, **kwargs):
    super().__init__(
      params=params,
      debug=debug,
      **kwargs
    )
    self.p = params
    self.debug = debug

  def _compute_theta(self, R, q):
    mu = q @ R
    var = q @ (R - mu) ** 2
    return {"mu": mu, "var": var}

  def _view_mean(self, view, R, columns):
    idx = columns.get_loc(view["ticker"])
    A = R[:, idx]
    b = view["value"]
    return A, b, view["type"], view.get("direction", "le")

  def _view_relative(self, view, R, columns):
    idx1 = columns.get_loc(view["ticker1"])
    idx2 = columns.get_loc(view["ticker2"])
    A = R[:, idx1] - R[:, idx2]
    b = view["value"]
    return A, b, view["type"], view.get("direction", "le")

  def _view_volatility(self, view, R, columns, theta):
    idx = columns.get_loc(view["ticker"])
    A = (R[:, idx] - theta["mu"][idx]) ** 2
    b = view["value"] ** 2
    return A, b, view["type"], view.get("direction", "le")

  def _view_correlation(self, view, R, columns, theta):
    idx1 = columns.get_loc(view["ticker1"])
    idx2 = columns.get_loc(view["ticker2"])

    vol_1 = np.sqrt(theta["var"][idx1])
    vol_2 = np.sqrt(theta["var"][idx2])
    rho = view["value"]
    cov_target = rho * vol_1 * vol_2

    A = (R[:, idx1] - theta["mu"][idx1]) * (R[:, idx2] - theta["mu"][idx2])
    return A, cov_target, view["type"], view.get("direction", "le")

  def _sort_views(self, views):
    sorted_views = {
      "mean": [],
      "relative": [],
      "volatility": [],
      "correlation": [],
    }

    for view in views:
      view_type = view["view"]
      if view_type not in sorted_views:
        raise ValueError(f"Unsupported view type: {view_type}")
      sorted_views[view_type].append(view)

    return sorted_views


