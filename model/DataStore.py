import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ClusterParams:
  quantiles: dict[str, float] = field(default_factory=dict)
  paths: np.ndarray = None

@dataclass
class TailFit:
  c_L: float
  scale_L: float
  c_U: float
  scale_U: float
  p_l: float
  p_u: float
  u_lower: float
  u_upper: float
  ecdf_data: np.ndarray

@dataclass
class CVaRHERCParams:
  vine_structure:np.ndarray=None
  tau_prev:np.ndarray=None
  vol_init:dict = field(default_factory=dict)
  r_init:dict = field(default_factory=dict)
  best_models:dict = field(default_factory=dict)
  best_fits:dict = field(default_factory=dict)
  best_lags:dict = field(default_factory=dict)
  marginal_distributions:dict = field(default_factory=dict)
  fit:bool=False

@dataclass
class ModelParams:
  distance_method:str="kendalltau"
  gap_bootstrap_samples:int=100
  cvar_q:float=0.95
  k_max:int=8
  shrink_target_ess:int=100
  shrink:bool=True
  min_effective_sample:int=30
  tail_power:float=0.8
  tail_threshold:float=0.80
  tau_shrinkage_params:float=0.20
  use_cluster_mvo:bool=True
  n_bootstrap:int = 1000
  block_size:Optional[int] = None
  seed:Optional[int]=None
  scale_factor:int=1000
  sobol_pow:int = 17
  n_steps:int = 63
  lags:int = 10
  gamma_var:float = 2
  gamma_cvar:float = 2
  lambda_anchor:float = 0.5

  def __post_init__(self):
    allowed_methods = {"kendalltau", "pearsoncorr"}
    if self.distance_method not in allowed_methods:
      raise ValueError(f"Invalid status. Choose from {allowed_methods}")

    if not (0 <= self.cvar_q <= 1):
      raise ValueError("Quantile must be between 0 and 1")



class DataStore:
  def __init__(self, debug:bool=False, **kwargs):
    super().__init__(
      debug=debug,
      **kwargs
    )
    self.debug = debug

  def _get_data(
      self,
      universe:dict,
      start:str,
      end:str,
      interval:str="1d",
      benchmark:str="^GSPC"
  ):
    tickers_clean = list(set(universe))

    self.benchmark_ticker = benchmark
    df_path = f"portfolio_{start}_{end}.parquet"

    if not os.path.exists(df_path):
      if benchmark not in tickers_clean:
        tickers_clean.append(benchmark)

      df = yf.download(tickers_clean, start, end, interval)["Close"]

      df.to_parquet(df_path)

    else:
      df = pd.read_parquet(df_path)

    bench_data = df["^GSPC"]
    data_raw = df.drop(columns=["^GSPC"])

    benchmark = bench_data.pct_change().dropna()
    returns = data_raw.pct_change().dropna()
    self.universe = data_raw.columns

    return returns, benchmark

  def plot_data(self):
    (np.cumsum(self.returns_raw * 100, axis=0) + 100).plot(figsize=(15, 10))
    plt.show()

  def plot_benchmark(self):
    (np.cumsum(self.benchmark * 100, axis=0) + 100).plot(figsize=(15, 10))
    plt.show()
