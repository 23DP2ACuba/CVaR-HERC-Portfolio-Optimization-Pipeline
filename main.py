from PortfolioModel import CVaRHERCPortfolio
from model.DataStore import ModelParams
from ForwardTest import ForwardTest

if __name__ == "__main__":
    test_universe = ['ARB', 'COM', 'CTA', 'DBMF', 'FTLS', 'PDBC', 'SPY']
    params = ModelParams()

    model = CVaRHERCPortfolio(
      debug=True,
      params=params,
    )
    returns, benchmark = model.get_data(
      universe=test_universe,
      start="2019-05-09",
      end="2026-10-07",
      benchmark_ticker="^GSPC"
    )

    model = CVaRHERCPortfolio(params=params, debug=True)

    model = CVaRHERCPortfolio(params=params, debug=True)
    ft = ForwardTest(model, leverage=2, risk_free_rate=0.04,
                                          risk_tolerance_threshold=-0.02, periods_per_year=252,
                                          n_forward_batches=1)
    results = ft.run(returns, label="2026-Q3")

    print(results[0])
    print(results[1])
