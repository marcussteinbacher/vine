# A Dynamic Vine Copula Based Portfolio Value-at-Risk and Expected Shortfall analysis
This package offers several tools for the modelling, analysis and backtesting of portfolio value-at-risk (VaR) and expected shortfall (ES).
A complete risk model requires at least the following steps. Every step offers several command-line tools.
> To run a script copy/move it from `/scripts` into root, e.g. where `config.py` is located!

## 1. Portfolio Construction
The starting point for each model is a specific portfolio, e.g. a portfolio with 20 assets. The portfolio returns must be located in the `data/20/portfolio_returns.parquet`.
I offer several stratgies to construct a fully populated portfolio out of a set of incomplete asset return series:
- `RandomImputation`: Imputes individual time-series with randomly chosen donor series.
- `MaximumCoverageImputation`: Imputes individual time-series (receiver) with the donor with the most reported returns on the receiver's missing days.
- `MaximumCorrelationImputation`: Completes the receiver with a donor series with the biggest pair-wise correlation (Spearman, Kendall, or Pearson).
- `MaximumSimilarityImputation`: Completes each receiver with the donor's series which scores the highest similarity, e.g. dynamic time-warping (DTW), Euclidean Distance, etc.

> Each imputation strategy can be inititated with a set of series which must be in the portfolio, e.g. the current Nasdaq 100 constituents.

```python
import pandas as pd
from tools.Imputation import MaximumCoverageImputation

# Load daily returns
returns = pd.read_parquet(...)

# Set portfolio initialization, titles that must be in the portfolio in its full extent
init = ["NVDA","MSFT","AAPL"]

# Set portfolio dimension
N = 20

# Instantiate the imputer
imputer = MaximumCoverageImputation(returns, tickers=init, n=N)

# Tuple of the daily portfolio composition and return
df_p, df_r = imputer.impute()
```

## 2. Choosing a Volatility Model
To adjust the log-returns to the current level of volatility one must choose a volatility model. All returns are subsequently adjusted in a rolling window manner with the temporarily aligned one-day ahead conditional risk foecasts.
All models assume constant mean. Implemented are:
- Garch, a symmetrical volatility process, eg. Garch(1,1)
- GJR-Garch, an asymmtrical volatility process, e.g. GJR-Garch(1,1,1)
- EGARCH, symmetrical or asymmetrical
- EWMA, e.g. EWMA(lambda=0.94)

Every volatility process must be paired with an assumption about the distribution of the innovations. Implemented are:
- Normal
- Student
- Empirical
- GeneralizedError

To create the volatility forecasts, run

```batch
cp scripts/run_volatility_forecasts.py .
python run_volatility_forecasts.py -p 20 -vm Garch -id Empirical
```
... for all options see `python run_volatility_forecasts.py --help`.

This will store batch-wise calculated risk forecasts and model summaries in `/temp`. These must be aggregated seperately with

```batch
cp scripts/build_volatility_data.py .
python build_volatility_data.py
```
> This will save the aggregated files, e.g. `volatility_forecasts.parquet`, into `data/20/Garch/Empirical/`. Make sure to create the folder(s) in advance!

## 3. Choosing Risk Model & Resolution Method
The one day-ahead VaR and ES forecasts are calculated in a rolling window manner based on adjusted returns which are calculated from the volatility forecasts in the previous step.
Implemented are:
- Historical Simualtion, `scripts/run_historical_simulation.py`
- Variance Covariance, `scripts/run_variance_covariance.py`
- Multivariate Copula , `scripts/run_multivariate_copula.py`
- Vine Copula, `scripts/run_vine_copula.py`

To create a risk forecast for an the previous portfolio with a Garch volatility model and Empirical innovations, e.g. a historical simualtion, run

```batch
cp scripts/run_historical_simulation.py .
python run_historical_simulation.py -p 20 -vm Garch -id Empirical
```

...for all options see `python run_historical_simulation.py --help`.

This will save the VaR & ES forecasts directly into `data/20/Garch/Empirical/HistoricalSimulation/`.

The risk forecasts are calculated concurrently in `tools.Runner` with auto-resumption if the calculation is aborted which makes it ready for cloud compute spot instances. Figure 1 shows the architecture of the parallel computation for a multivariate copula VaR & ES estimation.

![Runner](assets/runner_architecture.svg)

> The folder structure of a completed simulation is `data/{portfolio}/{volatility_process}/{innovation_distribution}/{simulation}/[{copula}][{/margin_distribution/}]`. A simulation is complete if there is `VaR.parquet` or `ES.parquet` in the child directory.

All completed simulation paths can be inspected via

```bash
source .bashrc
show_completed_simulations
```

## Model analysis & backtesting
The `tools` module offers several analysis tools, e.g. network visualisation and inspection tools for vine copula graphs in `tools.Graphs`.

The `backtest` module offers several VaR and ES backtests. Among others, 
- Kupiec (1995)
- Christofferson (1998)
- McNeil and Frey (2000)