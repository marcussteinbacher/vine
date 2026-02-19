# A Dynamic Vine Copula Based Portfolio Value-at-Risk and Expected Shortfall analysis
This package offers several tools for the modelling, analysing and backtesting portfolio value-at-risk (VaR) and expected shortfall (ES).
A complete risk model requires at least the following steps. Every step offers several command-line tools.
> To run a script copy/move it from `/scripts` into root, e.g. where `config.py` is located!
## 1. Portfolio Construction
The starting point for each model is a specific portfolio, e.g. a portfolio with 20 assets. The portfolio returns must be located in the `data/20/portfolio_returns.parquet`.
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
python run_volatility_forecasts.py -p 20 -vm Garch -id Empirical
```
... for all options see `python run_volatility_forecasts.py --help`.

This will store batch-wise calculated risk forecasts and model summaries in `/temp`. These must be aggregated seperately with

```batch
python build_volatility_dataframes.py
```
> This will save the aggregated files, e.g. `volatility_forecasts.parquet`, into `data/20/Garch/Empirical/`. Make sure to create the folder(s) in advance!

## 3. Choosing Risk Model & Resolution Method
The one day-ahead VaR and ES forecasts are calculated in a rolling window manner based on adjusted returns which are calculated from the volatility forecasts in the previous step.
Implemented are:
- Historical Simualtion, `run_historical_simulation.py`
- Variance Covariance, `run_variance_covariance.py`
- Multivariate Copula , `run_multivariate_copula.py` & `build_risk_dataframes.py`
- Vine Copula, `run_vine_copula.py` & `build_risk_dataframes.py`

To create a risk forecast for an the previous portfolio with a Garch volatility model and Empirical innovations, e.g. a historical simualtion, run

```batch
python run_historical_simulation.py -p 20 -vm Garch -id Empirical -r VaR -a 0.05
```

for all options see `python run_historical_simulation.py --help`.

This will save the 5%-VaR forecasts directly into `data/20/Garch/Empirical/HistoricalSimulation/`.

> The folder structure of a completed simulation is `data/{portfolio}/{volatility_process}/{innovation_distribution}/{simulation}/[{copula}][{/margin_distribution/}]`. A simulation is complete if there is `VaR.parquet` or `ES.parquet` in the child directory.

All completed simulation paths can be inspected via

```bash
source .bashrc
show_completed_simulations
```
## Backtesting
