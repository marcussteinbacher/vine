#!/bin/bash

set -e 

show_help() {
    echo "Helper script to run all scripts for a portfolio risk analysis of PORTFOLIO and arbitrary models in a single call."
    echo "Builds the folder structure, runs volatility forecast, runs all simulations and aggregates the results if necessary."
    echo "Info: The portfolio, i.e. 'portfolio_returns.parquet', must be created first!"
    echo ""
    echo "Options:"
    echo "  -h, --help  Show this help message and exit"
    echo "  PORTFOLIO   The desired portfolio"
    echo ""
    echo "Example:"
    echo "  ./run.sh 20"
}

if [[ "$#" -eq 0 ]]; then
    show_help
    exit 1
fi

case "$1" in
    -h|--help)
        show_help
        exit 0
        ;;
esac

DIR="./scripts"

# ------ General Model Parameters ------ 
PORTFOLIO=$1
VM="Garch"
ID="Empirical"
CP="Student"
MD="Empirical"

# ------ Build Simulation Folder Structure ------
mkdir -p "data/$PORTFOLIO/$VM/$ID/HistoricalSimulation"
mkdir -p "data/$PORTFOLIO/$VM/$ID/VarianceCovariance"
mkdir -p "data/$PORTFOLIO/$VM/$ID/MultivariateCopula/$CP/$MD"
mkdir -p "data/$PORTFOLIO/$VM/$ID/VineCopula/$MD"

# ------ Volatility Forecasts ------
cp "$DIR/run_volatility_forecasts.py" .
cp "$DIR/build_volatility_data.py" .

python run_volatility_forecasts.py -p "$PORTFOLIO" -vm "$VM" -id "$ID"
python build_volatility_data.py

rm run_volatility_forecasts.py
rm build_volatility_data.py

# ------ Historical Simulation ------
cp "$DIR/run_historical_simulation.py" .

python run_historical_simulation.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -r VaR
python run_historical_simulation.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -r ES

rm run_historical_simulation.py

# ------ VarianceCovariance ------
cp "$DIR/run_variance_covariance.py" .

python run_variance_covariance.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -r VaR
python run_variance_covariance.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -r ES

rm run_variance_covariance.py

# ------ MultivariateCopula ------
cp "$DIR/run_multivariate_copula.py" .
cp "$DIR/build_risk_data.py" .

python run_multivariate_copula.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -cp "$CP" -md "$MD" -fm itau --controls df=3 --save_freq 250
python build_risk_data.py

rm run_multivariate_copula.py

# ------ VineCopula ------
cp "$DIR/run_vine_copula.py" .

python run_vine_copula.py -p "$PORTFOLIO" -vm "$VM" -id "$ID" -md "$MD" -fm itau --save_freq 250
python build_risk_data.py

rm run_vine_copula.py
rm build_risk_data.py