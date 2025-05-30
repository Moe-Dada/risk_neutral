# risk-neutral
[![PyPI version](https://badge.fury.io/py/riskneutral.svg)](https://badge.fury.io/py/riskneutral)
[![Python Version](https://img.shields.io/pypi/pyversions/riskneutral.svg)](https://pypi.org/project/riskneutral)
[![Build Status](https://github.com/Moe-Dada/risk_neutral/actions/workflows/ci.yml/badge.svg)](https://github.com/Moe-Dada/risk_neutral/actions)
[![Coverage Status](https://coveralls.io/repos/github/Moe-Dada/risk_neutral/badge.svg?branch=main)](https://coveralls.io/github/Moe-Dada/risk_neutral?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Risk-Neutral Density Estimation Tools

A Python library for pricing options under various risk-neutral density assumptions, computing option-implied densities, and extracting model parameters from market data.

## Features

* **Option Pricers:** Black–Scholes–Merton (BSM), Mixture of Log-Normals for American Options (AMLN), Edgeworth Expansion (EW), Shimko Spline Method, and Mixture of Log-Normals (MLN).
* **Density Models:** Compute risk-neutral probability density functions for priced assets under different distributional assumptions.
* **Density Extraction:** Calibrate density model parameters to market option prices via optimization (BSM, AM, EW, MLN) or direct implied-volatility inversion (Shimko).
* **Utilities:** Implied volatility computation (`compute_implied_vol`), volatility-curve fitting (`fit_iv_curve`).

## Installation

```bash
pip install git+https://github.com/Moe-Dada/risk_neutral.git
```

Or clone and install locally:

```bash
git clone https://github.com/Moe-Dada/risk_neutral.git
cd risk_neutral
pip install .
```

## Quickstart

### Option Pricing

```python
from riskneutral.core_pricing import MarketParams, BSMParams, BSMPricer

# Market parameters: spot, risk-free rate, dividend yield
data = MarketParams(s0=100.0, r=0.05, y=0.02)
# BSM parameters: strike, time to expiry, volatility
params = BSMParams(k=100.0, te=1.0, sigma=0.2)
pricer = BSMPricer(market=data, params=params)
prices = pricer.price()
print(prices)  # {'d1': ..., 'd2': ..., 'call': ..., 'put': ...}
```

Switch pricer class for different models:

* `AMPricer` (Mixture of Log-Normals for American Options)
* `EWPricer` (Edgeworth Expansion)
* `ShimkoPricer` (Shimko Spline Method)
* `MLNPricer` (Mixture of Log-Normals)

### Density Computation

```python
import numpy as np
from riskneutral.density_computations import EwDensity, EWParams, MarketParams

mp = MarketParams(s0=100.0, r=0.05, y=0.02)
params = EWParams(k=100.0, te=1.0, sigma=0.2, skew=0.0, kurt=3.0)
model = EwDensity(market=mp, params=params)
x = np.linspace(50, 150, 200)
pdf = model.pdf(x)
```

Other density classes:

* `ShimkoDensity` (local-volatiltity skew)
* `AmDensity` (mixture-lognormal)
* `MlnDensity` (two-component lognormal mixture)

### Density Extraction

```python
import numpy as np
from riskneutral.density_extraction import DensityData, BsmDensityExtractor, BsmExtractConfig
from riskneutral.core_pricing import BSMPricer, BSMParams, MarketParams

# Simulate market option prices under BSM
r, y, te, s0, sigma = 0.03, 0.01, 0.5, 100.0, 0.25
strikes = np.array([90, 100, 110])
market_calls = np.array([
    BSMPricer(MarketParams(s0, r, y), BSMParams(K, te, sigma)).price()["call"]
    for K in strikes
])

data = DensityData(
    r=r, y=y, te=te, s0=s0,
    market_calls=market_calls,
    call_strikes=strikes
)
extractor = BsmDensityExtractor(data, BsmExtractConfig(lam=0.0))
result = extractor.extract()
print("Estimated mu, zeta:", result.params)
```

For other extractors, see `AmDensityExtractor`, `EwDensityExtractor`, `MlnDensityExtractor`, or use `ShimkoDirectExtractor` for implied-volatility-based extraction.

## Examples

- See the `examples/` folder for complete scripts demonstrating pricing, density computation, and extraction workflows.
- See `examples/examples_density_computations_and_plots.py` for density plots.
For example:
- ![shimko_density_plot](https://github.com/user-attachments/assets/adf9e34b-53f1-4367-8c10-327eb647db3d)
- ![mln_density_plot](https://github.com/user-attachments/assets/1b926ea5-065e-4048-ba09-4ceb14da0d06)
- ![ew_density_plot](https://github.com/user-attachments/assets/d49f396d-9f01-41a3-bf99-a1ccc7f9067f)
- ![american_mln_density_plot](https://github.com/user-attachments/assets/f285a12d-070d-4215-98b6-c45e35e67226)



## Testing

Run the full test suite with:

```bash
pytest
```

## Contributing

Contributions and issues are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
