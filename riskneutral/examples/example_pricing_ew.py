# example_pricing_ew.py
import numpy as np
from riskneutral.core_pricing import *
if __name__ == "__main__":
    te, sigma = 100/365, 0.25
    # baseline skew and kurtosis from log-normal
    v = np.sqrt(np.exp(sigma**2 * te) - 1)
    ln_skew = 3 * v + v**3
    ln_kurt = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8

    market = MarketParams(r=0.05, y=0.03, s0=1000.0)
    # compute EWPricer prices
    ew_params = EWParams(k=np.arange(800, 1201, 50), te=100/365, sigma=0.25, skew=ln_skew, kurt=ln_kurt)
    ew_pricer = EWPricer(market=market, params=ew_params)
    ew_prices = ew_pricer.price()



    print("Baseline EWPricer:")
    print("Strike | EWPricer Call")
    for K, ec in zip(np.arange(800, 1201, 50), ew_prices['call']):
        print(f"{K:6.0f} | {ec:.4f}")

