# example_pricing_shimko.py
from riskneutral.core_pricing import *

def main():
    # 1. Market inputs
    market = MarketParams(
        s0=1000.0,   # current spot price
        r=0.05,     # risk-free rate (5%)
        y=0.02      # dividend yield (2%)
    )

    # 2. Shimko-specific parameters
    #    a0 + a1*K + a2*K^2 defines the K-dependent vol surface
    shimko_params = ShimkoParams(
        k=950.0,    # strike
        te=60/365,     # time to expiry in years
        a0=0.30,    # base vol term
        a1=-0.00387,  # linear term
        a2=0.00000445,    # quadratic term
    )

    # 3. Instantiate and compute
    pricer = ShimkoPricer(market=market, params=shimko_params)
    result = pricer.price()

    # 4. Display results
    print("Shimko Model Results:")
    print(f"  Call Price = {result['call']:.4f}")
    print(f"  Put  Price = {result['put']:.4f}")

if __name__ == "__main__":
    main()
