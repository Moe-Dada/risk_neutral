# example_pricing_bsm.py

from riskneutral.core_pricing import *
import numpy as np
def main():
    # 1. Set up market parameters
    market = MarketParams(
        s0=42.0,   # spot price
        r=0.10,     # risk-free rate (5%)
        y=0.0      # dividend yield (2%)
    )

    # 2. Define BSM option params
    bsm = BSMParams(
        k=40.0,    # strike
        te=0.5,     # time to expiry in years
        sigma=0.2   # volatility (20%)
    )

    # 3. Instantiate pricer and compute
    pricer = BSMPricer(market=market, params=bsm)
    result = pricer.price()

    # 4. Display the output
    print("BSM Results:")
    print(f"  d1 = {result['d1']:.4f}")
    print(f"  d2 = {result['d2']:.4f}")
    print(f"  Call = {result['call']:.4f}")
    print(f"  Put  = {result['put']:.4f}")

if __name__ == '__main__':
    main()
# call should be 4.76, put should be 0.81, from Hull 8th, page 315, 316
# BSM Results:
#   d1 = 0.7693
#   d2 = 0.6278
#   Call = 4.7594
#   Put  = 0.8086