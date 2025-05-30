# example_density_am_mln.py
from riskneutral.density_extraction import *
from riskneutral.density_computations import *


# ----------------------
# Example: synthetic AM mixture of lognormal density extraction
# ----------------------
if __name__ == '__main__':
    # Synthetic market data_spx for AM model
    r = 0.01
    y = 0.0
    te = 60 / 365
    s0 = 100.0

    # AM mixture parameters
    w1, w2 = 0.4, 0.25
    u1, u2, u3 = 4.2, 4.5, 4.8
    s1, s2, s3 = 0.30, 0.20, 0.15
    p1, p2 = 0.25, 0.45
    p3 = 1 - p1 - p2

    # Compute expected f0
    ef0 = (p1 * np.exp(u1 + 0.5 * s1**2)
           + p2 * np.exp(u2 + 0.5 * s2**2)
           + p3 * np.exp(u3 + 0.5 * s3**2))
    print(f"Expected f0 = {ef0:.6f}")

    # Strike grid
    strikes = np.arange(50, 151)
    market_calls = np.zeros_like(strikes, dtype=float)
    market_puts = np.zeros_like(strikes, dtype=float)

    # Generate synthetic prices using AMPricer
    for i, K in enumerate(strikes):
        # choose weights based on moneyness
        w_call = w1 if K < ef0 else w2
        w_put = w2 if K < ef0 else w1

        pricer = AMPricer(
            market=MarketParams(s0=s0, r=r, y=y),
            params=AMParams(
                k=K, te=te, w=w_call,
                mus=(u1, u2, u3), sigmas=(s1, s2, s3),
                p1=p1, p2=p2
            )
        )
        res = pricer.price()
        market_calls[i] = res['call']
        market_puts[i] = res['put']

    # Display a snippet of synthetic data_spx
    print("Strike | Call | Put")
    for K, c, p in zip(strikes[:5], market_calls[:5], market_puts[:5]):
        print(f"{K:3d}    | {c:.4f} | {p:.4f}")

    # Pack into DensityData and extract parameters
    data = DensityData(
        r=r, y=y, te=te, s0=s0,
        market_calls=market_calls, call_strikes=strikes,
        market_puts=market_puts, put_strikes=strikes
    )
    config = AmExtractConfig()
    extractor = AmDensityExtractor(data, config)
    result = extractor.extract()

    print("Estimated AM density parameters: "), # Print parameter names and values
    param_names = ['w1', 'w2', 'u1', 'u2', 'u3', 's1', 's2', 's3', 'p1', 'p2']
    for name, val in zip(param_names, result.params):
        print(f"{name}: {val:.6f}")  # labelled output)
    print("Converged:", result.convergence)
    print("hessian:", result.hessian)

    # Compare to original theta vector
    original_theta = np.array([w1, w2, u1, u2, u3, s1, s2, s3, p1, p2])
    print("Original theta:", original_theta)