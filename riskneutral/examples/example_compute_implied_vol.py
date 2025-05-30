# ----------------------
# Example: implied volatility scan (R-style)
# ----------------------
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # BSM implied vol inversion example
    from riskneutral.core_pricing import *
    from riskneutral.density_extraction import *
    from riskneutral.density_computations import *

    import numpy as np

    r = 0.05
    y = 0.02
    te = 60 / 365
    s0 = 400.0

    # sigma.range = seq(0.1, 0.8, by=0.05)
    sigma_range = np.arange(0.1, 0.801, 0.05)

    k_range = np.floor(np.linspace(300, 500, len(sigma_range))).astype(int)

    # Generate BSM call prices
    bsm_calls = np.array([
        BSMPricer(
            market=MarketParams(s0=s0, r=r, y=y),
            params=BSMParams(k=int(K), te=te, sigma=float(sig))
        ).price()['call']
        for sig, K in zip(sigma_range, k_range)
    ])
    print("sigma_range=", sigma_range)
    print("k_range=", k_range)
    print("BSM calls=", np.round(bsm_calls, 6))

    # Compute implied vols
    implied_vols = compute_implied_vol(
        r=r, y=y, te=te, s0=s0,
        k=k_range, call_prices=bsm_calls,
        lower=0.001, upper=0.999
    )
    print("Implied vols=", np.round(implied_vols, 6))
    # Check closeness
    print("Close to sigma_range?", np.allclose(implied_vols, sigma_range, atol=1e-3))
