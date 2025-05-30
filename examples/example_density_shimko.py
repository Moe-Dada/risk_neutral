# example_density_shimko.py

from riskneutral.density_extraction import *
from riskneutral.density_computations import *
# ----------------------
# Example: synthetic Shimko density extraction
# ----------------------
if __name__ == '__main__':
    # Given parameters
    r = 0.05
    y = 0.02
    te = 60 / 365
    s0 = 1000.0
    sigma = 0.25

    # Strike grid
    k = np.arange(800, 1201, 5)

    # Generate synthetic BSM call prices
    bsm_calls = np.array([
        BSMPricer(
           market=MarketParams(s0=s0, r=r, y=y),
           params=BSMParams(k=Ki, te=te, sigma=sigma)
        ).price()['call']
        for Ki in k
    ])

    # Pack into DensityData
    data = DensityData(
        r=r, y=y, te=te, s0=s0,
        market_calls=bsm_calls,
        call_strikes=k
    )

    cfg = ShimkoDirectExtractConfig()
    extractor = ShimkoDirectExtractor(data, cfg)
    res = extractor.extract()
    print('shimko_density: ',res.shimko_density)
    print('implied_curve: ',res.implied_curve)
    print('implied_vols: ',res.implied_vols)

