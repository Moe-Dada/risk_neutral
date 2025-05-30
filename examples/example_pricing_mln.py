# example_pricing_mln.py

from riskneutral.core_pricing import *
if __name__ == "__main__":

# Given MLN inputs
    r = 0.05
    te = 60 / 365
    s0 = 100.0
    y = 0.02


    # ----------------------
    # MLNPricer standalone example
    # ----------------------
    alpha1=0.4     # mixture weight for component 1
    meanlog1=6.80   # µ₁ (log-space mean)
    meanlog2=6.95   # µ₂
    sdlog1=0.065    # σ₁ (log-space stdev)
    sdlog2=0.055     # σ₂

    k = 100.0  # strike(s)
    market = MarketParams(s0=s0, r=r, y=y)

    mln_params = MLNParams(
        k=k,
        te=te,
        alpha1=alpha1,
        meanlog1=meanlog1,
        meanlog2=meanlog2,
        sdlog1=sdlog1,
        sdlog2=sdlog2
    )
    mln_pricer = MLNPricer(market=market, params=mln_params)
    mln_res = mln_pricer.price()

    print("MLNPricer Results:")
    print(f"  Implied Spot (s0) = {mln_res['s0']:.6f}")
    print(f"  Call  = {mln_res['call']:.6f}")
    print(f"  Put   = {mln_res['put']:.6f}")

    # compute MLNPricer prices for several strikes
    mln_params_2  = MLNParams(
        k=np.arange(800, 1201, 50),
        te=te,
        alpha1=alpha1,
        meanlog1=meanlog1,
        meanlog2=meanlog2,
        sdlog1=sdlog1,
        sdlog2=sdlog2
    )
    mln_pricer_2 = MLNPricer(market=market, params=mln_params_2)
    mln_prices_2 = mln_pricer_2.price()



    print("Baseline MLNPricer:")
    print("Strike | MLNPricer Call")
    for K, mln in zip(np.arange(800, 1201, 50), mln_prices_2['call']):
        print(f"{K:6.0f} | {mln:.4f}")