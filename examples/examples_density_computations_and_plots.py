if __name__ == '__main__':
    import numpy as np
    # import pricer classes from core_pricing.py and density_computations.py
    from riskneutral.density_computations import *
    from riskneutral.core_pricing import *
    import matplotlib.pyplot as plt

    # ----------------------
    # EW ----------------------
    # Given inputs
    r = 0.05
    y = 0.03
    s0 = 1000.0  # example spot price
    market = MarketParams(s0=s0, r=r, y=y)
    te = 100 / 365
    sigma = 0.25
    strikes = np.arange(600, 1401, 1)
    # baseline skew and kurtosis from log-normal
    v = np.sqrt(np.exp(sigma ** 2 * te) - 1)
    ln_skew = 3 * v + v ** 3
    ln_kurt = 16 * v ** 2 + 15 * v ** 4 + 6 * v ** 6 + v ** 8

    skew_1 = ln_skew * 1.50
    kurt_1 = ln_kurt * 1.50

    ew_params = EWParams(k=strikes, te=te, sigma=sigma, skew=skew_1, kurt=kurt_1)
    density_ew = EwDensity(market=market, params=ew_params)
    print(density_ew.pdf(x=strikes))
    plt.figure(figsize=(14, 8))
    # Make the density plot
    plt.plot(strikes, density_ew.pdf(x=strikes))
    plt.xlabel("k")
    plt.ylabel("dx")
    plt.show()

    # Shimko ----------------------
    market = MarketParams(
        s0=400.0,   # current spot price
        r=0.05,     # risk-free rate (5%)
        y=0.02      # dividend yield (2%)
    )
    # 2. Shimko-specific parameters
    #    a0 + a1*K + a2*K^2 defines the K-dependent vol surface
    shimko_params = ShimkoParams(
        k=950.0,    # strike
        te=60/365,     # time to expiry in years
        a0=0.892,    # base vol term
        a1=-0.00387,  # linear term
        a2=0.00000445,    # quadratic term
    )
    strikes = np.arange(250, 500, 1)

    density_shimko = ShimkoDensity(market=market, params=shimko_params)
    print(density_shimko.pdf(x=strikes))
    plt.figure(figsize=(14, 8))
    # Make the density plot
    plt.plot(strikes, density_shimko.pdf(x=strikes))
    plt.xlabel("k")
    plt.ylabel("dx")
    plt.show()


    # AM ----------------------
    # AM mixture parameters
    w = 0.4
    u1, u2, u3 = 4.2, 4.5, 4.8
    sigma1, sigma2, sigma3 = 0.30, 0.20, 0.15
    p1, p2 = 0.25, 0.45
    strikes = np.arange(0, 250, 1)

    # Set up AM params
    am_params = AMParams(
        k=strikes,
        te=te,
        w=w,
        mus=(u1, u2, u3),
        sigmas=(sigma1, sigma2, sigma3),
        p1=p1,
        p2=p2
    )


    density_am = AmDensity(params=am_params)
    print(density_am.pdf(x=strikes))
    plt.figure(figsize=(14, 8))
    # Make the density plot
    plt.plot(strikes, density_am.pdf(x=strikes))
    plt.xlabel("k")
    plt.ylabel("dx")
    plt.show()

    # MLN ----------------------
    # Given MLN inputs
    alpha1 = 0.4  # mixture weight for component 1
    meanlog1 = 6.3  # µ₁ (log-space mean)
    meanlog2 = 6.5  # µ₂
    sdlog1 = 0.08  # σ₁ (log-space stdev)
    sdlog2 = 0.06  # σ₂

    strikes = np.arange(300, 900, 1)

    mln_params = MLNParams(
        k=strikes,
        te=te,
        alpha1=alpha1,
        meanlog1=meanlog1,
        meanlog2=meanlog2,
        sdlog1=sdlog1,
        sdlog2=sdlog2
    )

    density_mln = MlnDensity(params=mln_params)
    print(density_mln.pdf(x=strikes))
    plt.figure(figsize=(14, 8))
    # Make the density plot
    plt.plot(strikes, density_mln.pdf(x=strikes))
    plt.xlabel("k")
    plt.ylabel("dx")
    plt.show()

