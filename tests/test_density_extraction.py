# tests/test_density_extraction.py

import numpy as np
import pytest

from riskneutral.density_extraction import (
    compute_implied_vol, fit_iv_curve,
    DensityData,
    BsmDensityExtractor, BsmExtractConfig,
    AmDensityExtractor, AmExtractConfig,
    EwDensityExtractor, EwExtractConfig,
    MlnDensityExtractor, MlnExtractConfig,
    ShimkoDirectExtractor, ShimkoDirectExtractConfig
)
from riskneutral.core_pricing import (
    MarketParams, BSMParams, BSMPricer,
    AMParams, AMPricer,
    EWParams, EWPricer,
    MLNParams, MLNPricer,
    ShimkoParams, ShimkoPricer
)

# 1) compute_implied_vol & fit_iv_curve

def test_compute_implied_vol_roundtrip():
    r, y, te, s0 = 0.03, 0.01, 0.5, 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    sigma_true = np.array([0.15, 0.20, 0.25])

    prices = np.array([
        BSMPricer(MarketParams(s0, r, y), BSMParams(K, te, sig)).price()['call']
        for K, sig in zip(strikes, sigma_true)
    ])
    implied = compute_implied_vol(r, y, te, s0, strikes, prices, lower=0.001, upper=1.0)
    assert np.allclose(implied, sigma_true, atol=1e-4)

def test_fit_iv_curve_known_quadratic():
    strikes = np.array([90.0, 100.0, 110.0])
    a0, a1, a2 = 0.1, 0.02, 0.001
    vols = a0 + a1*strikes + a2*strikes**2
    a0_fit, a1_fit, a2_fit = fit_iv_curve(vols, strikes)
    assert pytest.approx(a0_fit, rel=1e-6) == a0
    assert pytest.approx(a1_fit, rel=1e-6) == a1
    assert pytest.approx(a2_fit, rel=1e-6) == a2

# 2) BSM density extraction

def test_bsm_density_extraction_recovers_mu_zeta():
    r, y, te, s0, sigma = 0.05, 0.02, 60/365, 1000.0, 0.25
    strikes = np.arange(500, 701, 50)

    calls = np.array([
        BSMPricer(MarketParams(s0, r, y), BSMParams(K, te, sigma)).price()['call']
        for K in strikes
    ])
    puts = np.array([
        BSMPricer(MarketParams(s0, r, y), BSMParams(K, te, sigma)).price()['put']
        for K in strikes
    ])

    data = DensityData(r=r, y=y, te=te, s0=s0,
                       market_calls=calls, call_strikes=strikes,
                       market_puts=puts, put_strikes=strikes)
    res = BsmDensityExtractor(data, BsmExtractConfig(lam=0.0)).extract()

    mu_est, zeta_est = res.params
    mu_true = np.log(s0) + (r - y - 0.5*sigma**2)*te
    zeta_true = sigma * np.sqrt(te)

    assert np.isclose(mu_est, mu_true, atol=5e-3)
    assert np.isclose(zeta_est, zeta_true, atol=5e-3)
    # convergence flag can be False in edge cases, but params are still close

# 3) EW density extraction

def test_ew_density_extraction_recovers_skew_and_kurtosis():
    r, y, te, s0, sigma = 0.05, 0.03, 100/365, 1000.0, 0.25
    strikes = np.arange(800, 1201, 50)

    v = np.sqrt(np.exp(sigma**2 * te) - 1)
    ln_skew = 3*v + v**3
    ln_kurt = 16*v**2 + 15*v**4 + 6*v**6 + v**8

    skew_true, kurt_true = ln_skew, ln_kurt
    calls = EWPricer(
        MarketParams(s0, r, y),
        EWParams(k=strikes, te=te, sigma=sigma, skew=skew_true, kurt=kurt_true)
    ).price()['call']

    data = DensityData(r=r, y=y, te=te, s0=s0,
                       market_calls=calls, call_strikes=strikes)
    res = EwDensityExtractor(data, EwExtractConfig(lam=0.0)).extract()

    sigma_est, skew_est, kurt_est = res.params
    # Allow 5% rel. error on skew/kurt, 5% on sigma
    assert np.isclose(sigma_est, sigma, rtol=0.05)
    assert np.isclose(skew_est, skew_true, rtol=0.05)
    assert np.isclose(kurt_est, kurt_true, rtol=0.05)

# 4) MLN density extraction

@pytest.mark.parametrize("alpha1", [0.3, 0.6])
def test_mln_density_extraction_sanity(alpha1):
    r, y, te = 0.05, 0.02, 60/365
    ml1, ml2 = 4.5, 4.7
    sd1, sd2 = 0.1, 0.15
    strikes = np.arange(800, 901, 25)

    calls_res = MLNPricer(MarketParams(0, r, y),
                         MLNParams(te=te, k=strikes, alpha1=alpha1,
                                   meanlog1=ml1, meanlog2=ml2, sdlog1=sd1, sdlog2=sd2)
    ).price()
    calls, s0_est = calls_res['call'], calls_res['s0']
    puts = MLNPricer(MarketParams(s0_est, r, y),
                     MLNParams(te=te, k=strikes, alpha1=alpha1,
                               meanlog1=ml1, meanlog2=ml2, sdlog1=sd1, sdlog2=sd2)
    ).price()['put']

    data = DensityData(r=r, y=y, te=te, s0=s0_est,
                       market_calls=calls, call_strikes=strikes,
                       market_puts=puts, put_strikes=strikes)
    res = MlnDensityExtractor(data, MlnExtractConfig(lam=0.0)).extract()
    a1_est, ml1_est, ml2_est, sd1_est, sd2_est = res.params

    # Sanity
    assert 0.0 <= a1_est <= 1.0
    assert sd1_est > 0 and sd2_est > 0
    # Mean-logs roughly recovered within 5%
    assert pytest.approx(ml1_est, rel=0.05) == ml1
    assert pytest.approx(ml2_est, rel=0.05) == ml2

# 5) AM density extraction

def test_am_density_extraction_weights_in_bounds():
    r, y, te, s0 = 0.05, 0.02, 60/365, 100.0
    w1, w2 = 0.4, 0.3
    u1, u2, u3 = 4.0, 4.2, 4.4
    s1, s2, s3 = 0.2, 0.15, 0.1
    p1, p2 = 0.25, 0.35
    strikes = np.arange(50, 81, 10)

    calls, puts = [], []
    ef0 = p1*np.exp(u1+0.5*s1**2) + p2*np.exp(u2+0.5*s2**2) + (1-p1-p2)*np.exp(u3+0.5*s3**2)
    for K in strikes:
        w_call = w1 if K < ef0 else w2
        w_put  = w2 if K < ef0 else w1
        out = AMPricer(
            MarketParams(s0, r, y),
            AMParams(k=K, te=te, w=w_call, mus=(u1,u2,u3), sigmas=(s1,s2,s3), p1=p1, p2=p2)
        ).price()
        calls.append(out['call']); puts.append(out['put'])
    calls = np.array(calls); puts = np.array(puts)

    data = DensityData(r=r, y=y, te=te, s0=s0,
                       market_calls=calls, call_strikes=strikes,
                       market_puts=puts, put_strikes=strikes)
    res = AmDensityExtractor(data, AmExtractConfig(lam=0.0)).extract()
    w1_est, w2_est = res.params[:2]

    assert 0.0 <= w1_est <= 1.0
    assert 0.0 <= w2_est <= 1.0

# 6) Shimko direct extraction

def test_shimko_direct_extraction_constant_vol():
    r, y, te, s0, sigma = 0.05, 0.02, 60/365, 100.0, 0.2
    strikes = np.arange(80, 121, 10)
    calls = np.array([
        BSMPricer(MarketParams(s0, r, y), BSMParams(K, te, sigma)).price()['call']
        for K in strikes
    ])
    data = DensityData(r=r, y=y, te=te, s0=s0,
                       market_calls=calls, call_strikes=strikes)
    extractor = ShimkoDirectExtractor(data, ShimkoDirectExtractConfig(lower=0.05, upper=0.5))
    res = extractor.extract()
    a0, a1, a2 = res.implied_curve.a0, res.implied_curve.a1, res.implied_curve.a2

    assert np.isclose(a0, sigma, atol=1e-2)
    assert abs(a1) < 1e-2
    assert abs(a2) < 1e-2
    # implied vols nearly constant
    assert np.allclose(res.implied_vols, sigma, atol=1e-3)
