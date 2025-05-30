#test_core_pricing.py
import numpy as np
import pytest
from riskneutral.core_pricing import (
    BSMPricer, MarketParams, BSMParams,
    EWPricer, EWParams,
    ShimkoPricer, ShimkoParams,
    MLNPricer, MLNParams,
    AMPricer, AMParams
)

# ----------------------
# BSM Pricer Tests
# ----------------------
def test_bsmp_put_call_parity():
    s0, k, r, y, te, sigma = 100.0, 90.0, 0.05, 0.02, 1.0, 0.2
    pr = BSMPricer(MarketParams(s0, r, y), BSMParams(k, te, sigma))
    res = pr.price()
    call, put = res['call'], res['put']
    lhs = call + k * np.exp(-r * te)
    rhs = put + s0 * np.exp(-y * te)
    assert np.isclose(lhs, rhs, atol=1e-8)


# ----------------------
# EW Pricer Tests
# ----------------------
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_ew_matches_bsm_when_zero_skew_kurt(strike):
    s0, r, y, te, sigma = 100.0, 0.05, 0.02, 0.5, 0.2
    # baseline BSM price
    bsm = BSMPricer(
        MarketParams(s0, r, y), BSMParams(strike, te, sigma)
    ).price()
    # lognormal skew/kurt
    v = np.sqrt(np.exp(sigma**2 * te) - 1)
    skew_ln = 3 * v + v**3
    kurt_ln = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8
    ew = EWPricer(
        MarketParams(s0, r, y),
        EWParams(strike, te, sigma, skew_ln, kurt_ln)
    ).price()
    assert np.isclose(ew['call'], bsm['call'], rtol=1e-6)
    assert np.isclose(ew['put'], bsm['put'], rtol=1e-6)

# ----------------------
# Shimko Pricer Tests
# ----------------------
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_shimko_reduces_to_bsm(strike):
    s0, r, y, te, sigma = 100.0, 0.05, 0.02, 1.0, 0.2
    # Shimko with a1=a2=0, a0=sigma
    shim = ShimkoPricer(
        MarketParams(s0, r, y), ShimkoParams(strike, te, sigma, 0.0, 0.0)
    ).price()
    bsm = BSMPricer(
        MarketParams(s0, r, y), BSMParams(strike, te, sigma)
    ).price()
    assert np.isclose(shim['call'], bsm['call'], rtol=1e-4)
    assert np.isclose(shim['put'], bsm['put'], rtol=1e-4)

# ----------------------
# MLN Pricer Tests
# ----------------------
@pytest.mark.parametrize("alpha1", [0.0, 0.5, 1.0])
def test_mln_pricer_properties(alpha1):
    s0, r, y, te, k = 100.0, 0.03, 0.01, 1.0, 100.0
    mlparams = MLNParams(te, k, alpha1, meanlog1=0.0, meanlog2=0.0, sdlog1=0.2, sdlog2=0.3)
    res = MLNPricer(MarketParams(s0, r, y), mlparams).price()
    # keys exist
    assert set(res.keys()) == {"call", "put", "s0"}
    # non-negativity
    assert res['call'] >= 0
    assert res['put'] >= 0

# ----------------------
# AM Pricer Tests
# ----------------------
def test_am_expected_f0_calculation():
    s0, r, y, te, k = 100.0, 0.03, 0.01, 1.0, 100.0
    w = 0.5
    u1, u2, u3 = 0.0, 0.0, 0.0
    s1, s2, s3 = 0.1, 0.1, 0.1
    p1, p2 = 0.3, 0.3
    pr = AMPricer(
        MarketParams(s0, r, y),
        AMParams(k, te, w, (u1, u2, u3), (s1, s2, s3), p1, p2)
    )
    res = pr.price()
    # expected_f0 where u=0 simplifies
    expected = np.sum(np.array([p1, p2, 1-p1-p2]) * np.exp(0 + 0.5 * np.array([s1, s2, s3])**2))
    assert np.isclose(res['expected_f0'], expected)
    # call and put non-negative
    assert res['call'] >= 0
    assert res['put'] >= 0
