# test_density_computation.py
import numpy as np
import pytest
from riskneutral.density_computations import EwDensity, ShimkoDensity, AmDensity, MlnDensity
from riskneutral.core_pricing import MarketParams, EWParams, ShimkoParams, AMParams, MLNParams
from scipy.stats import lognorm

# ----------------------
# EwDensity Tests
# ----------------------
def test_ew_density_reduces_to_lognorm():
    # For zero correction (skew/kurt = lognormal values), EwDensity = lognormal pdf
    r, y, te, s0, sigma = 0.05, 0.02, 1.0, 100.0, 0.2
    v = np.sqrt(np.exp(sigma**2 * te) - 1)
    skew_ln = 3 * v + v**3
    kurt_ln = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8
    market = MarketParams(s0=s0, r=r, y=y)
    params = EWParams(k=100.0, te=te, sigma=sigma, skew=skew_ln, kurt=kurt_ln)
    model = EwDensity(market, params)

    x = np.linspace(1e-3, 300, 1000)
    pdf_model = model.pdf(x)
    # corresponding lognormal pdf under risk-neutral drift
    m = np.log(s0) + (r - y - 0.5 * sigma**2) * te
    pdf_true = lognorm.pdf(x, s=sigma * np.sqrt(te), scale=np.exp(m))
    assert pdf_model.shape == x.shape
    assert np.allclose(pdf_model, pdf_true, rtol=1e-6)

# ----------------------
# AmDensity Tests
# ----------------------
@pytest.mark.parametrize("p1,p2", [(0.2,0.3), (0.4,0.5)])
def test_am_density_integrates_to_one(p1, p2):
    u1, u2, u3 = 4.0, 4.1, 4.2
    s1, s2, s3 = 0.2, 0.3, 0.4
    params = AMParams(
        k=90, te=1.0, w=0.5,
        mus=(u1, u2, u3), sigmas=(s1, s2, s3),
        p1=p1, p2=p2
    )
    model = AmDensity(params)
    x = np.linspace(1e-3, 1000, 100000)
    pdf = model.pdf(x)
    # non-negative
    assert np.all(pdf >= 0)
    # integral ~1
    integral = np.trapezoid(pdf, x)
    assert np.isclose(integral, 1.0, rtol=1e-3)

# ----------------------
# MlnDensity Tests
# ----------------------
@pytest.mark.parametrize("alpha1", [0.2, 0.5, 0.8])
def test_mln_density_integrates_to_one(alpha1):
    ml1, ml2 = 4.0, 4.5
    sd1, sd2 = 0.25, 0.35
    params = MLNParams(
        te=1.0, k=100.0,
        alpha1=alpha1, meanlog1=ml1, meanlog2=ml2, sdlog1=sd1, sdlog2=sd2
    )
    model = MlnDensity(params)
    x = np.linspace(1e-3, 1000, 100000)
    pdf = model.pdf(x)
    assert np.all(pdf >= 0)
    integral = np.trapezoid(pdf, x)
    assert np.isclose(integral, 1.0, rtol=1e-3)

# ----------------------
# ShimkoDensity Tests
# ----------------------
@pytest.mark.parametrize("a0", [0.1, 0.2, 0.3])
def test_shimko_density_integrates_to_one(a0):
    r, y, te, s0 = 0.05, 0.02, 1.0, 100.0
    # tests with a1=a2=0 (constant vol a0)
    params = ShimkoParams(k=100.0, te=te, a0=a0, a1=0.0, a2=0.0)
    model = ShimkoDensity(MarketParams(s0=s0, r=r, y=y), params)
    x = np.linspace(1e-3, 1000, 100000)
    pdf = model.pdf(x)
    # enforce non-negativity
    assert np.all(pdf >= -1e-6)
    integral = np.trapezoid(np.clip(pdf, 0, None), x)
    assert np.isclose(integral, 1.0, rtol=1e-2)
