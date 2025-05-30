# example_density_ew.py
# ----------------------
# Example: synthetic EW density extraction
# ----------------------
from riskneutral.density_extraction import *
from riskneutral.density_computations import *

r = 0.05
y = 0.03
te = 100 / 365
s0 = 1000.0
sigma = 0.25

# Strike grid for EW
strikes = np.arange(600, 1401, 1)

# Baseline lognormal moments
v = np.sqrt(np.exp(sigma**2 * te) - 1)
ln_skew = 3 * v + v**3
ln_kurt = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8

# Generate synthetic market calls via EWPricer
market_calls = EWPricer(
    market=MarketParams(s0=s0, r=r, y=y),
    params=EWParams(k=strikes, te=te, sigma=sigma, skew=ln_skew, kurt=ln_kurt)
).price()['call']

# Pack into DensityData and extract EW parameters
ew_data = DensityData(
    r=r, y=y, te=te, s0=s0,
    market_calls=market_calls, call_strikes=strikes
)
ew_cfg = EwExtractConfig(lam=1.0, hessian_flag=False)
ew_ext = EwDensityExtractor(ew_data, ew_cfg)
ew_res = ew_ext.extract()

# Print named EW parameters
sigma_est, skew_est, kurt_est = ew_res.params
print(f"sigma = {sigma_est:.6f}")
print(f"skew  = {skew_est:.6f}")
print(f"kurt  = {kurt_est:.6f}")
print("Converged:", ew_res.convergence)
