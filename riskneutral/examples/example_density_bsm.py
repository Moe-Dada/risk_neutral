# example_density_bsm.py
from riskneutral.density_extraction import *
from riskneutral.density_computations import *

# ----------------------
# Example: synthetic BSM density extraction
# ----------------------
r = 0.05
y = 0.01
te = 60 / 365
s0 = 1000.0
sigma = 0.25

# Generate call and put strike grids
call_strikes = np.arange(500, 1501, 25)
put_strikes  = np.arange(510, 1501, 25)

# Generate synthetic market prices via BSMPricer
market_calls = np.array([
    BSMPricer(market=MarketParams(s0=s0, r=r, y=y), params=BSMParams(k=K, te=te, sigma=sigma)).price()['call']
    for K in call_strikes
])
market_puts = np.array([
    BSMPricer(market=MarketParams(s0=s0, r=r, y=y), params=BSMParams(k=K, te=te, sigma=sigma)).price()['put']
    for K in put_strikes
])

# Pack into DensityData and extract parameters
bsm_data = DensityData(
    r=r, y=y, te=te, s0=s0,
    market_calls=market_calls, call_strikes=call_strikes,
    market_puts=market_puts, put_strikes=put_strikes
)
bsm_cfg = BsmExtractConfig(lam=1.0, hessian_flag=False)
bsm_ext = BsmDensityExtractor(bsm_data, bsm_cfg)
bsm_res = bsm_ext.extract()

# Print named parameters
mu_est, zeta_est = bsm_res.params
print(f"mu   = {mu_est:.6f}")
print(f"zeta = {zeta_est:.6f}")
print("Converged:", bsm_res.convergence)

# Compare to actual values
actual_mu   = np.log(s0) + (r - y - 0.5 * sigma**2) * te
actual_zeta = sigma * np.sqrt(te)
print(f"actual mu   = {actual_mu:.6f}")
print(f"actual zeta = {actual_zeta:.6f}")
