# example_density_mln.py

from riskneutral.density_extraction import *
from riskneutral.density_computations import *
# ----------------------
# Example: synthetic MLN density extraction
# ----------------------
r = 0.05
y = 0.02
te = 60 / 365
meanlog1 = 6.8
meanlog2 = 6.95
sdlog1 = 0.065
sdlog2 = 0.055
alpha1 = 0.4

##
#r, y, te = 0.05, 0.02, 60 / 365

# Strike grids
call_strikes = np.arange(800, 1201, 10)
put_strikes  = np.arange(805, 1201, 10)

##
#strikes = np.arange(800, 901, 25)
#meanlog1, meanlog2 = 4.5, 4.7
#sd1, sd2 = 0.1, 0.15
# Generate synthetic market via MLNPricer
# Calls
mln_pricer_calls = MLNPricer(
    market=MarketParams(s0=0, r=r, y=y),  # s0 placeholder, overwritten by pricer
    params=MLNParams(
        k=call_strikes, te=te,
        alpha1=alpha1,
        meanlog1=meanlog1, meanlog2=meanlog2,
        sdlog1=sdlog1, sdlog2=sdlog2
    )
)
call_prices = mln_pricer_calls.price()
market_calls = call_prices['call']
# Implied s0 from pricer
implied_s0 = call_prices['s0']

# Puts
mln_pricer_puts = MLNPricer(
    market=MarketParams(s0=implied_s0, r=r, y=y),
    params=MLNParams(
        k=put_strikes, te=te,
        alpha1=alpha1,
        meanlog1=meanlog1, meanlog2=meanlog2,
        sdlog1=sdlog1, sdlog2=sdlog2
    )
)
market_puts = mln_pricer_puts.price()['put']

print(f"Implied s0 = {implied_s0:.6f}")

# Pack into DensityData and extract MLN parameters
mln_data = DensityData(
    r=r, y=y, te=te, s0=implied_s0,
    market_calls=market_calls, call_strikes=call_strikes,
    market_puts=market_puts, put_strikes=put_strikes
)
mln_cfg = MlnExtractConfig(lam=1.0, hessian_flag=False)
mln_ext = MlnDensityExtractor(mln_data, mln_cfg)
mln_res = mln_ext.extract()

# Print named MLN parameters
param_names = ['alpha1', 'meanlog1', 'meanlog2', 'sdlog1', 'sdlog2']
for name, val in zip(param_names, mln_res.params):
    print(f"{name} = {val:.6f}")
print("Converged:", mln_res.convergence)

# Compare to actual values
print(f"actual alpha1  = {alpha1:.6f}")
print(f"actual meanlog1= {meanlog1:.6f}")
print(f"actual meanlog2= {meanlog2:.6f}")
print(f"actual sdlog1  = {sdlog1:.6f}")
print(f"actual sdlog2  = {sdlog2:.6f}")