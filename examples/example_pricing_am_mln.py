# example_pricing_am_mln.py
# we compute the am mixture of lognormal density

if __name__ == '__main__':
    from riskneutral.core_pricing import *

    # Given inputs
    r = 0.01
    te = 60 / 365
    s0 = 100.0        # example spot price
    y = 0.0           # assume zero dividend yield

    # AM mixture parameters
    w = 0.4
    u1, u2, u3 = 4.2, 4.5, 4.8
    sigma1, sigma2, sigma3 = 0.30, 0.20, 0.15
    p1, p2 = 0.25, 0.45
    k = 100.0         # example strike

    # Set up market and AM params
    market = MarketParams(s0=s0, r=r, y=y)
    am_params = AMParams(
        k=k,
        te=te,
        w=w,
        mus=(u1, u2, u3),
        sigmas=(sigma1, sigma2, sigma3),
        p1=p1,
        p2=p2
    )

    # Instantiate pricer and compute
    pricer = AMPricer(market=market, params=am_params)
    result = pricer.price()

    print("AMPricer Results:")
    print(f"  Call  = {result['call']:.6f}")
    print(f"  Put   = {result['put']:.6f}")
    print(f"  E[f0] = {result['expected_f0']:.6f}")
    print(f"  Prob(f0>k) = {result['prob_f0_gt_k']:.6f}")
    print(f"  Prob(f0<k) = {result['prob_f0_lt_k']:.6f}")