#core_pricing.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad


class OptionPricer(ABC):
    @abstractmethod
    def price(self) -> Dict[str, float]:
        """Compute and return option prices (and diagnostics)."""
        pass

@dataclass
class MarketParams:
    s0: float
    r: float
    y: float

@dataclass
class BSMParams:
    k: float
    te: float
    sigma: float

@dataclass
class BSMPricer(OptionPricer):
    market: MarketParams
    params: BSMParams

    def price(self) -> Dict[str, float]:
        s0, r, y = self.market.s0, self.market.r, self.market.y
        k, te, sigma = self.params.k, self.params.te, self.params.sigma
        d1 = (np.log(s0 / k) + (r - y + 0.5 * sigma ** 2) * te) / (sigma * np.sqrt(te))
        d2 = d1 - sigma * np.sqrt(te)
        call = s0 * np.exp(-y * te) * norm.cdf(d1) - k * np.exp(-r * te) * norm.cdf(d2)
        put = k * np.exp(-r * te) * norm.cdf(-d2) - s0 * np.exp(-y * te) * norm.cdf(-d1)
        return {"d1": d1, "d2": d2, "call": call, "put": put}


@dataclass
class AMParams:
    k: float
    te: float
    w: float
    mus: Tuple[float, float, float]
    sigmas: Tuple[float, float, float]
    p1: float
    p2: float

@dataclass
class AMPricer(OptionPricer):
    market: MarketParams
    params: AMParams

    def price(self) -> Dict[str, float]:
        r = self.market.r
        k, te, w = self.params.k, self.params.te, self.params.w
        u1, u2, u3 = self.params.mus
        sigma1, sigma2, sigma3 = self.params.sigmas
        p1, p2 = self.params.p1, self.params.p2
        p3 = 1 - p1 - p2

        mu_vec = np.array([u1, u2, u3])
        sigma_vec = np.array([sigma1, sigma2, sigma3])
        prop_vec = np.array([p1, p2, p3])

        ef0 = np.sum(prop_vec * np.exp(mu_vec + 0.5 * sigma_vec ** 2))
        prob_gr_k = np.sum(prop_vec * (1 - norm.cdf((np.log(k) - mu_vec) / sigma_vec)))
        prob_ls_k = 1 - prob_gr_k
        ef0_gr_k = (
            np.sum(
                prop_vec * np.exp((2 * mu_vec + sigma_vec ** 2) / 2)
                * (1 - norm.cdf((np.log(k) - mu_vec - sigma_vec ** 2) / sigma_vec))
            )
            / prob_gr_k
        )
        ef0_ls_k = (ef0 - ef0_gr_k * prob_gr_k) / prob_ls_k

        call_val = w * (ef0_gr_k - k) * prob_gr_k + (1 - w) * max(
            ef0 - k,
            np.exp(-r * te) * (ef0_gr_k - k) * prob_gr_k
        )
        call_val = max(0, call_val)
        put_val = w * (k - ef0_ls_k) * prob_ls_k + (1 - w) * max(
            k - ef0,
            np.exp(-r * te) * (k - ef0_ls_k) * prob_ls_k
        )
        put_val = max(0, put_val)

        return {
            "call": call_val,
            "put": put_val,
            "expected_f0": ef0,
            "prob_f0_gt_k": prob_gr_k,
            "prob_f0_lt_k": prob_ls_k,
            "expected_f0_gt_k": ef0_gr_k,
            "expected_f0_lt_k": ef0_ls_k,
        }

@dataclass
class EWParams:
    k: float
    te: float
    sigma: float
    skew: float
    kurt: float

@dataclass
class EWPricer(OptionPricer):
    market: MarketParams
    params: EWParams

    def price(self) -> Dict[str, float]:
        s0, r, y = self.market.s0, self.market.r, self.market.y
        k, te, sigma, skew, kurt = (
            self.params.k,
            self.params.te,
            self.params.sigma,
            self.params.skew,
            self.params.kurt
        )
        discount = np.exp(-r * te)

        # BSM baseline
        d1 = (np.log(s0 / k) + (r - y + 0.5 * sigma ** 2) * te) / (sigma * np.sqrt(te))
        d2 = d1 - sigma * np.sqrt(te)
        cp = s0 * np.exp(-y * te) * norm.cdf(d1) - k * np.exp(-r * te) * norm.cdf(d2)

        # cumulant adjustments
        v = np.sqrt(np.exp(sigma ** 2 * te) - 1)
        m = np.log(s0) + (r - y - 0.5 * sigma ** 2) * te
        skew_ln = 3 * v + v ** 3
        kurt_ln = 16 * v ** 2 + 15 * v ** 4 + 6 * v ** 6 + v ** 8
        cumul = (s0 * np.exp((r - y) * te) * v) ** 2

        dln = (1 / (k * sigma * np.sqrt(te))) * norm.pdf((np.log(k) - m) / (sigma * np.sqrt(te)))
        d1_ln = -(1 + ((np.log(k) - m) / (te * sigma ** 2))) * dln / k
        d2_ln = -(2 + ((np.log(k) - m) / (te * sigma ** 2))) * d1_ln / k - dln / (k ** 2 * sigma ** 2)

        ew_call = (
            cp
            - discount * (skew - skew_ln) * cumul ** 1.5 * d1_ln / 6
            + discount * (kurt - kurt_ln) * cumul ** 2 * d2_ln / 24
        )
        ew_put = ew_call + k * np.exp(-r * te) - s0 * np.exp(-y * te)

        return {"call": ew_call, "put": ew_put}


@dataclass
class ShimkoParams:
    k: float
    te: float
    a0: float
    a1: float
    a2: float


@dataclass
class ShimkoPricer(OptionPricer):
    market: MarketParams
    params: ShimkoParams

    def price(self) -> Dict[str, float]:
        s0, r, y = self.market.s0, self.market.r, self.market.y
        k, te, a0, a1, a2 = (
            self.params.k,
            self.params.te,
            self.params.a0,
            self.params.a1,
            self.params.a2,
        )
        sigma_k = a0 + a1 * k + a2 * k ** 2
        mu = np.log(s0) + (r - y - 0.5 * sigma_k ** 2) * te

        def integrand(x):
            return ((x - k) / x) * (1 / (np.sqrt(2 * np.pi * te) * sigma_k)) * np.exp(
                -((np.log(x) - mu) ** 2) / (2 * sigma_k ** 2 * te)
            )

        call = np.exp(-r * te) * quad(integrand, k, np.inf, limit=500)[0]
        put = call + np.exp(-r * te) * k - s0 * np.exp(-y * te)
        return {"call": call, "put": put}

@dataclass
class MLNParams:
    te: float
    k: float
    alpha1: float
    meanlog1: float
    meanlog2: float
    sdlog1: float
    sdlog2: float

@dataclass
class MLNPricer(OptionPricer):
    market: MarketParams
    params: MLNParams

    def price(self) -> Dict[str, float]:
        r, y = self.market.r, self.market.y
        te, k = self.params.te, self.params.k
        a1 = self.params.alpha1
        ml1, ml2 = self.params.meanlog1, self.params.meanlog2
        sd1, sd2 = self.params.sdlog1, self.params.sdlog2

        discount = np.exp(-r * te)
        ev1 = np.exp(ml1 + 0.5 * sd1 ** 2)
        ev2 = np.exp(ml2 + 0.5 * sd2 ** 2)
        s0 = np.exp((y - r) * te) * (a1 * ev1 + (1 - a1) * ev2)

        u1 = (np.log(k) - ml1) / sd1
        tmp1 = ev1 * (1 - norm.cdf(u1 - sd1)) - k * (1 - norm.cdf(u1))
        c1 = discount * tmp1

        u2 = (np.log(k) - ml2) / sd2
        tmp2 = ev2 * (1 - norm.cdf(u2 - sd2)) - k * (1 - norm.cdf(u2))
        c2 = discount * tmp2

        call = a1 * c1 + (1 - a1) * c2
        put = call - s0 * np.exp(-y * te) + k * discount
        return {"call": call, "put": put, "s0": s0}