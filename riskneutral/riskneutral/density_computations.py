# density_computations.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.stats import norm, lognorm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# import pricer classes from core_pricing.py
from riskneutral.core_pricing import MarketParams, EWParams, MLNParams, ShimkoParams, AMParams

# ----------------------
# density blueprint infrastructure
# ----------------------

class DensityModel(ABC):
    """
    Abstract base for density models. Each model must implement pdf(x).
    """
    @abstractmethod
    def pdf(self, x:np.ndarray) -> np.ndarray:
        pass

class EwDensity(DensityModel):
    def __init__(self, market: MarketParams, params: EWParams):
        self.r = market.r
        self.y = market.y
        self.s0 = market.s0
        self.te = params.te
        self.sigma = params.sigma
        self.skew = params.skew
        self.kurt = params.kurt

    def pdf(self, x: np.ndarray) -> np.ndarray:
        v = np.sqrt(np.exp(self.sigma**2 * self.te) - 1)
        m = np.log(self.s0) + (self.r - self.y - 0.5 * self.sigma**2) * self.te
        skew_ln = 3 * v + v**3
        kurt_ln = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8
        cumul = (self.s0 * np.exp((self.r - self.y) * self.te) * v)**2

        # lognormal density
        pdf_ln = lognorm.pdf(x, s=self.sigma * np.sqrt(self.te), scale=np.exp(m))
        # derivatives
        dens = pdf_ln
        frst = - (1 + (np.log(x) - m) / (self.te * self.sigma**2)) * dens / x
        scnd = - (2 + (np.log(x) - m) / (self.te * self.sigma**2)) * frst / x - dens / (x**2 * self.sigma**2)
        thrd = - (3 + (np.log(x) - m) / (self.te * self.sigma**2)) * scnd / x - 2 * frst / (x**2 * self.sigma**2) + dens / (x**3 * self.sigma**2)
        frth = - (4 + (np.log(x) - m) / (self.te * self.sigma**2)) * thrd / x - 3 * scnd / (x**2 * self.sigma**2)
        frth += 3 * frst / (x**3 * self.sigma**2) - 2 * dens / (x**4 * self.sigma**2)

        return dens - (self.skew - skew_ln) * cumul**1.5 * thrd / 6 + (self.kurt - kurt_ln) * cumul**2 * frth / 24
@dataclass
class ShimkoDensityParams:
    r: float
    te: float
    s0: float
    y: float
    a0: float
    a1: float
    a2: float

class ShimkoDensity(DensityModel):
    def __init__(self, market: MarketParams, params: ShimkoParams):
        self.r, self.s0, self.y = market.r, market.s0, market.y
        self.te, self.a0, self.a1, self.a2 =  params.te, params.a0, params.a1, params.a2

    def pdf(self, x: np.ndarray) -> np.ndarray:
        sigma_k = self.a0 + self.a1 * x + self.a2 * x**2
        v = sigma_k * np.sqrt(self.te)
        d1 = (np.log(self.s0 / x) + (self.r - self.y + 0.5 * sigma_k**2) * self.te) / v
        d2 = d1 - v
        d1x = -1/(x * v) + (1 - d1/v) * (self.a1 + 2 * self.a2 * x)
        d2x = d1x - (self.a1 + 2 * self.a2 * x)
        return - norm.pdf(d2) * (d2x - (self.a1 + 2 * self.a2 * x) * (1 - d2 * d2x) - 2 * self.a2 * x)

class AmDensity(DensityModel):
    def __init__(self,params: AMParams):
        self.k, self.te, self.w =   params.k, params.te, params.w
        self.u1, self.u2, self.u3 = params.mus
        self.s1, self.s2, self.s3 = params.sigmas
        self.p1, self.p2 = params.p1, params.p2
        self.p3 = 1 - self.p1 - self.p2


    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (self.p1 * lognorm.pdf(x, s=self.s1, scale=np.exp(self.u1)) +
                self.p2 * lognorm.pdf(x, s=self.s2, scale=np.exp(self.u2)) +
                self.p3 * lognorm.pdf(x, s=self.s3, scale=np.exp(self.u3)))

class MlnDensity(DensityModel):
    def __init__(self, params: MLNParams):
        self.a1 = params.alpha1
        self.a2 = 1 - self.a1
        self.ml1, self.ml2 = params.meanlog1, params.meanlog2
        self.sd1, self.sd2 = params.sdlog1, params.sdlog2

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return (self.a1 * lognorm.pdf(x, s=self.sd1, scale=np.exp(self.ml1)) +
                self.a2 * lognorm.pdf(x, s=self.sd2, scale=np.exp(self.ml2)))
