# density_extraction.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.optimize import brentq
from riskneutral.core_pricing import AMPricer, EWPricer, MLNPricer, MarketParams, BSMParams, BSMPricer, ShimkoPricer, ShimkoParams
# ----------------------
# Core extractor infrastructure
# ----------------------
@dataclass
class DensityData:
    r: float
    y: float
    te: float
    s0: float
    market_calls: np.ndarray
    call_strikes: np.ndarray
    call_weights: Optional[np.ndarray] = None
    market_puts: Optional[np.ndarray] = None
    put_strikes: Optional[np.ndarray] = None
    put_weights: Optional[np.ndarray] = None

@dataclass
class ExtractConfig:
    initial_values: Optional[np.ndarray] = None
    lam: float = 1.0
    hessian_flag: bool = False
    options: dict = field(default_factory=lambda: {'maxiter': 10000})

@dataclass
class ExtractionResult:
    params: np.ndarray
    convergence: bool
    hessian: Optional[np.ndarray]

class ObjectiveFunction(ABC):
    def __init__(self, data: DensityData, lam: float):
        self.data = data
        self.lam = lam
    @abstractmethod
    def __call__(self, theta: np.ndarray) -> float:
        pass

class DensityExtractor(ABC):
    def __init__(self, data: DensityData, config: ExtractConfig):
        self.data = data
        self.config = config
        self.objective = self._build_objective()
    @abstractmethod
    def _build_objective(self) -> ObjectiveFunction:
        pass
    @abstractmethod
    def _initial_grid_search(self) -> np.ndarray:
        pass
    def extract(self) -> ExtractionResult:
        theta0 = self.config.initial_values
        if theta0 is None or np.any(np.isnan(theta0)):
            theta0 = self._initial_grid_search()
        res = minimize(fun=self.objective, x0=theta0, options=self.config.options)
        hess = res.hess_inv.todense() if (self.config.hessian_flag and hasattr(res, 'hess_inv')) else None
        return ExtractionResult(params=res.x, convergence=res.success, hessian=hess)

# ----------------------
# BSM extraction/objective
# ----------------------
@dataclass
class BsmExtractConfig(ExtractConfig): pass

class BsmObjective(ObjectiveFunction):
    def __call__(self, theta: np.ndarray) -> float:
        mu, zeta = theta
        d = self.data
        # weights
        wc = d.call_weights
        if wc is None or wc.size == 0:
            wc = np.ones_like(d.call_strikes)
        wp = d.put_weights
        if wp is None or wp.size == 0:
            wp = np.ones_like(d.put_strikes)
        # invalid parameter
        if zeta <= 0:
            return 1e7
        df = np.exp(-d.r * d.te)
        ev = np.exp(mu + 0.5 * zeta**2)
        # calls
        d1 = (np.log(d.call_strikes) - mu - zeta**2) / zeta
        d2 = d1 + zeta
        theo_c = df * (ev * (1 - norm.cdf(d1)) - d.call_strikes * (1 - norm.cdf(d2)))
        # puts
        p1 = (np.log(d.put_strikes) - mu - zeta**2) / zeta
        p2 = p1 + zeta
        theo_p = df * (d.put_strikes * norm.cdf(p2) - ev * norm.cdf(p1))
        # loss
        loss = np.sum(wc * (theo_c - d.market_calls)**2) + np.sum(wp * (theo_p - d.market_puts)**2)
        loss += self.lam * (d.s0 * np.exp(-d.y * d.te) - ev * np.exp(-d.r * d.te))**2
        return loss

class BsmDensityExtractor(DensityExtractor):
    def _build_objective(self):
        return BsmObjective(self.data, self.config.lam)
    def _initial_grid_search(self):
        d = self.data
        band = (d.r - d.y - 0.5 * 0.3**2) * d.te
        mus = np.linspace(np.log(d.s0) - band, np.log(d.s0) + band, 10)
        zetas = np.sqrt(d.te) * np.linspace(0.05, 0.9, 10)
        grid = np.array(np.meshgrid(mus, zetas)).T.reshape(-1, 2)
        vals = [self.objective(theta) for theta in grid]
        return grid[np.argmin(vals)]

# ----------------------
# AM extraction/objective
# ----------------------
@dataclass
class AmExtractConfig(ExtractConfig): pass

class AmObjective(ObjectiveFunction):
    def __call__(self, theta: np.ndarray) -> float:
        w1, w2, u1, u2, u3, s1, s2, s3, p1, p2 = theta
        p3 = 1 - p1 - p2
        # weights
        ws = self.data.call_weights
        if ws is None or ws.size == 0:
            wc = np.ones_like(self.data.call_strikes)
            wp = np.ones_like(self.data.call_strikes)
        else:
            wc = self.data.call_weights
            wp = self.data.put_weights if self.data.put_weights is not None else np.ones_like(self.data.call_strikes)
        # constraints
        if any([w1 < 0, w1 > 1, w2 < 0, w2 > 1, s1 < 0, s2 < 0, s3 < 0, p1 < 0, p2 < 0, p1 + p2 > 1]):
            return 1e7
        d = self.data
        ef0 = (p1 * np.exp(u1 + 0.5 * s1**2) + p2 * np.exp(u2 + 0.5 * s2**2) + p3 * np.exp(u3 + 0.5 * s3**2))
        loss = 0.0
        for i, K in enumerate(d.call_strikes):
            w = w1 if K < ef0 else w2
            pr = AMPricer(
                market=type('M', (), {'s0': d.s0, 'r': d.r, 'y': d.y})(),
                params=type('P', (), {'k': K, 'te': d.te, 'w': w,
                                      'mus': (u1, u2, u3),
                                      'sigmas': (s1, s2, s3),
                                      'p1': p1, 'p2': p2})()
            )
            res = pr.price()
            loss += wc[i] * (res['call'] - d.market_calls[i])**2 + wp[i] * (res['put'] - d.market_puts[i])**2
        loss += self.lam * (d.s0 - ef0 * np.exp(-d.r * d.te))**2
        return loss

class AmDensityExtractor(DensityExtractor):
    def _build_objective(self): return AmObjective(self.data, self.config.lam)
    def _initial_grid_search(self):
        d = self.data
        band = abs((d.r - 0.5 * 0.3**2) * d.te)
        grid = []
        for w1 in (0.1, 0.9):
            for w2 in (0.1, 0.9):
                for u1 in (np.log(d.s0) - 5 * band, np.log(d.s0) - 4 * band):
                    for u2 in (np.log(d.s0) - band, np.log(d.s0) + band):
                        for u3 in (np.log(d.s0) + 4 * band, np.log(d.s0) + 5 * band):
                            for s1 in (0.1, 0.4):
                                for s2 in (0.1, 0.4):
                                    for s3 in (0.1, 0.4):
                                        for p1 in (0.1, 0.5):
                                            for p2 in (0.2, 0.6):
                                                th = np.array([w1, w2, u1, u2, u3, s1, s2, s3, p1, p2])
                                                grid.append((th, self.objective(th)))
        return min(grid, key=lambda x: x[1])[0]

# ----------------------
# EW extraction/objective
# ----------------------
@dataclass
class EwExtractConfig(ExtractConfig): pass

class EwObjective(ObjectiveFunction):
    def __call__(self, theta: np.ndarray) -> float:
        sigma, skew, kurt = theta
        if sigma < 0 or kurt < 0:
            return 1e7
        d = self.data
        calls = EWPricer(
            market=type('M', (), {'s0': d.s0, 'r': d.r, 'y': d.y})(),
            params=type('P', (), {'k': d.call_strikes, 'te': d.te,
                                  'sigma': sigma, 'skew': skew, 'kurt': kurt})()
        ).price()['call']
        wc = d.call_weights
        if wc is None or wc.size == 0:
            wc = np.ones_like(d.call_strikes)
        loss = np.sum(wc * (calls - d.market_calls)**2)
        m = np.log(d.s0) + (d.r - d.y - 0.5 * sigma**2) * d.te
        ev = np.exp(m + 0.5 * sigma**2 * d.te)
        loss += self.lam * (d.s0 * np.exp(-d.y * d.te) - ev * np.exp(-d.r * d.te))**2
        return loss

class EwDensityExtractor(DensityExtractor):
    def _build_objective(self): return EwObjective(self.data, self.config.lam)
    def _initial_grid_search(self):
        d = self.data
        grid = []
        for sigma in np.arange(0.1, 0.91, 0.1):
            v = np.sqrt(np.exp(sigma**2 * d.te) - 1)
            skew_ln = 3 * v + v**3
            kurt_ln = 16 * v**2 + 15 * v**4 + 6 * v**6 + v**8
            th = np.array([sigma, skew_ln, kurt_ln])
            grid.append((th, self.objective(th)))
        return min(grid, key=lambda x: x[1])[0]

# ----------------------
# MLN extraction/objective
# ----------------------
@dataclass
class MlnExtractConfig(ExtractConfig): pass

class MlnObjective(ObjectiveFunction):
    def __call__(self, theta: np.ndarray) -> float:
        a1, ml1, ml2, sd1, sd2 = theta
        d = self.data
        if a1 < 0 or a1 > 1 or sd1 < 0 or sd2 < 0:
            return 1e7
        calls = MLNPricer(
            market=type('M', (), {'s0': d.s0, 'r': d.r, 'y': d.y})(),
            params=type('P', (), {'k': d.call_strikes, 'te': d.te,
                                  'alpha1': a1, 'meanlog1': ml1,
                                  'meanlog2': ml2, 'sdlog1': sd1, 'sdlog2': sd2})()
        ).price()['call']
        puts = MLNPricer(
            market=type('M', (), {'s0': d.s0, 'r': d.r, 'y': d.y})(),
            params=type('P', (), {'k': d.put_strikes, 'te': d.te,
                                  'alpha1': a1, 'meanlog1': ml1,
                                  'meanlog2': ml2, 'sdlog1': sd1, 'sdlog2': sd2})()
        ).price()['put']
        wc = d.call_weights if isinstance(d.call_weights, np.ndarray) else np.ones_like(d.call_strikes)
        wp = d.put_weights if isinstance(d.put_weights, np.ndarray) else np.ones_like(d.put_strikes)
        loss = np.sum(wc * (calls - d.market_calls)**2) + np.sum(wp * (puts - d.market_puts)**2)
        ev = a1*np.exp(ml1+0.5*sd1**2) + (1-a1)*np.exp(ml2+0.5*sd2**2)
        loss += self.lam * (d.s0 * np.exp(-d.y * d.te) - ev * np.exp(-d.r * d.te))**2
        return loss

class MlnDensityExtractor(DensityExtractor):
    def _build_objective(self): return MlnObjective(self.data, self.config.lam)

    def _initial_grid_search(self):
        d = self.data
        band = (d.r - d.y - 0.5 * 0.3**2) * d.te
        grid = []
        alphas = np.linspace(0.1, 0.9, 17)
        mean_logs = np.linspace(np.log(d.s0)-band, np.log(d.s0)+band, 4)
        sd_logs = np.sqrt(d.te) * np.linspace(0.05, 0.9, 7)
        for a in alphas:
            for m1 in mean_logs:
                for m2 in mean_logs:
                    for s1 in sd_logs:
                        for s2 in sd_logs:
                            theta = np.array([a, m1, m2, s1, s2])
                            grid.append((theta, self.objective(theta)))
        return min(grid, key=lambda x: x[1])[0]

# ----------------------
# Shimko direct extraction (implied-volatility based)
# ----------------------
from scipy.optimize import brentq

# Compute implied volatilities via BSM inversion
def compute_implied_vol(r: float, y: float, te: float, s0: float,
                        k: np.ndarray, call_prices: np.ndarray,
                        lower: float, upper: float) -> np.ndarray:
    vols = np.empty_like(call_prices)
    for i, (Ki, price) in enumerate(zip(k, call_prices)):
        # function whose root is the implied vol
        def f(sigma):
            bs = BSMPricer(market=MarketParams(s0=s0, r=r, y=y),
                           params=BSMParams(k=Ki, te=te, sigma=sigma)).price()['call']
            return bs - price
        vols[i] = brentq(f, lower, upper)
    return vols

# Fit a quadratic arc to implied vol curve: vol = a0 + a1*k + a2*k^2
def fit_iv_curve(vols: np.ndarray, k: np.ndarray) -> Tuple[float, float, float]:
    # fit quadratic: np.polyfit returns [a2, a1, a0]
    coeffs = np.polyfit(k, vols, 2)
    a2, a1, a0 = coeffs
    return a0, a1, a2

@dataclass
class ShimkoDirectExtractConfig(ExtractConfig):
    lower: float = -10.0
    upper: float = 10.0
@dataclass
class ImpliedVolCurve:
    a0: float
    a1: float
    a2: float

@dataclass
class ShimkoDirectResult:
    implied_curve: ImpliedVolCurve
    implied_vols: np.ndarray
    shimko_density: np.ndarray

class ShimkoDirectExtractor:
    """
    Performs Shimko density extraction via implied vol inversion and polynomial fitting.
    """
    def __init__(self, data: DensityData, config: ShimkoDirectExtractConfig):
        self.data = data
        self.config = config

    def extract(self) -> ShimkoDirectResult:
        d = self.data

        # implied vols
        vols = compute_implied_vol(
            r=d.r, y=d.y, te=d.te, s0=d.s0,
            k=d.call_strikes, call_prices=d.market_calls,
            lower=self.config.lower, upper=self.config.upper
        )
        # fit curve
        a0, a1, a2 = fit_iv_curve(vols, d.call_strikes)
        implied_curve = ImpliedVolCurve(a0=a0, a1=a1, a2=a2)
        # compute Shimko density via standalone density model
        from riskneutral.density_computations import ShimkoDensity, ShimkoDensityParams
        shimko_density = ShimkoDensity(market=MarketParams(s0=d.s0, r=d.r, y=d.y),
                    params=ShimkoParams(k=d.call_strikes, te=d.te, a0=a0, a1=a1, a2=a2)).pdf(d.call_strikes)
        return ShimkoDirectResult(implied_curve=implied_curve,
                                  implied_vols=vols,
                                  shimko_density=shimko_density)

