"""ARIMA 베이스라인 (모델 스케일로 변환된 시계열에 적합).

* 차수: 지수/시계열별로 한 번만 선택 (ADF 로 d in {0,1} 결정 -> p,q 는 AIC 격자 탐색), 테스트 이전 데이터만 사용
* 파라미터: origin 'refit_every' 개마다 재적합, 그 사이에는 파라미터 고정 + 새 데이터로 필터링(smooth)
* 예측 분포: h 스텝 평균 + 예측표준오차 x (정규 분위수 | 표준화 잔차의 경험 분위수)
"""
from __future__ import annotations

import logging
import time
import warnings

import numpy as np
from scipy.stats import norm

log = logging.getLogger(__name__)


def _make_model(endog, order):
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA(endog, order=order, trend="c" if order[1] == 0 else "n")


def select_order(train: np.ndarray, max_p: int = 2, max_q: int = 2) -> tuple[int, int, int]:
    from statsmodels.tsa.stattools import adfuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        d = 0 if adfuller(train, autolag="AIC")[1] < 0.05 else 1
    best, best_aic = (1, d, 0), np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    aic = _make_model(train, (p, d, q)).fit().aic
                except Exception:
                    continue
                if np.isfinite(aic) and aic < best_aic:
                    best, best_aic = (p, d, q), aic
    return best


class ArimaRunner:
    name = "arima"
    uses_context = False

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.orders: dict[str, tuple[int, int, int]] = {}
        self.last_ms_per_forecast: float | None = None

    def predict(self, x, origins, horizon, ctx, levels, key: str = "default"):
        ctx_len = int(self.cfg.get("context", 1024))
        refit_every = int(self.cfg.get("refit_every", 20))
        dist = self.cfg.get("dist", "bootstrap")
        z_normal = norm.ppf(levels)

        if key not in self.orders:
            n_sel = int(self.cfg.get("order_select_obs", 1500))
            train = x[:origins[0] + 1]
            train = train[~np.isnan(train)][-n_sel:]
            self.orders[key] = select_order(train, self.cfg.get("max_p", 2), self.cfg.get("max_q", 2))
            log.info("ARIMA order [%s] = %s", key, self.orders[key])
        order = self.orders[key]

        out = np.full((len(origins), len(levels)), np.nan)
        params = None
        z_emp = None
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, i in enumerate(origins):
                endog = x[max(0, i + 1 - ctx_len):i + 1]
                endog = endog[~np.isnan(endog)]
                try:
                    mod = _make_model(endog, order)
                    if params is None or k % refit_every == 0:
                        res = mod.fit()
                        params = res.params
                    else:
                        res = mod.smooth(params)
                    fc = res.get_forecast(steps=horizon)
                    mu = float(np.asarray(fc.predicted_mean)[-1])
                    se = float(np.asarray(fc.se_mean)[-1])
                    if dist == "bootstrap":
                        if z_emp is None or k % refit_every == 0:
                            r = np.asarray(res.resid)[20:]               # 초기 상태 잔차 제외
                            r = r - r.mean()
                            z_emp = r / r.std()
                        out[k] = mu + se * np.quantile(z_emp, levels)
                    else:
                        out[k] = mu + se * z_normal
                except Exception as e:                                    # 수렴 실패 등
                    log.warning("ARIMA 실패 (origin=%s): %s", i, e)
        self.last_ms_per_forecast = (time.perf_counter() - t0) / max(len(origins), 1) * 1000
        return out
