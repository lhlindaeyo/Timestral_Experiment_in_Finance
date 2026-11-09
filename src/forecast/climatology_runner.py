"""무조건 경험적 분위수 기준선: origin 시점까지의 과거 지표값 전체(또는 최근 lookback)의 분위수.

모델이 이걸 못 이기면 의미가 없다는 점검용 기준선.
"""
from __future__ import annotations

import time

import numpy as np


class ClimatologyRunner:
    name = "climatology"
    uses_context = False

    def __init__(self, cfg: dict | None = None):
        self.lookback = (cfg or {}).get("lookback")
        self.last_ms_per_forecast: float | None = None

    def predict(self, x, origins, horizon, ctx, levels, key="default"):
        t0 = time.perf_counter()
        valid0 = int(np.flatnonzero(~np.isnan(x))[0])
        out = np.empty((len(origins), len(levels)))
        for k, i in enumerate(origins):
            lo = valid0 if self.lookback is None else max(valid0, i + 1 - int(self.lookback))
            out[k] = np.quantile(x[lo:i + 1], levels)
        self.last_ms_per_forecast = (time.perf_counter() - t0) / max(len(origins), 1) * 1000
        return out
