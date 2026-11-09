"""Timestral 크기별 로컬 추론 지연시간 측정 (합성 시계열, 결과는 DataFrame 으로 반환)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.common import load_config, quantile_levels
from src.forecast.timestral_runner import TimestralRunner


def measure_latency(sizes=("5M", "28M", "123M", "209M"), ctx: int = 1024, horizon: int = 20,
                    n: int = 64, batch_size: int = 32, device: str | None = None) -> pd.DataFrame:
    cfg = dict(load_config("experiment")["timestral"])
    cfg["batch_size"] = batch_size
    if device:
        cfg["device"] = device
    rng = np.random.default_rng(0)
    x = np.log1p(np.abs(rng.standard_normal(ctx + n + 10)).cumsum() * 0.01)
    origins = np.arange(ctx - 1, ctx - 1 + n)
    levels = quantile_levels()
    rows = []
    for s in sizes:
        r = TimestralRunner(s, cfg)
        t0 = time.perf_counter(); r._load(); load_s = time.perf_counter() - t0
        r.predict(x, origins[:batch_size], horizon, ctx, levels)       # warm-up
        r.predict(x, origins, horizon, ctx, levels)
        rows.append({"model": f"Timestral-{s}", "load_s": round(load_s, 1), "ctx": ctx, "horizon": horizon,
                     "batch": batch_size, "device": cfg["device"],
                     "ms_per_forecast": round(r.last_ms_per_forecast, 1)})
    return pd.DataFrame(rows)
