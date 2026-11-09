"""Rolling-origin 백테스트.

origin t 에서 모델은 t 까지의 trailing 지표 시계열만 본다 (미래 가격 불사용).
"t 이후 h일의 지표" == "t+h 시점의 trailing 값" 이므로 정답은 series[t+h].
예측 저장 형식: 모델×지수×시계열×horizon×ctx 별 parquet, 컬럼
    origin_date, target_date, y_origin, y_true, q01 ... q99   (모두 원 스케일)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from src.common import FORECASTS, METRICS, ensure_dirs, load_config, qcols, quantile_levels
from src.data.process import load_processed
from src.forecast.base import get_forecaster
from src.targets.spec import SeriesSpec, load_experiments, load_series_specs

log = logging.getLogger(__name__)


def make_origins(raw: pd.Series, test_start: str, test_end: str | None, step: int,
                 horizon: int, max_ctx: int, live: bool = False) -> np.ndarray:
    """origin 위치(정수) 배열.

    모든 컨텍스트 길이·모델이 같은 origin 을 쓰도록, 가장 긴 컨텍스트가 채워지는 첫 위치 이후로 시작한다.
    live=True 면 마지막 날짜 하나만 (실전 예측용, 정답 없음).
    """
    n = len(raw)
    first_valid = int(np.flatnonzero(raw.notna().to_numpy())[0])
    if live:
        return np.array([n - 1])
    p0 = max(int(raw.index.searchsorted(pd.Timestamp(test_start))), first_valid + max_ctx - 1)
    last = n - 1 - horizon
    if test_end:
        last = min(last, int(raw.index.searchsorted(pd.Timestamp(test_end), side="right")) - 1)
    return np.arange(p0, last + 1, step)


def backtest_frame(raw: pd.Series, spec: SeriesSpec, forecaster, horizon: int, ctx: int | None,
                   origins: np.ndarray, key: str = "default") -> pd.DataFrame:
    """순수 계산 (파일 IO 없음): origin 별 원 스케일 분위수 DataFrame."""
    levels = quantile_levels()
    x = spec.to_model(raw.to_numpy())
    x[raw.isna().to_numpy()] = np.nan
    q_model = forecaster.predict(x, origins, horizon, ctx, levels, key=key)
    q_raw = spec.quantiles_to_raw(q_model)
    q_raw[np.isnan(q_model).any(axis=1)] = np.nan             # 실패한 origin 은 NaN 유지

    dates = raw.index
    vals = raw.to_numpy()
    tgt = origins + horizon
    ok = tgt < len(raw)
    df = pd.DataFrame({
        "origin_date": dates[origins],
        "target_date": pd.DatetimeIndex([dates[t] if o else pd.NaT for t, o in zip(tgt, ok)]),
        "y_origin": vals[origins],
        "y_true": np.where(ok, vals[np.minimum(tgt, len(raw) - 1)], np.nan),
    })
    return pd.concat([df, pd.DataFrame(q_raw, columns=qcols(levels))], axis=1)


def forecast_path(ticker: str, series: str, horizon: int, model: str, ctx: int | None):
    c = "NA" if ctx is None else str(ctx)
    return FORECASTS / f"{ticker}__{series}__h{horizon}__{model}__c{c}.parquet"


def _log_timing(row: dict) -> None:
    path = METRICS / "timing.csv"
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def run_backtest(tickers: list[str] | None = None, models: list[str] | None = None,
                 experiments: list[tuple[str, int]] | None = None, ctxs: list[int] | None = None,
                 overwrite: bool = False) -> None:
    ensure_dirs()
    dcfg, ecfg = load_config("data"), load_config("experiment")
    tickers = tickers or list(dcfg["tickers"])
    models = models or ecfg["models"]
    experiments = experiments or load_experiments()
    ctx_all = ecfg["context_lengths"]
    ctxs = ctxs or ctx_all
    specs = load_series_specs()

    for model in models:                                     # 모델을 바깥 루프로: 가중치는 한 번만 로드
        fc = get_forecaster(model, ecfg)
        for ticker in tickers:
            df = load_processed(ticker)
            for series, h in experiments:
                spec, raw = specs[series], df[series]
                origins = make_origins(raw, ecfg["test_start"], ecfg.get("test_end"),
                                       ecfg["origin_step"], h, max(ctx_all))
                for ctx in (ctxs if fc.uses_context else [None]):
                    path = forecast_path(ticker, series, h, model, ctx)
                    if path.exists() and not overwrite:
                        log.info("skip (exists): %s", path.name)
                        continue
                    t0 = time.perf_counter()
                    out = backtest_frame(raw, spec, fc, h, ctx, origins, key=f"{ticker}:{series}")
                    total = time.perf_counter() - t0
                    out.insert(0, "model", model)
                    out.insert(1, "ticker", ticker)
                    out.insert(2, "series", series)
                    out.insert(3, "horizon", h)
                    out.insert(4, "ctx", -1 if ctx is None else ctx)
                    out.to_parquet(path)
                    _log_timing({"model": model, "ticker": ticker, "series": series, "horizon": h,
                                 "ctx": -1 if ctx is None else ctx, "n_forecasts": len(out),
                                 "total_s": round(total, 3),
                                 "ms_per_forecast": round(total / max(len(out), 1) * 1000, 3),
                                 "device": ecfg["timestral"]["device"] if model.startswith("timestral") else "cpu",
                                 "timestamp": datetime.now().isoformat(timespec="seconds")})
                    log.info("%s | %s %s h%d ctx=%s | %d origins, %.1fs", model, ticker, series, h, ctx,
                             len(out), total)


def forecast_latest(ticker: str, model: str, ctx: int = 1024,
                    experiments: list[tuple[str, int]] | None = None) -> pd.DataFrame:
    """가장 최근 날짜를 origin 으로 한 실전 밴드 (정답 없음)."""
    ecfg = load_config("experiment")
    fc = get_forecaster(model, ecfg)
    specs = load_series_specs()
    df = load_processed(ticker)
    rows = []
    for series, h in (experiments or load_experiments()):
        raw = df[series]
        origins = make_origins(raw, ecfg["test_start"], None, 1, h, ctx, live=True)
        # arima/climatology 는 테스트 이전 데이터로 차수를 고르므로 live 에서도 전체 이력을 사용
        out = backtest_frame(raw, specs[series], fc, h, ctx if fc.uses_context else None, origins,
                             key=f"{ticker}:{series}")
        out.insert(0, "series", series)
        out.insert(1, "horizon", h)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)
