"""Rolling-origin 백테스트 (결과는 파일로 저장하지 않고 DataFrame 으로 반환 -> 노트북에서 바로 평가/시각화).

origin t 에서 모델은 t 까지의 trailing 지표 시계열만 본다 (미래 가격 불사용).
"t 이후 h일의 지표" == "t+h 시점의 trailing 값" 이므로 정답은 series[t+h].

run_forecasts() 가 돌려주는 long-format DataFrame 컬럼 (모두 원 스케일):
    model, ticker, series, horizon, ctx, origin_date, target_date, y_origin, y_true,
    q01 ... q99, ms_per_forecast
(ctx == -1 은 컨텍스트 길이가 무관한 모델: arima, climatology)
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from src.common import load_config, qcols, quantile_levels
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
    """순수 계산: origin 별 원 스케일 분위수 DataFrame (메타 컬럼 없음)."""
    levels = quantile_levels()
    x = spec.to_model(raw.to_numpy())
    x[raw.isna().to_numpy()] = np.nan
    q_model = forecaster.predict(x, origins, horizon, ctx, levels, key=key)
    q_raw = spec.quantiles_to_raw(q_model)
    q_raw[np.isnan(q_model).any(axis=1)] = np.nan             # 실패한 origin 은 NaN 유지

    dates, vals = raw.index, raw.to_numpy()
    tgt = origins + horizon
    ok = tgt < len(raw)
    df = pd.DataFrame({
        "origin_date": dates[origins],
        "target_date": pd.DatetimeIndex([dates[t] if o else pd.NaT for t, o in zip(tgt, ok)]),
        "y_origin": vals[origins],
        "y_true": np.where(ok, vals[np.minimum(tgt, len(raw) - 1)], np.nan),
    })
    return pd.concat([df, pd.DataFrame(q_raw, columns=qcols(levels))], axis=1)


def _tag(df: pd.DataFrame, model, ticker, series, h, ctx, ms) -> pd.DataFrame:
    df.insert(0, "model", model)
    df.insert(1, "ticker", ticker)
    df.insert(2, "series", series)
    df.insert(3, "horizon", h)
    df.insert(4, "ctx", -1 if ctx is None else ctx)
    df["ms_per_forecast"] = ms
    return df


def run_forecasts(tickers: list[str] | None = None, models: list[str] | None = None,
                  experiments: list[tuple[str, int]] | None = None, ctxs: list[int] | None = None,
                  origin_step: int | None = None, test_start: str | None = None,
                  test_end: str | None = None, progress: bool = True) -> pd.DataFrame:
    """모델×지수×(시계열,horizon)×ctx 전체 백테스트를 돌려 long-format DataFrame 으로 반환.

    인자를 생략하면 configs/*.yaml 값을 쓴다. 노트북에서 빠르게 돌려볼 땐 origin_step 을 키우거나
    tickers/models/ctxs 를 줄이면 된다. origin 시작점은 요청한 ctxs 중 가장 긴 값을 기준으로 맞춘다.
    """
    from tqdm.auto import tqdm

    dcfg, ecfg = load_config("data"), load_config("experiment")
    tickers = tickers or list(dcfg["tickers"])
    models = models or ecfg["models"]
    experiments = experiments or load_experiments()
    ctxs = ctxs or ecfg["context_lengths"]
    step = origin_step or ecfg["origin_step"]
    test_start = test_start or ecfg["test_start"]
    test_end = test_end if test_end is not None else ecfg.get("test_end")
    specs = load_series_specs()
    data = {t: load_processed(t) for t in tickers}

    jobs = []                                                # (model, ticker, series, h, ctx)
    forecasters = {m: get_forecaster(m, ecfg) for m in models}
    for m in models:
        for t in tickers:
            for s, h in experiments:
                for c in (ctxs if forecasters[m].uses_context else [None]):
                    jobs.append((m, t, s, h, c))

    out = []
    for m, t, s, h, c in tqdm(jobs, desc="forecast runs", disable=not progress):
        raw = data[t][s]
        origins = make_origins(raw, test_start, test_end, step, h, max(ctxs))
        t0 = time.perf_counter()
        df = backtest_frame(raw, specs[s], forecasters[m], h, c, origins, key=f"{t}:{s}")
        ms = (time.perf_counter() - t0) / max(len(df), 1) * 1000
        out.append(_tag(df, m, t, s, h, c, ms))
        log.info("%s | %s %s h%d ctx=%s | %d origins, %.1f ms/forecast", m, t, s, h, c, len(df), ms)
    return pd.concat(out, ignore_index=True)


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
        out = backtest_frame(raw, specs[series], fc, h, ctx if fc.uses_context else None, origins,
                             key=f"{ticker}:{series}")
        out.insert(0, "series", series)
        out.insert(1, "horizon", h)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)
