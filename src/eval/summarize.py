"""results/forecasts/*.parquet -> results/metrics/{summary_groups,summary_tau}.csv

regime: origin 날짜가 crisis_periods 에 들어가면 'crisis', 아니면 'calm', 전체는 'all'.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from src.common import FORECASTS, METRICS, ensure_dirs, load_config, qcol
from src.data.process import load_processed
from src.eval.coverage_tests import christoffersen_independence, conditional_coverage, kupiec_pof
from src.eval.metrics import avg_pinball, coverage, mase_median, naive_scale, pinball

log = logging.getLogger(__name__)
GROUP = ["model", "ticker", "series", "horizon", "ctx"]


def load_all_forecasts(pattern: str = "*.parquet") -> pd.DataFrame:
    files = sorted(FORECASTS.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{FORECASTS} 에 예측 파일이 없습니다. 먼저 backtest 를 실행하세요")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def crisis_mask(dates: pd.Series, periods: dict) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for a, b in periods.values():
        m |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return m


def summarize(all_fc: pd.DataFrame | None = None, save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs()
    ecfg = load_config("experiment")
    taus, step = ecfg["eval_taus"], ecfg["origin_step"]
    fc = load_all_forecasts() if all_fc is None else all_fc
    fc = fc.dropna(subset=["y_true"] + [qcol(t) for t in taus]).copy()
    fc["regime"] = np.where(crisis_mask(fc["origin_date"], ecfg["crisis_periods"]), "crisis", "calm")

    timing = None
    tpath = METRICS / "timing.csv"
    if tpath.exists():
        timing = (pd.read_csv(tpath).groupby(GROUP)["ms_per_forecast"].last().reset_index())

    scales: dict = {}
    grp_rows, tau_rows = [], []
    for keys, g_all in fc.groupby(GROUP):
        model, ticker, series, h, ctx = keys
        if (ticker, series, h) not in scales:
            scales[(ticker, series, h)] = naive_scale(load_processed(ticker)[series], h, ecfg["test_start"])
        scale = scales[(ticker, series, h)]
        sub = max(1, math.ceil(h / step))               # 겹치지 않는 부분표본 간격
        for regime in ("all", "calm", "crisis"):
            g = g_all if regime == "all" else g_all[g_all["regime"] == regime]
            if len(g) < 5:
                continue
            g = g.sort_values("origin_date")
            base = dict(zip(GROUP, keys), regime=regime)
            grp_rows.append({**base, "n": len(g), "n_eff": len(g) // sub, "mase_median": mase_median(g, scale),
                             "avg_pinball": avg_pinball(g), "coverage80": coverage(g, 0.10, 0.90)})
            for tau in taus:
                hits = (g["y_true"] < g[qcol(tau)]).to_numpy()
                kp = kupiec_pof(hits, tau)
                ind = christoffersen_independence(hits[::sub])        # 비중첩 부분표본
                cc = conditional_coverage(hits[::sub], tau)
                tau_rows.append({**base, "tau": tau, "pinball": pinball(g["y_true"], g[qcol(tau)], tau),
                                 "violation_rate": kp["rate"], "hits": kp["hits"],
                                 "kupiec_p": kp["p"], "christoffersen_p": ind["p"], "cc_p": cc["p"]})
    groups, tau_df = pd.DataFrame(grp_rows), pd.DataFrame(tau_rows)
    if timing is not None and len(groups):
        groups = groups.merge(timing, on=GROUP, how="left")
    if save:
        groups.to_csv(METRICS / "summary_groups.csv", index=False)
        tau_df.to_csv(METRICS / "summary_tau.csv", index=False)
        log.info("saved summary_groups.csv (%d rows), summary_tau.csv (%d rows)", len(groups), len(tau_df))
    return groups, tau_df
