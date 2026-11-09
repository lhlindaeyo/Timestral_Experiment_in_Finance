"""run_forecasts() 결과 DataFrame -> 요약 지표 DataFrame 2개 (파일 저장 없음).

groups : (model, ticker, series, horizon, ctx, regime) 별 MASE / 평균 pinball / coverage80 / n / n_eff / ms_per_forecast
tau    : 위 그룹 × tau 별 pinball / 위반율 / Kupiec p / Christoffersen p / 조건부 커버리지 p
regime : origin 날짜가 crisis_periods 에 들어가면 'crisis', 아니면 'calm', 전체는 'all'.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.common import load_config, qcol
from src.data.process import load_processed
from src.eval.coverage_tests import christoffersen_independence, conditional_coverage, kupiec_pof
from src.eval.metrics import avg_pinball, coverage, mase_median, naive_scale, pinball

GROUP = ["model", "ticker", "series", "horizon", "ctx"]


def crisis_mask(dates: pd.Series, periods: dict) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for a, b in periods.values():
        m |= (dates >= pd.Timestamp(a)) & (dates <= pd.Timestamp(b))
    return m


def summarize(fc: pd.DataFrame, origin_step: int | None = None,
              test_start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """origin_step/test_start 는 run_forecasts 에 넘긴 값과 같게 (겹치지 않는 부분표본 간격, MASE 스케일 계산용)."""
    ecfg = load_config("experiment")
    taus = ecfg["eval_taus"]
    step = origin_step or ecfg["origin_step"]
    test_start = test_start or ecfg["test_start"]
    fc = fc.dropna(subset=["y_true"] + [qcol(t) for t in taus]).copy()
    fc["regime"] = np.where(crisis_mask(fc["origin_date"], ecfg["crisis_periods"]), "crisis", "calm")

    scales: dict = {}
    grp_rows, tau_rows = [], []
    for keys, g_all in fc.groupby(GROUP):
        _, ticker, series, h, _ = keys
        if (ticker, series, h) not in scales:
            scales[(ticker, series, h)] = naive_scale(load_processed(ticker)[series], h, test_start)
        scale = scales[(ticker, series, h)]
        sub = max(1, math.ceil(h / step))               # 겹치지 않는 부분표본 간격
        for regime in ("all", "calm", "crisis"):
            g = g_all if regime == "all" else g_all[g_all["regime"] == regime]
            if len(g) < 5:
                continue
            g = g.sort_values("origin_date")
            base = dict(zip(GROUP, keys), regime=regime)
            grp_rows.append({**base, "n": len(g), "n_eff": len(g) // sub, "mase_median": mase_median(g, scale),
                             "avg_pinball": avg_pinball(g), "coverage80": coverage(g, 0.10, 0.90),
                             "ms_per_forecast": g["ms_per_forecast"].iloc[0]})
            for tau in taus:
                hits = (g["y_true"] < g[qcol(tau)]).to_numpy()
                kp = kupiec_pof(hits, tau)
                ind = christoffersen_independence(hits[::sub])        # 비중첩 부분표본
                cc = conditional_coverage(hits[::sub], tau)
                tau_rows.append({**base, "tau": tau, "pinball": pinball(g["y_true"], g[qcol(tau)], tau),
                                 "violation_rate": kp["rate"], "hits": kp["hits"],
                                 "kupiec_p": kp["p"], "christoffersen_p": ind["p"], "cc_p": cc["p"]})
    return pd.DataFrame(grp_rows), pd.DataFrame(tau_rows)
