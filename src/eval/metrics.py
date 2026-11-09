"""분위수 예측 평가 지표 (모두 원 스케일에서 계산)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import qcol, quantile_levels


def pinball(y, q, tau: float) -> float:
    d = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def violation_rate(y, q) -> float:
    """실현값이 분위수 아래로 간 비율 (목표: tau)."""
    return float(np.mean(np.asarray(y) < np.asarray(q)))


def avg_pinball(df: pd.DataFrame, levels=None) -> float:
    """전체 분위수 격자 평균 pinball (CRPS 근사; 2배 하면 CRPS 에 가까움)."""
    levels = quantile_levels() if levels is None else levels
    return float(np.mean([pinball(df["y_true"], df[qcol(t)], t) for t in levels]))


def coverage(df: pd.DataFrame, lo: float = 0.10, hi: float = 0.90) -> float:
    """[q_lo, q_hi] 구간이 실현값을 포함한 비율 (명목: hi-lo)."""
    y = df["y_true"]
    return float(((y >= df[qcol(lo)]) & (y <= df[qcol(hi)])).mean())


def mase_median(df: pd.DataFrame, scale: float) -> float:
    """중앙값 예측의 MASE. scale = 학습 구간에서 h 스텝 naive 예측의 평균 절대오차."""
    return float((df["y_true"] - df[qcol(0.5)]).abs().mean() / scale)


def naive_scale(series: pd.Series, horizon: int, end: str | pd.Timestamp) -> float:
    """end 이전 구간의 mean |x[t+h] - x[t]|."""
    s = series.loc[:pd.Timestamp(end)].dropna()
    return float((s.shift(-horizon) - s).abs().mean())
