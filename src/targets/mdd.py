"""Trailing Maximum Drawdown.

m_w(t) = 직전 w개 종가 p[t-w+1..t] 안에서의 (고점 대비 최대낙폭). 값은 항상 <= 0 (예: -0.12 = -12%).
처음 w-1개는 NaN. 라벨과 모델 입력이 모두 이 함수로 만들어지므로 정의가 어긋날 수 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def trailing_mdd(close, window: int) -> np.ndarray:
    """close: 1-D 종가 배열. 반환: 같은 길이의 배열 (앞 window-1개 NaN)."""
    p = np.asarray(close, dtype=float)
    out = np.full(p.shape, np.nan)
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(p) < window:
        return out
    win = sliding_window_view(p, window)            # (N-w+1, w)
    runmax = np.maximum.accumulate(win, axis=1)
    dd = win / runmax - 1.0
    out[window - 1:] = dd.min(axis=1)
    return out


def trailing_mdd_series(close: pd.Series, window: int) -> pd.Series:
    return pd.Series(trailing_mdd(close.to_numpy(), window), index=close.index, name=f"m{window}")
