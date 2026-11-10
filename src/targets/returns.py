"""Trailing 로그수익률.

r_w(t) = ln(C_t / C_{t-w}) : t 까지 직전 w 거래일의 누적 로그수익률. 앞 w개는 NaN.
"t 이후 h일의 수익률" 은 곧 r_h(t+h) 이므로(겹치지 않는 구간) 모델의 h 번째 스텝 분위수가 원하는 수익률 분포다.
MDD 와 달리 부호가 있는 값이라 변환 없이(identity) 그대로 모델에 넣고, 분위수 방향도 뒤집지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trailing_log_return(close: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    c = close.astype(float)
    out = np.log(c / c.shift(window))
    out.name = f"r{window}"
    return out
