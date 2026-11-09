"""Yang-Zhang 변동성 (오버나이트 갭 + 시가-종가 + Rogers-Satchell 의 최소분산 결합).

일별 입력: O, H, L, C 와 전일 종가 C_{t-1}.
    o = ln(O_t / C_{t-1}),  c = ln(C_t / O_t),  h = ln(H_t / O_t),  l = ln(L_t / O_t)
    RS = h(h-c) + l(l-c)
n일 윈도우:
    s_o^2 = var(o, ddof=1),  s_c^2 = var(c, ddof=1),  s_rs^2 = mean(RS)
    k = 0.34 / (1.34 + (n+1)/(n-1))
    sigma^2 = s_o^2 + k s_c^2 + (1-k) s_rs^2     (일별 분산)
    YZ = sqrt(sigma^2 * annualize)                 (연율화, 기본 252)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def yang_zhang(ohlc: pd.DataFrame, window: int = 20, annualize: int = 252) -> pd.Series:
    """ohlc: 컬럼 open/high/low/close (소문자). 반환: 연율화 YZ 변동성 (앞 window개 NaN)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    o, h, l, c = (ohlc[k].astype(float) for k in ("open", "high", "low", "close"))
    prev_c = c.shift(1)

    log_o = np.log(o / prev_c)      # overnight
    log_c = np.log(c / o)           # open -> close
    log_h = np.log(h / o)
    log_l = np.log(l / o)
    rs = log_h * (log_h - log_c) + log_l * (log_l - log_c)

    s_o2 = log_o.rolling(window).var(ddof=1)
    s_c2 = log_c.rolling(window).var(ddof=1)
    s_rs2 = rs.rolling(window).mean()
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var = s_o2 + k * s_c2 + (1.0 - k) * s_rs2
    # 수치 오차로 미세한 음수가 나올 수 있어 0 으로 자름
    vol = np.sqrt(np.clip(var, 0.0, None) * annualize)
    vol.name = f"yz{window}"
    return vol
