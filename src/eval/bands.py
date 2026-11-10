"""밴드 해석 규칙 (docs 2절).

m5  : 5일 MDD 가 q0.05 보다 깊으면 '패닉', q0.10~q0.05 사이면 '깊은 조정', q0.50~q0.10 이면 '흔한 조정'
m20 : 20일 정기 매수 시 정상 범위 = q0.50(중앙) ~ q0.10, q0.05 는 최악 시나리오 기준선
yz20: 변동성이 q0.10~q0.90 안이면 '정상', 위면 '고변동', 아래면 '저변동'
MDD 는 음수이므로 "깊다" = 값이 더 작다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import qcol


def classify_mdd(df: pd.DataFrame, value_col: str = "y_true") -> pd.Series:
    v = df[value_col]
    out = np.select(
        [v < df[qcol(0.05)], v < df[qcol(0.10)], v < df[qcol(0.50)]],
        ["panic(<q05)", "deep(q05~q10)", "common(q10~q50)"],
        default="mild(>q50)",
    )
    return pd.Series(out, index=df.index, name="mdd_state")


def classify_return(df: pd.DataFrame, value_col: str = "y_true") -> pd.Series:
    """수익률: q05 아래 = 하방 꼬리, q95 위 = 상방 꼬리, 그 사이는 정상."""
    v = df[value_col]
    out = np.select([v < df[qcol(0.05)], v > df[qcol(0.95)]], ["tail_down(<q05)", "tail_up(>q95)"], default="normal")
    return pd.Series(out, index=df.index, name="return_state")


def classify_vol(df: pd.DataFrame, value_col: str = "y_true") -> pd.Series:
    v = df[value_col]
    out = np.select([v > df[qcol(0.90)], v < df[qcol(0.10)]], ["high(>q90)", "low(<q10)"], default="normal")
    return pd.Series(out, index=df.index, name="vol_state")
