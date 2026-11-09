"""MDD / Yang-Zhang 를 손계산 · 독립 구현과 대조."""
import math

import numpy as np
import pandas as pd
import pytest

from src.targets.mdd import trailing_mdd
from src.targets.spec import SeriesSpec
from src.targets.yang_zhang import yang_zhang
from src.common import quantile_levels


def test_mdd_hand_calc():
    p = [100, 110, 99, 105, 90]
    # 윈도우 5: 고점 110, 저점 90 -> 90/110-1
    m5 = trailing_mdd(p, 5)
    assert np.isnan(m5[:4]).all()
    assert m5[4] == pytest.approx(90 / 110 - 1)
    # 윈도우 3
    m3 = trailing_mdd(p, 3)
    assert np.isnan(m3[:2]).all()
    assert m3[2] == pytest.approx(99 / 110 - 1)          # [100,110,99]
    assert m3[3] == pytest.approx(99 / 110 - 1)          # [110,99,105]
    assert m3[4] == pytest.approx(90 / 105 - 1)          # [99,105,90]


def test_mdd_monotone_up_is_zero_and_nonpositive():
    assert (trailing_mdd(np.arange(1, 30, dtype=float), 20)[19:] == 0).all()
    rng = np.random.default_rng(0)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    m = trailing_mdd(p, 20)
    assert (m[19:] <= 0).all() and (m[19:] > -1).all()


def test_mdd_peak_before_window_is_ignored():
    # 고점 200 은 5일 윈도우 밖 -> 낙폭에 포함되면 안 됨
    p = [200, 100, 100, 100, 100, 100, 100, 90]
    assert trailing_mdd(p, 5)[7] == pytest.approx(90 / 100 - 1)


def _yz_reference(o, h, l, c, i, n, ann=252):
    """순수 파이썬 루프 구현 (i 를 끝으로 하는 n일 윈도우)."""
    idx = range(i - n + 1, i + 1)
    ov = [math.log(o[t] / c[t - 1]) for t in idx]
    cv = [math.log(c[t] / o[t]) for t in idx]
    rs = [math.log(h[t] / o[t]) * (math.log(h[t] / o[t]) - math.log(c[t] / o[t])) +
          math.log(l[t] / o[t]) * (math.log(l[t] / o[t]) - math.log(c[t] / o[t])) for t in idx]

    def var(v):
        m = sum(v) / len(v)
        return sum((a - m) ** 2 for a in v) / (len(v) - 1)

    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return math.sqrt((var(ov) + k * var(cv) + (1 - k) * sum(rs) / n) * ann)


def _make_ohlc(n=120, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.004, n))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_yz_matches_reference_loop():
    df = _make_ohlc()
    yz = yang_zhang(df, 20)
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    for i in (20, 57, 119):
        assert yz.iloc[i] == pytest.approx(_yz_reference(o, h, l, c, i, 20))
    assert yz.iloc[:20].isna().all()


def test_yz_zero_when_no_intraday_or_gap_variation():
    # 매일 같은 로그수익률로 시가=전일종가, 종가=고가, 저가=시가 -> 분산 0
    c = 100 * 1.01 ** np.arange(40)
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": c, "low": o, "close": c}, index=pd.bdate_range("2020-01-01", periods=40))
    assert yang_zhang(df, 20).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_spec_roundtrip_and_tail_flip():
    q = quantile_levels()
    s = SeriesSpec("m20", "mdd", 20, "log1p_mag")
    # 낙폭 크기 분포: 모델 분위수 = log1p(m_q). m 이 클수록(깊을수록) 상위 분위수
    mag = np.linspace(0.01, 0.30, len(q))[None, :]
    raw = s.quantiles_to_raw(np.log1p(mag))
    assert raw.shape == (1, len(q)) and (raw <= 0).all()
    assert (np.diff(raw, axis=1) >= 0).all()                         # 오름차순
    assert raw[0, 0] == pytest.approx(-0.30)                         # q01 = 가장 깊은 낙폭 (= 모델 q99)
    assert raw[0, -1] == pytest.approx(-0.01)
    x = s.to_model(np.array([-0.12]))
    assert x[0] == pytest.approx(math.log1p(0.12))

    v = SeriesSpec("yz20", "yang_zhang", 20, "log")
    assert v.quantiles_to_raw(np.log(np.array([[0.1] * len(q)])))[0, 0] == pytest.approx(0.1)
