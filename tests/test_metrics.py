import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from src.eval.coverage_tests import christoffersen_independence, kupiec_pof
from src.eval.metrics import pinball, violation_rate


def test_pinball_hand_calc():
    # y=1, q=0 -> tau*1 ; y=0, q=1 -> (1-tau)*1
    assert pinball([1.0], [0.0], 0.05) == pytest.approx(0.05)
    assert pinball([0.0], [1.0], 0.05) == pytest.approx(0.95)
    assert violation_rate([0, 2], [1, 1]) == 0.5


def test_kupiec_hand_calc():
    # n=100, x=5, tau=0.05 -> 정확히 일치: LR=0, p=1
    hits = np.r_[np.ones(5), np.zeros(95)]
    r = kupiec_pof(hits, 0.05)
    assert r["lr"] == pytest.approx(0.0, abs=1e-9) and r["p"] == pytest.approx(1.0)
    # n=100, x=10, tau=0.05: LR = -2[ln(.95^90 .05^10) - ln(.9^90 .1^10)]
    hits = np.r_[np.ones(10), np.zeros(90)]
    lr = -2 * ((90 * np.log(.95) + 10 * np.log(.05)) - (90 * np.log(.9) + 10 * np.log(.1)))
    r = kupiec_pof(hits, 0.05)
    assert r["lr"] == pytest.approx(lr) and r["p"] == pytest.approx(chi2.sf(lr, 1))


def test_kupiec_zero_hits_is_finite():
    r = kupiec_pof(np.zeros(200), 0.05)
    assert np.isfinite(r["lr"]) and r["p"] < 0.01


def test_christoffersen_detects_clustering():
    clustered = np.zeros(200, dtype=int); clustered[100:110] = 1       # 10연속 위반
    spread = np.zeros(200, dtype=int); spread[::20] = 1                # 고르게 분포
    assert christoffersen_independence(clustered)["p"] < 0.01
    assert christoffersen_independence(spread)["p"] > 0.05           # 군집 없음 -> 기각되지 않음
