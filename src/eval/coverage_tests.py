"""VaR 백테스트 방식의 위반(hit) 검정.

hit_t = 1{ y_t < q_tau(t) }.  좋은 분위수 예측이면 hit ~ Bernoulli(tau), 서로 독립.
  * Kupiec POF          : 위반 비율이 tau 와 같은가 (무조건 커버리지)
  * Christoffersen 독립 : 위반이 군집(연속 발생)하지 않는가
  * 조건부 커버리지(cc) : 위 둘의 결합, chi2(2)
origin 윈도우가 겹치면 hit 이 자기상관되므로 독립 검정은 겹치지 않는 부분표본(간격 >= horizon)에서 수행하길 권장.
"""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def _ll(p: float, n_ok: float, n_hit: float) -> float:
    """Bernoulli 로그우도 (xlogy 로 0*log0 = 0 처리)."""
    return xlogy(n_ok, 1.0 - p) + xlogy(n_hit, p)


def kupiec_pof(hits, tau: float) -> dict:
    h = np.asarray(hits, dtype=int)
    n, x = len(h), int(h.sum())
    if n == 0:
        return {"n": 0, "hits": 0, "rate": np.nan, "lr": np.nan, "p": np.nan}
    pi = x / n
    lr = -2.0 * (_ll(tau, n - x, x) - _ll(pi, n - x, x))
    return {"n": n, "hits": x, "rate": pi, "lr": float(lr), "p": float(chi2.sf(max(lr, 0.0), 1))}


def christoffersen_independence(hits) -> dict:
    h = np.asarray(hits, dtype=int)
    if len(h) < 2:
        return {"lr": np.nan, "p": np.nan}
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    tot = n00 + n01 + n10 + n11
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / tot
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = (xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11))
    lr = -2.0 * (ll_null - ll_alt)
    return {"lr": float(lr), "p": float(chi2.sf(max(lr, 0.0), 1))}


def conditional_coverage(hits, tau: float) -> dict:
    k = kupiec_pof(hits, tau)
    c = christoffersen_independence(hits)
    if np.isnan(k["lr"]) or np.isnan(c["lr"]):
        return {"lr": np.nan, "p": np.nan}
    lr = k["lr"] + c["lr"]
    return {"lr": float(lr), "p": float(chi2.sf(lr, 2))}
