"""노트북용 시각화. 모든 함수는 matplotlib Figure 를 반환하고 파일은 저장하지 않는다.

사용 예 (노트북 셀):
    fig = fan_chart(select(fc, model="timestral-209M", ticker="KOSPI", series="m20", horizon=20, ctx=1024),
                    "Timestral-209M | KOSPI m20 h20")
셀이 끝나면 inline 으로 그림이 뜨고, 그림 우클릭 -> '이미지 복사' 로 가져가면 된다.
(변수에 할당해 두면 이중 출력이 없다.)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.common import load_config, qcol


def select(df: pd.DataFrame, **conds) -> pd.DataFrame:
    """select(fc, model="arima", ticker="KOSPI") — 컬럼==값 (리스트면 isin) 필터."""
    m = pd.Series(True, index=df.index)
    for k, v in conds.items():
        m &= df[k].isin(v) if isinstance(v, (list, tuple, set)) else df[k] == v
    return df[m]


def fan_chart(df: pd.DataFrame, title: str = "", crisis_periods: dict | None = None):
    """단일 (모델, 지수, 시계열, horizon, ctx) 예측: 분위수 밴드 vs 실현값. q05 아래 이탈은 빨간 점."""
    df = df.dropna(subset=["y_true"]).sort_values("target_date")
    x = df["target_date"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.fill_between(x, df[qcol(0.05)], df[qcol(0.95)], alpha=0.18, color="tab:blue", label="q05-q95")
    ax.fill_between(x, df[qcol(0.10)], df[qcol(0.90)], alpha=0.30, color="tab:blue", label="q10-q90")
    ax.plot(x, df[qcol(0.50)], color="tab:blue", lw=1, label="median")
    ax.plot(x, df["y_true"], color="black", lw=0.8, label="realized")
    viol = df[df["y_true"] < df[qcol(0.05)]]
    ax.scatter(viol["target_date"], viol["y_true"], color="red", s=14, zorder=5, label="< q05")
    for a, b in (crisis_periods or load_config("experiment")["crisis_periods"]).values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="gray", alpha=0.12)
    ax.set_title(title)
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def calibration_plot(tau_df: pd.DataFrame, ticker: str, series: str, horizon: int,
                     ctx: int | None = None, regime: str = "all"):
    """명목 tau vs 실제 위반율. 대각선에 가까울수록 캘리브레이션이 좋다. ctx 지정 시 그 ctx(+ctx 무관 모델)만."""
    d = tau_df[(tau_df.ticker == ticker) & (tau_df.series == series) & (tau_df.horizon == horizon)
               & (tau_df.regime == regime)]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for (model, c), g in d.groupby(["model", "ctx"]):
        if ctx is not None and c not in (ctx, -1):
            continue
        g = g.sort_values("tau")
        ax.plot(g["tau"], g["violation_rate"], marker="o", ms=3, label=model if c == -1 else f"{model} c{c}")
    ax.set_xlabel("nominal tau"); ax.set_ylabel("observed violation rate")
    ax.set_title(f"Calibration: {ticker} {series} h{horizon} ({regime})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def size_comparison(tau_df: pd.DataFrame, series: str, horizon: int, tau: float = 0.05,
                    regime: str = "all", ctx: int | None = None):
    """모델(크기)별 위반율(목표선 점선)과 pinball 을 지수별 막대로 비교."""
    d = tau_df[(tau_df.series == series) & (tau_df.horizon == horizon) & (tau_df.tau == tau)
               & (tau_df.regime == regime)].copy()
    if ctx is not None:
        d = d[d.ctx.isin([ctx, -1])]
    d["label"] = d["model"] + d["ctx"].map(lambda c: "" if c == -1 else f"/c{c}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, ttl in zip(axes, ["violation_rate", "pinball"], [f"violation rate (target {tau})", f"pinball tau={tau}"]):
        d.pivot_table(index="label", columns="ticker", values=col).plot.bar(ax=ax)
        if col == "violation_rate":
            ax.axhline(tau, color="k", ls="--", lw=0.8)
        ax.set_title(f"{series} h{horizon} ({regime}): {ttl}"); ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    return fig


def latency_bar(lat: pd.DataFrame):
    """measure_latency() 결과 막대그래프."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(lat["model"], lat["ms_per_forecast"], color="tab:blue")
    for i, v in enumerate(lat["ms_per_forecast"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("ms / forecast"); ax.set_title(f"Inference latency (ctx={lat['ctx'].iloc[0]}, {lat['device'].iloc[0]})")
    fig.tight_layout()
    return fig
