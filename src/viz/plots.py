"""시각화: fan chart, 캘리브레이션, 모델 크기별 비교."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import FIGURES, ensure_dirs, load_config, qcol  # noqa: E402
from src.eval.summarize import load_all_forecasts  # noqa: E402


def fan_chart(df: pd.DataFrame, title: str, path=None, crisis_periods: dict | None = None):
    """단일 (모델, 지수, 시계열, horizon, ctx) 예측: 분위수 밴드 vs 실현값. q05 아래 이탈은 점으로 표시."""
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
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def calibration_plot(tau_df: pd.DataFrame, ticker: str, series: str, horizon: int, path=None, ctx: int | None = None):
    """명목 tau vs 실제 위반율. 대각선에 가까울수록 캘리브레이션이 좋다. ctx=None 이면 모델별 최고 ctx 사용."""
    d = tau_df[(tau_df.ticker == ticker) & (tau_df.series == series) & (tau_df.horizon == horizon)
               & (tau_df.regime == "all")]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for (model, c), g in d.groupby(["model", "ctx"]):
        if ctx is not None and c not in (ctx, -1):
            continue
        g = g.sort_values("tau")
        ax.plot(g["tau"], g["violation_rate"], marker="o", ms=3, label=f"{model} c{c}")
    ax.set_xlabel("nominal tau"); ax.set_ylabel("observed violation rate")
    ax.set_title(f"Calibration: {ticker} {series} h{horizon}")
    ax.legend(fontsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def size_comparison(tau_df: pd.DataFrame, series: str, horizon: int, tau: float = 0.05, path=None,
                    regime: str = "all"):
    """모델(크기)별 위반율과 pinball 비교 막대그래프 (지수 평균 아님: 지수별 막대)."""
    d = tau_df[(tau_df.series == series) & (tau_df.horizon == horizon) & (tau_df.tau == tau)
               & (tau_df.regime == regime)].copy()
    d["label"] = d["model"] + d["ctx"].map(lambda c: "" if c == -1 else f"/c{c}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)
    for ax, col, ttl in zip(axes, ["violation_rate", "pinball"], [f"violation rate (target {tau})", f"pinball tau={tau}"]):
        pv = d.pivot_table(index="label", columns="ticker", values=col)
        pv.plot.bar(ax=ax)
        if col == "violation_rate":
            ax.axhline(tau, color="k", ls="--", lw=0.8)
        ax.set_title(f"{series} h{horizon}: {ttl}"); ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig


def plot_all(model_for_fans: str = "timestral-209M", ctx: int = 1024) -> None:
    """저장된 예측/지표에서 기본 그림 세트를 results/figures 에 저장."""
    ensure_dirs()
    from src.common import METRICS
    tau_df = pd.read_csv(METRICS / "summary_tau.csv")
    fc = load_all_forecasts()
    for (ticker, series, h), _ in fc.groupby(["ticker", "series", "horizon"]):
        calibration_plot(tau_df, ticker, series, h, FIGURES / f"calib__{ticker}__{series}_h{h}.png", ctx=ctx)
        g = fc[(fc.ticker == ticker) & (fc.series == series) & (fc.horizon == h) & (fc.model == model_for_fans)
               & (fc.ctx == ctx)]
        if len(g):
            fan_chart(g, f"{model_for_fans} c{ctx} | {ticker} {series} h{h}",
                      FIGURES / f"fan__{ticker}__{series}_h{h}__{model_for_fans}.png")
    for series, h in fc[["series", "horizon"]].drop_duplicates().itertuples(index=False):
        size_comparison(tau_df, series, h, 0.05, FIGURES / f"sizes__{series}_h{h}__tau05.png")
