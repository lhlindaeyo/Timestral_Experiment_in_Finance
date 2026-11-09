"""파이프라인 진입점.

  python run.py fetch                       # 1. 데이터 다운로드 (Yahoo Finance)
  python run.py build                       # 2. 정제 + m5/m20/yz20 계산
  python run.py download-models [--sizes 5M 209M]
  python run.py latency [--sizes 5M 209M]   # 로컬 추론 지연시간 측정
  python run.py backtest [--tickers KOSPI] [--models timestral-5M arima] [--ctx 512 1024] [--overwrite]
  python run.py evaluate                    # 지표/검정 요약 CSV
  python run.py plot                        # 그림 저장
  python run.py latest --ticker KOSPI --model timestral-209M --ctx 1024   # 오늘 기준 밴드
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from src.common import LOGS, ROOT, ensure_dirs, load_config, qcol


def cmd_download_models(sizes):
    from huggingface_hub import snapshot_download
    prefix = load_config("experiment")["timestral"]["repo_prefix"]
    for s in sizes:
        target = ROOT / "models" / f"Timestral-{s}"
        print(f"downloading {prefix}Timestral-{s} -> {target}")
        snapshot_download(repo_id=f"{prefix}Timestral-{s}", local_dir=target)


def cmd_latency(sizes, ctx, horizon, n, batch):
    from src.forecast.timestral_runner import TimestralRunner
    cfg = dict(load_config("experiment")["timestral"])
    rng = np.random.default_rng(0)
    x = np.log1p(np.abs(rng.standard_normal(ctx + n + 10)).cumsum() * 0.01)
    origins = np.arange(ctx - 1, ctx - 1 + n)
    levels = np.asarray(load_config("targets")["quantile_levels"])
    print(f"device={cfg['device']} ctx={ctx} horizon={horizon} n={n} batch={batch}")
    for s in sizes:
        cfg["batch_size"] = batch
        r = TimestralRunner(s, cfg)
        t0 = time.perf_counter(); r._load(); load_s = time.perf_counter() - t0
        r.predict(x, origins[:batch], horizon, ctx, levels)               # warm-up
        r.predict(x, origins, horizon, ctx, levels)
        print(f"Timestral-{s:>4}: load {load_s:5.1f}s | {r.last_ms_per_forecast:8.1f} ms/forecast")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fetch").add_argument("--tickers", nargs="*")
    sp.add_parser("build").add_argument("--tickers", nargs="*")
    p = sp.add_parser("download-models"); p.add_argument("--sizes", nargs="*", default=["5M", "28M", "123M", "209M"])
    p = sp.add_parser("latency")
    p.add_argument("--sizes", nargs="*", default=["5M", "28M", "123M", "209M"])
    p.add_argument("--ctx", type=int, default=1024); p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--n", type=int, default=64); p.add_argument("--batch", type=int, default=32)
    p = sp.add_parser("backtest")
    p.add_argument("--tickers", nargs="*"); p.add_argument("--models", nargs="*")
    p.add_argument("--ctx", nargs="*", type=int); p.add_argument("--overwrite", action="store_true")
    p.add_argument("--series", nargs="*", help="예: m20 yz20 (지정하면 해당 시계열 실험만)")
    sp.add_parser("evaluate")
    sp.add_parser("plot")
    p = sp.add_parser("latest")
    p.add_argument("--ticker", required=True); p.add_argument("--model", default="timestral-209M")
    p.add_argument("--ctx", type=int, default=1024)
    a = ap.parse_args()

    ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(LOGS / "run.log", encoding="utf-8")])

    if a.cmd == "fetch":
        from src.data.fetch import fetch_all; fetch_all(a.tickers)
    elif a.cmd == "build":
        from src.data.clean import clean_all
        from src.data.process import build_processed
        clean_all(a.tickers); build_processed(a.tickers)
    elif a.cmd == "download-models":
        cmd_download_models(a.sizes)
    elif a.cmd == "latency":
        cmd_latency(a.sizes, a.ctx, a.horizon, a.n, a.batch)
    elif a.cmd == "backtest":
        from src.eval.backtest import run_backtest
        from src.targets.spec import load_experiments
        exps = [e for e in load_experiments() if not a.series or e[0] in a.series]
        run_backtest(a.tickers, a.models, exps, a.ctx, a.overwrite)
    elif a.cmd == "evaluate":
        from src.eval.summarize import summarize
        groups, _ = summarize()
        print(groups.sort_values(["series", "horizon", "ticker", "avg_pinball"]).to_string(index=False))
    elif a.cmd == "plot":
        from src.viz.plots import plot_all; plot_all()
    elif a.cmd == "latest":
        from src.eval.backtest import forecast_latest
        out = forecast_latest(a.ticker, a.model, a.ctx)
        cols = ["series", "horizon", "origin_date", "y_origin"] + [qcol(t) for t in (0.05, 0.10, 0.50, 0.90, 0.95)]
        print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n※ m5/m20 은 MDD(음수), yz20 은 연율화 변동성. q05 = 하위 5% (MDD 에서는 가장 깊은 낙폭).")


if __name__ == "__main__":
    main()
