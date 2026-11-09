"""원본 OHLC 정제 -> data/interim/{KEY}.parquet, 정제 리포트 JSON.

* NaN / 비양수 가격 행 삭제
* High/Low 가 O,C 를 감싸도록 보정 (H=max(O,H,L,C), L=min(O,H,L,C))
* 평탄 봉(H==L): 구 코스피 등에서 O=H=L=C 로 기록된 날. 기본은 표시만 하고 유지 (YZ 가 낮게 나올 수 있음)
* 과도한 일간 수익률은 삭제하지 않고 리포트에만 기록
* 달력: 각 지수의 자체 거래일 사용 (ffill 없음)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.common import DATA_INTERIM, DATA_RAW, ensure_dirs, load_config

log = logging.getLogger(__name__)


def clean_ohlc(df: pd.DataFrame, max_abs_return: float = 0.25,
               drop_flat_bars: bool = False) -> tuple[pd.DataFrame, dict]:
    report: dict = {"rows_raw": int(len(df))}
    d = df[["open", "high", "low", "close"]].astype(float).copy()

    bad = d.isna().any(axis=1) | (d <= 0).any(axis=1)
    report["dropped_nan_or_nonpositive"] = int(bad.sum())
    d = d[~bad]

    hi = d[["open", "high", "low", "close"]].max(axis=1)
    lo = d[["open", "high", "low", "close"]].min(axis=1)
    report["envelope_fixed_rows"] = int(((d["high"] < hi) | (d["low"] > lo)).sum())
    d["high"], d["low"] = hi, lo

    flat = d["high"] == d["low"]
    report["flat_bars"] = int(flat.sum())
    report["flat_bar_dates_head"] = [str(x.date()) for x in d.index[flat][:10]]
    if drop_flat_bars:
        d = d[~flat]
        report["flat_bars_dropped"] = True

    ret = np.log(d["close"]).diff()
    big = ret.abs() > max_abs_return
    report["large_return_days"] = [{"date": str(i.date()), "ret": float(ret[i])} for i in d.index[big]]

    d = d.sort_index()
    report.update(rows_clean=int(len(d)), start=str(d.index[0].date()), end=str(d.index[-1].date()))
    if report["flat_bars"]:
        log.warning("flat bar %d일 발견 -> YZ 변동성이 과소추정될 수 있음", report["flat_bars"])
    return d, report


def clean_all(keys: list[str] | None = None) -> None:
    ensure_dirs()
    cfg = load_config("data")
    reports = {}
    for key in cfg["tickers"]:
        if keys and key not in keys:
            continue
        raw = pd.read_parquet(DATA_RAW / f"{key}.parquet")
        d, rep = clean_ohlc(raw, cfg["clean"]["max_abs_return"], cfg["clean"]["drop_flat_bars"])
        d.to_parquet(DATA_INTERIM / f"{key}.parquet")
        reports[key] = rep
        log.info("%s: %d -> %d rows (flat bars %d)", key, rep["rows_raw"], rep["rows_clean"], rep["flat_bars"])
    with open(DATA_INTERIM / "clean_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
