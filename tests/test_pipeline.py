"""합성 데이터로 end-to-end (네트워크/모델 가중치 불필요). Timestral 은 가짜 모델로 인터페이스만 검증."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.common import qcols, quantile_levels
from src.eval.backtest import backtest_frame, make_origins
from src.eval.bands import classify_mdd
from src.eval.summarize import summarize
from src.forecast.arima_runner import ArimaRunner
from src.forecast.climatology_runner import ClimatologyRunner
from src.forecast.timestral_runner import TimestralRunner
from src.targets.spec import load_series_specs
from tests.test_targets import _make_ohlc


class FakeModel:
    """TimestralForecaster 와 같은 출력 형태: quantiles (B,1,H,21), quantile_levels."""
    def forecast(self, context, prediction_length):
        ctx = np.asarray(context)
        B, H, lv = len(ctx), prediction_length, quantile_levels()
        sd = ctx[:, -20:].std(axis=1)[:, None] + 1e-3
        q = ctx[:, -1:][:, :, None] + sd[:, :, None] * norm.ppf(lv)[None, None, :]      # (B,1,21)
        q = np.repeat(q[:, :, None, :], H, axis=2)[:, 0:1]                                # (B,1,H,21)
        return SimpleNamespace(quantiles=q, quantile_levels=tuple(lv))


@pytest.fixture(scope="module")
def processed():
    ohlc = _make_ohlc(n=900, seed=3)
    for n, s in load_series_specs().items():
        ohlc[n] = s.compute(ohlc)
    return ohlc


def _check(df, n_origins):
    assert len(df) == n_origins
    q = df[qcols()].to_numpy()
    assert np.isfinite(q).all()
    assert (np.diff(q, axis=1) >= 0).all()                    # 분위수 비교차


@pytest.mark.parametrize("series,h", [("m5", 5), ("m20", 20), ("yz20", 20), ("r1", 1), ("r5", 5), ("r20", 20)])
def test_runners_shapes_and_order(processed, series, h):
    spec, raw = load_series_specs()[series], processed[series]
    origins = make_origins(raw, "2021-06-01", None, 10, h, max_ctx=256)
    fake = TimestralRunner("5M", {"batch_size": 8}, model=FakeModel())
    _check(backtest_frame(raw, spec, fake, h, 256, origins), len(origins))
    _check(backtest_frame(raw, spec, ClimatologyRunner({}), h, None, origins), len(origins))
    arima = ArimaRunner({"context": 300, "refit_every": 5, "dist": "bootstrap", "order_select_obs": 400,
                         "max_p": 1, "max_q": 1})
    _check(backtest_frame(raw, spec, arima, h, None, origins[:8], key=series), 8)


def test_mdd_quantiles_are_nonpositive_and_vol_positive(processed):
    origins = make_origins(processed["m20"], "2021-06-01", None, 10, 20, 256)
    d = backtest_frame(processed["m20"], load_series_specs()["m20"], ClimatologyRunner({}), 20, None, origins)
    assert (d[qcols()].to_numpy() <= 0).all()
    d = backtest_frame(processed["yz20"], load_series_specs()["yz20"], ClimatologyRunner({}), 20, None, origins)
    assert (d[qcols()].to_numpy() > 0).all()


def test_return_forecast_is_signed_and_aligned(processed):
    """수익률 분위수는 음수/양수 모두 가능하고, 정답은 t 이후 h일 로그수익률(= close[t+h]/close[t])."""
    raw, spec = processed["r5"], load_series_specs()["r5"]
    origins = make_origins(raw, "2021-06-01", None, 10, 5, 256)
    d = backtest_frame(raw, spec, ClimatologyRunner({}), 5, None, origins)
    q = d[qcols()].to_numpy()
    assert (q[:, 0] < 0).all() and (q[:, -1] > 0).all()
    fwd = np.log(processed["close"].to_numpy()[origins + 5] / processed["close"].to_numpy()[origins])
    assert np.allclose(d["y_true"], fwd)


def test_truth_alignment(processed):
    """y_true == series[t+h], y_origin == series[t]."""
    raw = processed["m20"]
    origins = make_origins(raw, "2021-06-01", None, 10, 20, 256)
    d = backtest_frame(raw, load_series_specs()["m20"], ClimatologyRunner({}), 20, None, origins)
    assert np.allclose(d["y_true"], raw.to_numpy()[origins + 20])
    assert np.allclose(d["y_origin"], raw.to_numpy()[origins])
    assert (d["target_date"].to_numpy() == raw.index[origins + 20].to_numpy()).all()


def test_no_lookahead(processed):
    """origin 이후 데이터를 망가뜨려도 그 origin 의 예측은 변하지 않아야 함."""
    spec = load_series_specs()["m20"]
    raw = processed["m20"]
    origins = make_origins(raw, "2021-06-01", None, 10, 20, 256)[:5]
    broken = raw.copy()
    broken.iloc[origins[-1] + 1:] = -0.99
    fake = TimestralRunner("5M", {"batch_size": 4}, model=FakeModel())
    for fc in (fake, ClimatologyRunner({})):
        a = backtest_frame(raw, spec, fc, 20, 256, origins)[qcols()].to_numpy()
        b = backtest_frame(broken, spec, fc, 20, 256, origins)[qcols()].to_numpy()
        assert np.allclose(a, b)


def test_live_origin_has_no_truth(processed):
    raw = processed["yz20"]
    o = make_origins(raw, "2021-06-01", None, 5, 5, 256, live=True)
    d = backtest_frame(raw, load_series_specs()["yz20"], ClimatologyRunner({}), 5, None, o)
    assert len(d) == 1 and np.isnan(d["y_true"].iloc[0]) and pd.isna(d["target_date"].iloc[0])


def test_run_forecasts_summarize_plots_in_memory(processed, monkeypatch):
    """파일 저장 없이: run_forecasts -> summarize -> plots (합성 데이터, 기준선 모델만)."""
    import src.eval.backtest as B
    import src.eval.summarize as S
    from src.viz.plots import calibration_plot, fan_chart, select, size_comparison
    monkeypatch.setattr(B, "load_processed", lambda k: processed)
    monkeypatch.setattr(S, "load_processed", lambda k: processed)

    fc = B.run_forecasts(tickers=["SP500"], models=["climatology"], experiments=[("m20", 20), ("yz20", 20)],
                         ctxs=[256], origin_step=5, test_start="2021-06-01", progress=False)
    assert {"model", "ticker", "series", "horizon", "ctx", "origin_date", "y_true", "q05", "ms_per_forecast"} <= set(fc.columns)
    assert set(fc.series) == {"m20", "yz20"} and (fc.ctx == -1).all()

    groups, tau = summarize(fc, origin_step=5, test_start="2021-06-01")
    assert {"mase_median", "avg_pinball", "coverage80", "n_eff"} <= set(groups.columns)
    assert {"violation_rate", "kupiec_p", "christoffersen_p", "pinball"} <= set(tau.columns)

    one = select(fc, series="m20", horizon=20)
    assert classify_mdd(one).isin(["panic(<q05)", "deep(q05~q10)", "common(q10~q50)", "mild(>q50)"]).all()
    import matplotlib.pyplot as plt
    for fig in (fan_chart(one, "t"), calibration_plot(tau, "SP500", "m20", 20), size_comparison(tau, "m20", 20)):
        assert fig.axes
        plt.close(fig)
