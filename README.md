# Timestral 기반 MDD · 변동성 밴드 예측

S&P500 · 나스닥 · 코스피 일봉으로 **5일/20일 MDD 분포**와 **20일 Yang-Zhang 변동성 분포**를 Timestral(5M/28M/123M/209M)로 제로샷 예측하고, ARIMA·무조건 분위수 기준선과 캘리브레이션을 비교하는 실험.

핵심 아이디어: 모델은 시계열의 분위수만 내놓으므로, **목표 지표를 시계열(trailing m5, m20, yz20)로 만든 뒤 그 시계열을 예측**한다.
"t 이후 h일의 지표" = "t+h 시점의 trailing 값" 이라 모델의 h 번째 스텝 분위수가 곧 원하는 분포이고, 라벨과 입력은 `src/targets/` 의 같은 함수로 만든다.

## 작업 방식: 결과 파일 없이 노트북에서 실행 → 시각화

예측·지표·그림은 **파일로 저장하지 않는다.** 노트북에서 함수를 불러 모델을 돌리고(`run_forecasts`), 평가(`summarize`)하고, 그 자리에서 그림(`fan_chart`, `calibration_plot`, `size_comparison`, …)을 그린다. 그림은 셀 아래에 inline 으로 뜨므로 우클릭 → 이미지 복사로 가져가면 된다. 결과는 커널 변수(`fc`, `groups`, `tau`)에만 있다.

```python
from src.eval.backtest import run_forecasts
from src.eval.summarize import summarize
from src.viz.plots import select, fan_chart, calibration_plot, size_comparison

fc = run_forecasts(tickers=["KOSPI"], models=["timestral-209M", "arima", "climatology"], ctxs=[1024], origin_step=20)
groups, tau = summarize(fc, origin_step=20)
fig = fan_chart(select(fc, model="timestral-209M", series="m20", horizon=20), "209M | KOSPI m20 h20")
```

## 빠른 시작

```bash
python -m venv .venv && .venv\Scripts\activate        # (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt                       # torch>=2.4, timestral, jupyter 포함

python run.py fetch                                   # 1. 데이터 다운로드 (Yahoo Finance) -> data/raw
python run.py build                                   # 2. 정제 + m5/m20/yz20 -> data/interim, data/processed
pytest                                                #    타깃 정의 / 지표 / 파이프라인 테스트
python run.py download-models --sizes 5M 209M         # 3. 가중치 -> models/ (선택: 없으면 HF 에서 자동 다운로드)
jupyter lab notebooks/                                # 4. 노트북 실행
```

| 노트북 | 내용 |
|---|---|
| `01_eda` | 데이터 점검(평탄 봉·이상치), 목표 시계열 모양. `REFRESH=True` 면 다운로드·계산까지 노트북에서 수행 |
| `02_target_check` | MDD/YZ 손계산 대조, 라벨 정렬(`origin t → t+h`) 그림 |
| `03_zero_shot_demo` | 한 지수에서 Timestral vs ARIMA vs 기준선: fan chart, 캘리브레이션, **오늘 기준 밴드** |
| `04_backtest_report` | 3지수 × 모델 크기 전체 백테스트 → 지표표 → 크기별 비교/캘리브레이션/위기·평시/fan chart/추론 시간 |

첫 셀 아래 파라미터 셀(`TICKERS`, `MODELS`, `CTXS`, `ORIGIN_STEP`)만 바꾸면 규모를 조절할 수 있다. 처음엔 `TICKERS=["KOSPI"]`, `ORIGIN_STEP=20` 으로 확인하고 늘리자. 커널을 닫으면 결과가 사라지므로 오래 걸린 실행 뒤에는 필요한 그림부터 복사해 두자.

## 폴더 구성

```
├─ run.py                    # 데이터 준비 / 가중치 다운로드 CLI (fetch, build, download-models)
├─ configs/
│   ├─ data.yaml             # 티커(^GSPC, ^IXIC, ^KS11), 기간, 휴장일/정제 규칙
│   ├─ targets.yaml          # MDD 윈도우(5,20), YZ 윈도우(20), 변환, 분위수 레벨, 실험(시계열×horizon)
│   └─ experiment.yaml       # 모델 크기, context 길이, origin 간격, 테스트 기간, ARIMA 설정, 위기 구간
├─ models/                   # Timestral-5M/ 28M/ 123M/ 209M/ 가중치 (git 제외)
├─ data/{raw,interim,processed}/
├─ src/
│   ├─ common.py             # 경로, 설정 로딩, 분위수 컬럼명
│   ├─ data/                 # fetch.py, clean.py, process.py
│   ├─ targets/              # mdd.py, yang_zhang.py, spec.py   ← 라벨·입력 생성 단일 지점
│   ├─ forecast/             # timestral_runner.py, arima_runner.py, climatology_runner.py, latency.py, base.py
│   ├─ eval/                 # backtest.py(run_forecasts), metrics.py, coverage_tests.py, summarize.py, bands.py
│   └─ viz/                  # plots.py (fan chart, 캘리브레이션, 크기별 비교, 지연시간) — Figure 반환
├─ notebooks/                # 01_eda, 02_target_check, 03_zero_shot_demo, 04_backtest_report
├─ tests/                    # 손계산 대조 단위 테스트, 누수·정렬·인메모리 파이프라인 테스트
└─ logs/
```

## 설계 요약

| 항목 | 내용 |
|---|---|
| 지표 | `m5`/`m20`: 직전 5/20 종가 윈도우의 고점 대비 최대낙폭(음수). `yz20`: 20일 Yang-Zhang × √252 |
| 실험 | (m5, h=5), (m20, h=20), (yz20, h=20), (yz20, h=5·보조) |
| 모델 입력 변환 | MDD: 낙폭 크기 m=-mdd 에 `log1p`, 변동성: `log`. 평가는 원 스케일 |
| 꼬리 방향 | 크기 m 로 변환했으므로 모델 레벨 τ' ↔ MDD 레벨 1-τ'. (MDD 하위 5% = 크기의 상위 95%) `SeriesSpec.quantiles_to_raw` |
| 분위수 | Timestral 21개 (0.01, 0.05, 0.10, … 0.95, 0.99). 레벨이 다르면 보간 |
| Origin | `origin_step` 간격, 한 번의 실행 안에서 모든 모델·ctx 가 같은 origin 사용 (가장 긴 ctx 가 차는 시점 이후) |
| 누수 방지 | origin t 의 입력은 t 까지의 trailing 값만. `tests/test_pipeline.py::test_no_lookahead` |
| ARIMA | 지수·지표별 차수 1회 선택(ADF→d, AIC→p,q, 테스트 이전 데이터), origin 20개마다 재적합, 표준화 잔차 부트스트랩 × 예측표준오차 |
| 기준선 | 무조건 경험적 분위수(expanding) |
| 평가 | pinball(0.05/0.10/…), 위반율, Kupiec(전체) · Christoffersen(겹치지 않는 부분표본), coverage80, MASE(중앙값), 추론 시간. 위기/평시 분리 (origin 날짜 기준) |

## 알아둘 점

- 20일 horizon + 5일 origin 간격이면 윈도우가 겹쳐 유효 표본이 훨씬 작다 (`groups` 의 `n_eff`). MDD q0.05 꼬리는 위기 몇 번이 사실상 표본 전부라 검정력이 낮다.
- 코스피 원본에 `O=H=L=C` 인 평탄 봉이 있을 수 있다 (`data/interim/clean_report.json`). 기본은 표시만 하고 유지(`drop_flat_bars: false`).
- 벤치마크 수치는 Mistral 발표값이고, 사전학습에 실제 지수 데이터가 섞였을 가능성을 배제할 수 없다. 이 밴드는 위험 구간을 읽는 참고 도구이며 매매 신호로 쓰기엔 검증이 부족하다.
- 모델 가중치 라이선스: Modified MIT (월 매출 2천만 달러 초과 기업은 상용 라이선스 필요), 코드는 Apache-2.0.
