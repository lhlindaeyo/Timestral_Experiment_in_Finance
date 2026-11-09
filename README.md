# Timestral 기반 MDD · 변동성 밴드 예측

S&P500 · 나스닥 · 코스피 일봉으로 **5일/20일 MDD 분포**와 **20일 Yang-Zhang 변동성 분포**를 Timestral(5M/28M/123M/209M)로 제로샷 예측하고, ARIMA·무조건 분위수 기준선과 캘리브레이션을 비교하는 실험.

핵심 아이디어: 모델은 시계열의 분위수만 내놓으므로, **목표 지표를 시계열(trailing m5, m20, yz20)로 만든 뒤 그 시계열을 예측**한다.
"t 이후 h일의 지표" = "t+h 시점의 trailing 값" 이라 모델의 h 번째 스텝 분위수가 곧 원하는 분포이고, 라벨과 입력은 `src/targets/` 의 같은 함수로 만든다.

## 빠른 시작

```bash
python -m venv .venv && .venv\Scripts\activate        # (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt                       # torch>=2.4, timestral 포함

python run.py fetch                                   # 1. 데이터 (Yahoo Finance)
python run.py build                                   # 2. 정제 + m5/m20/yz20 계산
pytest                                                #    타깃 정의 / 지표 / 파이프라인 테스트
python run.py download-models --sizes 5M 209M         # 3. 가중치 -> models/ (209M 약 837MB)
python run.py latency --sizes 5M 28M 123M 209M        #    로컬 추론 지연시간 측정
python run.py backtest --tickers KOSPI --models arima climatology timestral-5M --ctx 512 1024   # 4. 한 지수로 먼저
python run.py backtest                                #    전체 (3 지수 × 6 모델 × 4 지표 × ctx 4종, 오래 걸림)
python run.py evaluate                                # 5. results/metrics/summary_*.csv
python run.py plot                                    #    results/figures/*.png
python run.py latest --ticker KOSPI --model timestral-209M --ctx 1024    # 오늘 기준 밴드
```

## 폴더 구성

```
├─ run.py                    # CLI 진입점 (위 명령)
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
│   ├─ forecast/             # timestral_runner.py, arima_runner.py, climatology_runner.py, base.py
│   ├─ eval/                 # backtest.py(rolling origin), metrics.py, coverage_tests.py, summarize.py, bands.py
│   └─ viz/                  # plots.py (fan chart, 캘리브레이션, 크기별 비교)
├─ notebooks/                # 01_eda, 02_target_check, 03_zero_shot_demo, 04_backtest, 05_report
├─ tests/                    # 손계산 대조 단위 테스트, 누수·정렬 테스트
├─ results/{forecasts,metrics,figures}/
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
| Origin | 5일 간격, 모든 모델·ctx 가 같은 origin 사용 (가장 긴 ctx 2048 이 차는 시점 이후) |
| 누수 방지 | origin t 의 입력은 t 까지의 trailing 값만. `tests/test_pipeline.py::test_no_lookahead` |
| ARIMA | 지수·지표별 차수 1회 선택(ADF→d, AIC→p,q, 테스트 이전 데이터), origin 20개마다 재적합, 표준화 잔차 부트스트랩 × 예측표준오차 |
| 기준선 | 무조건 경험적 분위수(expanding) |
| 평가 | pinball(0.05/0.10/…), 위반율, Kupiec(전체) · Christoffersen(겹치지 않는 부분표본), coverage80, MASE(중앙값), 추론 시간. 위기/평시 분리 (origin 날짜 기준) |

예측 원본은 `results/forecasts/` 에 모델×지수×지표×horizon×ctx 별 parquet 로 저장하고(git 제외), 지표와 그림은 항상 이 저장본에서 다시 계산한다.

## 알아둘 점

- 20일 horizon + 5일 origin 간격이면 윈도우가 겹쳐 유효 표본이 훨씬 작다 (`summary_groups.csv` 의 `n_eff`). MDD q0.05 꼬리는 위기 몇 번이 사실상 표본 전부라 검정력이 낮다.
- 코스피 원본에 `O=H=L=C` 인 평탄 봉이 있을 수 있다 (`data/interim/clean_report.json`). 기본은 표시만 하고 유지(`drop_flat_bars: false`).
- 벤치마크 수치는 Mistral 발표값이고, 사전학습에 실제 지수 데이터가 섞였을 가능성을 배제할 수 없다. 이 밴드는 위험 구간을 읽는 참고 도구이며 매매 신호로 쓰기엔 검증이 부족하다.
- 모델 가중치 라이선스: Modified MIT (월 매출 2천만 달러 초과 기업은 상용 라이선스 필요), 코드는 Apache-2.0.
