# COVID-19 예측 API

이 프로젝트는 FastAPI를 사용하여 COVID-19 확진자 수를 예측하는 머신러닝 모델을 API로 제공합니다.

본 프로젝트는 다음 튜토리얼을 기반으로 개발되었습니다:
https://pseudo-lab.github.io/Tutorial-Book/chapters/time-series/Ch4-LSTM.html

주피터 노트북 형식이었던 원본 코드를 API 서비스에 맞게 수정하였습니다.

## 주요 기능

- COVID-19 일일 확진자 수 데이터 전처리
- LSTM 기반의 시계열 예측 모델
- FastAPI를 이용한 RESTful API 제공
- 모델 훈련 및 저장 기능
- 실시간 예측 기능

## 기술 스택

- Python 3.8+
- FastAPI
- PyTorch
- Pandas
- NumPy
- Gunicorn (WSGI HTTP 서버)
- Uvicorn (ASGI 서버)

## 설치 방법

1. 저장소 클론:
   ```
   git clone https://github.com/yourusername/covid-prediction-api.git
   cd covid-prediction-api
   ```

2. 가상 환경 생성 및 활성화:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
   ```

3. 필요한 패키지 설치:
   ```
   pip install -r requirements.txt
   ```

## 사용 방법

1. 모델 훈련:
   ```
   python -m app.ml.train
   ```

2. API 서버 실행 (개발 환경):
   ```
   uvicorn app.main:app --reload
   ```

3. API 서버 실행 (프로덕션 환경):
   ```
   gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

4. API 문서 확인:
   브라우저에서 `http://localhost:8000/docs` 접속

## API 엔드포인트

- `GET /`: 웰컴 메시지
- `POST /predict`: COVID-19 확진자 수 예측
  - 요청 바디: 최근 7일간의 확진자 수 데이터
  - 응답: 다음 날의 예측 확진자 수

## 설정

`app/core/config.py` 파일에서 다음 설정을 변경할 수 있습니다:
- 모델 하이퍼파라미터
- 데이터 경로
- 모델 저장 경로

## 프로젝트 구조

```
covid_prediction_api/
│
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 애플리케이션 진입점
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py     # API 엔드포인트 정의
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # 설정 파일
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── train.py
│   └── schemas/
│       ├── __init__.py
│       └── prediction.py    # Pydantic 모델 (요청/응답 스키마)
│
├── data/
│   └── time_series_covid19_confirmed_global.csv
│
├── models/
│   └── covid_predictor.pth
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 프로덕션 배포

프로덕션 환경에서는 Gunicorn과 Uvicorn을 함께 사용하여 애플리케이션을 실행합니다. 이 방식은 다음과 같은 이점을 제공합니다:

- 다중 워커 프로세스 지원
- 자동 프로세스 관리 및 재시작
- 로드 밸런싱

서버를 시작하려면 다음 명령을 사용하세요:

```
gunicorn app.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

- `--workers 4`: 4개의 워커 프로세스를 생성합니다. 서버의 CPU 코어 수에 따라 조정하세요.
- `--worker-class uvicorn.workers.UvicornWorker`: Uvicorn 워커를 사용합니다.
- `--bind 0.0.0.0:8000`: 모든 네트워크 인터페이스에서 8000 포트로 바인딩합니다.

보안 및 성능 최적화를 위해 프록시 서버(예: Nginx)를 앞단에 두는 것을 권장합니다.