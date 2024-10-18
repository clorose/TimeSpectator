from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 데이터 디렉토리
DATA_DIR = ROOT_DIR / 'data'

# 모델 저장 디렉토리
MODEL_DIR = ROOT_DIR / 'models'

# 데이터 파일 경로
DATA_FILE = DATA_DIR / 'time_series_covid19_confirmed_global.csv'

# 모델 파일 경로
MODEL_FILE = MODEL_DIR / 'covid_predictor.pth'

# 모델 하이퍼파라미터
N_FEATURES = 1  # 입력 피처 수 (일일 확진자 수)
N_HIDDEN = 4    # LSTM 히든 레이어의 뉴런 수
SEQ_LENGTH = 5  # 입력 시퀀스 길이 (며칠 동안의 데이터를 볼 것인가)
N_LAYERS = 1    # LSTM 레이어 수

# 학습 관련 설정
RANDOM_SEED = 42  # 재현성을 위한 랜덤 시드
NUM_EPOCHS = 100  # 총 학습 에폭 수
VERBOSE = 10      # 몇 에폭마다 로그를 출력할 것인가
PATIENCE = 50     # Early stopping을 위한 인내심

# API 관련 설정
API_V1_STR = "/api/v1"
PROJECT_NAME = "COVID-19 Prediction API"

# 이 값들은 학습 과정에서 계산되어야 하며, 모델과 함께 저장되어야 합니다.
MIN = None  # 데이터 정규화를 위한 최소값
MAX = None  # 데이터 정규화를 위한 최대값