from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
MODEL_DIR = ROOT_DIR / 'models'
DATA_FILE = DATA_DIR / 'time_series_covid19_confirmed_global.csv'
MODEL_FILE = MODEL_DIR / 'covid_predictor.pth'

N_FEATURES = 1
N_HIDDEN = 4
SEQ_LENGTH = 7  # Changed from 5 to 7 to match the API input
N_LAYERS = 1

RANDOM_SEED = 42
NUM_EPOCHS = 100
VERBOSE = 10
PATIENCE = 50

# These values should be calculated during training and saved with the model
MIN = 0  # Placeholder
MAX = 1  # Placeholder
