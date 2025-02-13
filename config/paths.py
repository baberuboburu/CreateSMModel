import os

# Root Dir from this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data Dir
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ETTH1_DATA = os.path.join(DATA_DIR, "ETTh1.csv")
ETTH2_DATA = os.path.join(DATA_DIR, "ETTh2.csv")
DNPUs_DATA = os.path.join(DATA_DIR, "IO.dat")

# Model Dir
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TCN_MODEL_DIR = os.path.join(MODELS_DIR, "tcn")
TIMESFM_MODEL_DIR = os.path.join(MODELS_DIR, "timesfm")
TTM_MODEL_DIR = os.path.join(MODELS_DIR, "ttm")
TTM_MODEL_ZEROSHOT_DIR = os.path.join(TTM_MODEL_DIR, "zeroshots")
TTM_MODEL_FEWSHOT_DIR = os.path.join(TTM_MODEL_DIR, "fewshots")

# Result Dir (inference, analysis)
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, "predictions")
ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "analysis")
ANALYSIS_DNPUs_DIR = os.path.join(ANALYSIS_DIR, "DNPUs")
ANALYSIS_ETTh1_DIR = os.path.join(ANALYSIS_DIR, "ETTh1")

# Result Dir (inference) each architecture
PREDICTIONS_TCN_DIR = os.path.join(PREDICTIONS_DIR, "tcn")
PREDICTIONS_TIMESFM_DIR = os.path.join(PREDICTIONS_DIR, "timesfm")
PREDICTIONS_TIMESFM_ZEROSHOT_DIR = os.path.join(PREDICTIONS_TIMESFM_DIR, "zeroshots")
PREDICTIONS_TIMESFM_FEWSHOT_DIR = os.path.join(PREDICTIONS_TIMESFM_DIR, "fewshots")
PREDICTIONS_TTM_DIR = os.path.join(PREDICTIONS_DIR, "ttm")

# Result Dir (analysis) each architecture
ANALYSIS_ETTH1_DIR = os.path.join(ANALYSIS_DIR, "ETTh1")
ANALYSIS_IO_DIR = os.path.join(ANALYSIS_DIR, "IO")

# hyperparameter file
HYPERPARAMS_FILE = os.path.join(PROJECT_ROOT, "config", "hyperparams.py")
