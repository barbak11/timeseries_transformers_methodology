from pathlib import Path


# Shared paths
DATA_DIR = Path("data")
DISEASE_DIR = Path("Zab/processed")
ARTIFACTS_ROOT = Path("artifacts")
ARTIFACT_ROOT_ALIASES = {
    # Keep legacy filename prefixes (e.g. coldstart_2_*) while writing to existing folder.
    "coldstart_2": "coldstart",
}

TRAFFIC_CLASSIC_PARQUET = DATA_DIR / "traffic_classic.parquet"
TRAFFIC_BOOST_PARQUET = DATA_DIR / "traffic_boosting.parquet"
DISEASE_SEQUENTIAL_PARQUET = DISEASE_DIR / "disease_sequential_dataset.parquet"
DISEASE_SNAPSHOT_PARQUET = DISEASE_DIR / "disease_snapshot_dataset.parquet"

SPARSE_CLASSIC_PARQUET = DATA_DIR / "df_classic_sparseness_full.parquet"
SPARSE_BOOST_PARQUET = DATA_DIR / "df_boosting_sparseness_full.parquet"
PCR_CLASSIC_PARQUET = DISEASE_DIR / "pcr_sequential.parquet"
PCR_BOOST_PARQUET = DISEASE_DIR / "pcr_snapshot_boosting.parquet"


def artifact_dirs(spec_key: str) -> dict[str, Path]:
    """Построить стандартные каталоги артефактов для специфики.

    Args:
        spec_key: Ключ специфики (например, `sparse`, `short`, `coldstart_2`).

    Returns:
        Словарь путей: base/models/predictions/metrics.
    """
    root_name = ARTIFACT_ROOT_ALIASES.get(spec_key, spec_key)
    base = ARTIFACTS_ROOT / root_name
    return {
        "base": base,
        "models": base / "models",
        "predictions": base / "predictions",
        "metrics": base / "metrics",
    }


# Shared baseline hyperparameters
RANDOM_SEED = 42
TEST_HORIZON_YEARS = 1

XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

LSTM_SEQ_LEN = 14
LSTM_HIDDEN = 64
LSTM_BATCH = 1024
LSTM_LR = 1e-3
LSTM_GRAD_CLIP = 1.0
LSTM_VAL_FRAC = 0.15
LSTM_ES_PATIENCE = 4

TARGET_COL = "checks_cnt"
WEATHER_COLS = [
    "apparent_temperature_max",
    "apparent_temperature_mean",
    "apparent_temperature_min",
    "relative_humidity_2m_mean",
    "relative_humidity_2m_min",
    "snowfall_sum",
    "sunshine_duration",
    "temperature_2m_max",
    "temperature_2m_mean",
    "temperature_2m_min",
    "windspeed_10m_mean",
]


SHORT_CONFIG = {
    "SPEC_KEY": "short",
    "RETRAIN": True,
    "RUN_TRAFFIC_TASK": True,
    "RUN_DISEASE_TASK": True,
    "PRINT_METRICS_TO_STDOUT": False,
    "SHORT_OPEN_DAYS_MIN": 7,
    "SHORT_OPEN_DAYS_MAX": 90,
    "ETS_MIN_TRAIN_DAYS": 14,
    "SARIMAX_MIN_TRAIN_DAYS": 20,
    "LSTM_MIN_TRAIN_DAYS": 30,
    "LSTM_SEQ_LEN": 14,
    "LSTM_HIDDEN": 64,
    "LSTM_EPOCHS": 50,
    "LSTM_BATCH": 1024,
    "LSTM_LR": 1e-3,
    "LSTM_GRAD_CLIP": 1.0,
    "LSTM_ES_PATIENCE": 4,
    "LSTM_VAL_FRAC": 0.15,
    "LSTM_VAL_BATCH": 1024,
    "LSTM_CONFIGS": (
        {"hidden": 64, "layers": 1, "dropout": 0.0},
        {"hidden": 96, "layers": 1, "dropout": 0.0},
        {"hidden": 128, "layers": 2, "dropout": 0.1},
    ),
    "OPTUNA_N_TRIALS": 50,
    "DISEASE_SHORT_SHARE": 0.10,
    "DISEASE_MIN_WEEKS": 50,
    "DISEASE_MAX_WEEKS": 150,
    "TFT_LOOKBACK": 28,
    "TFT_HORIZON": 7,
    "TFT_HIDDEN": 32,
    "TFT_LSTM_LAYERS": 1,
    "TFT_ATTN_HEADS": 2,
    "TFT_DROPOUT": 0.15,
    "TFT_EPOCHS": 50,
    "TFT_BATCH": 64,
    "TFT_LR": 1e-3,
    "TFT_GRAD_CLIP": 1.0,
    "TFT_PATIENCE": 7,
    "TFT_TRAIN_STRIDE": 1,
    "TFT_VAL_FRAC": 0.15,
    "TFT_KNOWN_FUTURE_COLS": [
        "weekday_num", "month_num", "week_num", "day_num", "year_num",
        "holiday_flg", "weekend_day", "working_day",
        "first_january", "new_year", "may_holidays", "victory_day",
        "russia_day", "first_september", "unity_day", "preholidays",
        "school_autumn_spring_holiday", "school_holiday",
        "school_summer_holiday", "school_winter_holiday", "quarter_num",
    ],
    "TFT_OBSERVED_PAST_COLS": [
        "apparent_temperature_mean",
        "temperature_2m_mean",
        "windspeed_10m_mean",
        "relative_humidity_2m_mean",
        "snowfall_sum",
        "sunshine_duration",
    ],
    "TFT_STATIC_CAT_COLS": [
        "region_name", "restformat_name", "restrentgroup_name",
        "mcdonaldstype_name", "wctype_name", "flg_mall",
    ],
    "TFT_STATIC_NUM_COLS": [
        "cashdesk_cnt", "kiosk_cnt", "seat_cnt",
        "ownareaall_sqm", "workinghours_cnt",
    ],
    "TFT_CAT_EMB_DIM": 8,
    "TFT_ARCH_CONFIGS": (
        {"name": "TFT-compact", "lookback": 12, "horizon": 1, "d": 24, "lstm_layers": 1, "n_heads": 1, "drop": 0.10},
        {"name": "TFT-base", "lookback": 16, "horizon": 1, "d": 32, "lstm_layers": 1, "n_heads": 2, "drop": 0.15},
        {"name": "TFT-context", "lookback": 24, "horizon": 1, "d": 48, "lstm_layers": 2, "n_heads": 4, "drop": 0.20},
    ),
}


COLDSTART_CONFIG = {
    "SPEC_KEY": "coldstart_2",
    "RETRAIN": True,
    "RUN_TRAFFIC_TASK": True,
    "RUN_DISEASE_TASK": True,
    "PRINT_METRICS_TO_STDOUT": False,
    "SHORT_OPEN_DAYS_MIN": 7,
    "SHORT_OPEN_DAYS_MAX": 90,
    "OPTUNA_N_TRIALS": 50,
    "DISEASE_SHORT_SHARE": 0.10,
    "DISEASE_MIN_WEEKS": 50,
    "DISEASE_MAX_WEEKS": 150,
    "TFT_LOOKBACK": 28,
    "TFT_HORIZON": 7,
    "TFT_HIDDEN": 32,
    "TFT_LSTM_LAYERS": 1,
    "TFT_ATTN_HEADS": 2,
    "TFT_DROPOUT": 0.15,
    "TFT_EPOCHS": 50,
    "TFT_BATCH": 64,
    "TFT_LR": 1e-3,
    "TFT_GRAD_CLIP": 1.0,
    "TFT_PATIENCE": 7,
    "TFT_TRAIN_STRIDE": 1,
    "TFT_VAL_FRAC": 0.15,
    "TFT_KNOWN_FUTURE_COLS": SHORT_CONFIG["TFT_KNOWN_FUTURE_COLS"],
    "TFT_OBSERVED_PAST_COLS": [],
    "TFT_STATIC_CAT_COLS": SHORT_CONFIG["TFT_STATIC_CAT_COLS"],
    "TFT_STATIC_NUM_COLS": SHORT_CONFIG["TFT_STATIC_NUM_COLS"],
    "TFT_CAT_EMB_DIM": 8,
    "TFT_ARCH_CONFIGS": (
        {"name": "TFT-coldstart-lite", "lookback": 14, "horizon": 7, "d": 32, "lstm_layers": 1, "n_heads": 2, "drop": 0.10},
        {"name": "TFT-coldstart-balanced", "lookback": 21, "horizon": 7, "d": 40, "lstm_layers": 1, "n_heads": 4, "drop": 0.15},
        {"name": "TFT-coldstart-context", "lookback": 28, "horizon": 7, "d": 48, "lstm_layers": 2, "n_heads": 4, "drop": 0.20},
    ),
    "TFT_SCALE_GROUP_LEVELS": (
        ("region_name", "restformat_name", "restrentgroup_name", "flg_mall"),
        ("region_name", "restformat_name", "flg_mall"),
        ("region_name", "restformat_name"),
        ("restformat_name", "flg_mall"),
        ("restformat_name",),
    ),
    "TFT_SCALE_MIN_SAMPLES": 60,
}


SPARSE_CONFIG = {
    "SPEC_KEY": "sparse",
    "DISEASE_SPEC_KEY": "disease",
    "RUN_SPARSE_TASK": True,
    "RUN_DISEASE_TASK": True,
    "LSTM_EPOCHS": 50,
    "PATCHTST_SEQ_LEN": 28,
    "PATCHTST_PATCH": 7,
    "PATCHTST_D_MODEL": 64,
    "PATCHTST_HEADS": 4,
    "PATCHTST_LAYERS": 2,
    "PATCHTST_EPOCHS": 50,
    "PATCHTST_BATCH": 1024,
    "PATCHTST_LR": 1e-3,
    "PATCHTST_CONFIGS": [
        {"name": "compact", "seq_len": 28, "patch_len": 7, "d_model": 32, "n_heads": 4, "n_layers": 2},
        {"name": "base", "seq_len": 56, "patch_len": 7, "d_model": 64, "n_heads": 4, "n_layers": 3},
    ],
}


CORRELATED_CONFIG = {
    "SPEC_KEY": "correlated",
    "RETRAIN": True,
    "ENABLE_TASK_TRAFFIC": True,
    "ENABLE_TASK_DISEASE": True,
    "VAL_DAYS_BEFORE_TEST": 90,
    "VAL_WEEKS_BEFORE_TEST": 12,
    "ETS_MIN_TRAIN_DAYS": 14,
    "SARIMA_MIN_TRAIN_DAYS": 20,
    "LSTM_EPOCHS_MAX": 50,
    "LSTM_MIN_TRAIN_DAYS": 30,
    "OPTUNA_N_TRIALS": 50,
    "SARIMA_GRID_MAX": 10,
    "SARIMA_MAXITER": 120,
    "SARIMA_MAXITER_DISEASE": 120,
    "SARIMA_MAXITER_DISEASE_FAST": 40,
    "LSTM_CONFIGS": (
        {"hidden": 64, "layers": 2, "dropout": 0.1},
        {"hidden": 32, "layers": 1, "dropout": 0.1},
        {"hidden": 128, "layers": 3, "dropout": 0.15},
    ),
    "TFT_LOOKBACK": 24,
    "TFT_BATCH": 512,
    "TFT_EPOCHS_MAX": 50,
    "TFT_LR": 1e-3,
    "TFT_GRAD_CLIP": 1.0,
    "TFT_ES_PATIENCE": 4,
    "TFT_VAL_FRAC": 0.15,
    "TFT_CONFIGS": (
        {"name": "TFT_small", "hidden": 32, "layers": 1, "dropout": 0.10},
        {"name": "TFT_medium", "hidden": 64, "layers": 2, "dropout": 0.15},
        {"name": "TFT_large", "hidden": 128, "layers": 3, "dropout": 0.20},
    ),
    "MVT_LOOKBACK": 56,
    "MVT_HORIZON": 7,
    "MVT_HIDDEN": 64,
    "MVT_FF_DIM": 128,
    "MVT_HEADS_TIME": 4,
    "MVT_HEADS_SERIES": 4,
    "MVT_N_TEMP_LAYERS": 2,
    "MVT_N_SERIES_LAYERS": 1,
    "MVT_DROPOUT": 0.1,
    "MVT_BATCH": 16,
    "MVT_EPOCHS": 50,
    "MVT_LR": 5e-4,
    "MVT_WEIGHT_DECAY": 1e-4,
    "MVT_GRAD_CLIP": 1.0,
    "MVT_PATIENCE": 7,
    "MVT_VAL_FRAC": 0.15,
    "MVT_ONECYCLE_PCT_START": 0.12,
    "MVT_CONFIGS": (
        {
            "name": "MV-Transformer-compact",
            "lookback": 84,
            "horizon": 7,
            "hidden": 64,
            "heads_time": 4,
            "heads_series": 4,
            "n_temp_layers": 2,
            "n_series_layers": 1,
            "ff_dim": 128,
            "dropout": 0.10,
        },
        {
            "name": "MV-Transformer-base",
            "lookback": 56,
            "horizon": 7,
            "hidden": 64,
            "heads_time": 4,
            "heads_series": 4,
            "n_temp_layers": 2,
            "n_series_layers": 2,
            "ff_dim": 128,
            "dropout": 0.10,
        },
        {
            "name": "MV-Transformer-context",
            "lookback": 56,
            "horizon": 7,
            "hidden": 128,
            "heads_time": 4,
            "heads_series": 4,
            "n_temp_layers": 2,
            "n_series_layers": 1,
            "ff_dim": 256,
            "dropout": 0.20,
        },
    ),
}
