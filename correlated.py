# -*- coding: utf-8 -*-
"""
Две задачи с единым пайплайном моделей (ETS, SARIMA, XGBoost, LSTM, MV-Transformer):

1) Трафик — кластер ресторанов, дневная сетка.
2) Заболеваемость — недельные ряды (классика для ETS/SARIMA/LSTM/MVT); XGBoost на snapshot-таблице
   (лаги от даты прогноза).

Сплит по датам: последний год = тест; для boosting по snapshot — по snapshot_dt.

Запуск:  python nir_correlated_baseline.py

Флаг RETRAIN (ниже): при True — обучить LSTM и 4 конфигурации MV-Transformer и сохранить веса в
artifacts/correlated/models/ (имена файлов с префиксом задачи: traffic / disease). При False —
загрузить сохранённые чекпоинты и выполнить только инференс (без переобучения).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.tools.sm_exceptions import ConvergenceWarning as SMConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm
import optuna
from xgboost import XGBRegressor

import config as cfg

# См. docstring модуля: артефакты в artifacts/correlated/; префиксы имён — traffic / disease.
SC = cfg.CORRELATED_CONFIG
RETRAIN = SC["RETRAIN"]
# Выбор задач для запуска из run_all_tasks.
ENABLE_TASK_TRAFFIC = SC["ENABLE_TASK_TRAFFIC"]
ENABLE_TASK_DISEASE = SC["ENABLE_TASK_DISEASE"]

optuna.logging.set_verbosity(optuna.logging.WARNING)
TARGET_COL = cfg.TARGET_COL
WEATHER_COLS = cfg.WEATHER_COLS

SPEC_KEY = SC["SPEC_KEY"]
RANDOM_SEED = cfg.RANDOM_SEED
DATA_DIR = cfg.DATA_DIR
CLASSIC_PARQUET = cfg.TRAFFIC_CLASSIC_PARQUET
BOOST_PARQUET = cfg.TRAFFIC_BOOST_PARQUET
DISEASE_SEQUENTIAL_PARQUET = cfg.DISEASE_SEQUENTIAL_PARQUET
DISEASE_SNAPSHOT_PARQUET = cfg.DISEASE_SNAPSHOT_PARQUET
CLUSTER_CSV = Path("artifacts/correlated_group/best_correlated_series.csv")

TEST_HORIZON_YEARS = cfg.TEST_HORIZON_YEARS

# validation: отрезок перед test (для подбора гиперпараметров)
VAL_DAYS_BEFORE_TEST = SC["VAL_DAYS_BEFORE_TEST"]
VAL_WEEKS_BEFORE_TEST = SC["VAL_WEEKS_BEFORE_TEST"]

ETS_MIN_TRAIN_DAYS = SC["ETS_MIN_TRAIN_DAYS"]
SARIMA_MIN_TRAIN_DAYS = SC["SARIMA_MIN_TRAIN_DAYS"]

LSTM_SEQ_LEN = cfg.LSTM_SEQ_LEN
LSTM_HIDDEN = cfg.LSTM_HIDDEN
LSTM_EPOCHS_MAX = SC["LSTM_EPOCHS_MAX"]
LSTM_BATCH = cfg.LSTM_BATCH
LSTM_LR = cfg.LSTM_LR
LSTM_GRAD_CLIP = cfg.LSTM_GRAD_CLIP
LSTM_MIN_TRAIN_DAYS = SC["LSTM_MIN_TRAIN_DAYS"]
LSTM_ES_PATIENCE = cfg.LSTM_ES_PATIENCE
LSTM_VAL_FRAC = cfg.LSTM_VAL_FRAC

XGB_N_ESTIMATORS = cfg.XGB_N_ESTIMATORS
XGB_MAX_DEPTH = cfg.XGB_MAX_DEPTH
XGB_LEARNING_RATE = cfg.XGB_LEARNING_RATE
OPTUNA_N_TRIALS = SC["OPTUNA_N_TRIALS"]
SARIMA_GRID_MAX = SC["SARIMA_GRID_MAX"]
SARIMA_MAXITER = SC["SARIMA_MAXITER"]
SARIMA_MAXITER_DISEASE = SC["SARIMA_MAXITER_DISEASE"]
SARIMA_MAXITER_DISEASE_FAST = SC["SARIMA_MAXITER_DISEASE_FAST"]

LSTM_CONFIGS: tuple[dict[str, float | int], ...] = SC["LSTM_CONFIGS"]

TFT_LOOKBACK = SC["TFT_LOOKBACK"]
TFT_BATCH = SC["TFT_BATCH"]
TFT_EPOCHS_MAX = SC["TFT_EPOCHS_MAX"]
TFT_LR = SC["TFT_LR"]
TFT_GRAD_CLIP = SC["TFT_GRAD_CLIP"]
TFT_ES_PATIENCE = SC["TFT_ES_PATIENCE"]
TFT_VAL_FRAC = SC["TFT_VAL_FRAC"]
TFT_CONFIGS: tuple[dict[str, float | int | str], ...] = SC["TFT_CONFIGS"]

MVT_LOOKBACK = SC["MVT_LOOKBACK"]
MVT_HORIZON = SC["MVT_HORIZON"]
MVT_HIDDEN = SC["MVT_HIDDEN"]
MVT_FF_DIM = SC["MVT_FF_DIM"]
MVT_HEADS_TIME = SC["MVT_HEADS_TIME"]
MVT_HEADS_SERIES = SC["MVT_HEADS_SERIES"]
MVT_N_TEMP_LAYERS = SC["MVT_N_TEMP_LAYERS"]
MVT_N_SERIES_LAYERS = SC["MVT_N_SERIES_LAYERS"]
MVT_DROPOUT = SC["MVT_DROPOUT"]
MVT_BATCH = SC["MVT_BATCH"]
MVT_EPOCHS = SC["MVT_EPOCHS"]
MVT_LR = SC["MVT_LR"]
MVT_WEIGHT_DECAY = SC["MVT_WEIGHT_DECAY"]
MVT_GRAD_CLIP = SC["MVT_GRAD_CLIP"]
MVT_PATIENCE = SC["MVT_PATIENCE"]
MVT_VAL_FRAC = SC["MVT_VAL_FRAC"]
MVT_ONECYCLE_PCT_START = SC["MVT_ONECYCLE_PCT_START"]
MVT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Явный перебор (без Optuna): только lookback, hidden, ff_dim, temporal/cross layers, dropout.
MVT_CONFIGS_TRAFFIC: tuple[dict[str, str | int | float], ...] = (
    {
        "name": "TRAFFIC_T1",
        "lookback": 56,
        "hidden": 64,
        "ff_dim": 128,
        "n_temp_layers": 2,
        "n_series_layers": 1,
        "dropout": 0.1,
    },
    {
        "name": "TRAFFIC_T2",
        "lookback": 84,
        "hidden": 64,
        "ff_dim": 128,
        "n_temp_layers": 2,
        "n_series_layers": 1,
        "dropout": 0.1,
    },
    {
        "name": "TRAFFIC_T3",
        "lookback": 56,
        "hidden": 64,
        "ff_dim": 128,
        "n_temp_layers": 2,
        "n_series_layers": 2,
        "dropout": 0.1,
    },
    {
        "name": "TRAFFIC_T4",
        "lookback": 56,
        "hidden": 128,
        "ff_dim": 256,
        "n_temp_layers": 2,
        "n_series_layers": 1,
        "dropout": 0.2,
    },
)

MVT_CONFIGS_DISEASE: tuple[dict[str, str | int | float], ...] = (
    {
        "name": "DISEASE_T1",
        "lookback": 26,
        "hidden": 32,
        "ff_dim": 64,
        "n_temp_layers": 2,
        "n_series_layers": 1,
        "dropout": 0.1,
    },
    {
        "name": "DISEASE_T2",
        "lookback": 52,
        "hidden": 32,
        "ff_dim": 64,
        "n_temp_layers": 2,
        "n_series_layers": 1,
        "dropout": 0.1,
    },
    {
        "name": "DISEASE_T3",
        "lookback": 52,
        "hidden": 64,
        "ff_dim": 128,
        "n_temp_layers": 3,
        "n_series_layers": 1,
        "dropout": 0.1,
    },
    {
        "name": "DISEASE_T4",
        "lookback": 52,
        "hidden": 64,
        "ff_dim": 128,
        "n_temp_layers": 2,
        "n_series_layers": 2,
        "dropout": 0.2,
    },
)

MVT_CAL_COLS = [
    "weekday_num", "month_num", "day_num",
    "holiday_flg", "weekend_day", "working_day",
]

@dataclass(frozen=True)
class TaskSpec:
    """Одна задача прогноза. xgb_parquet — parquet для XGBoost (disease: snapshot с лагами)."""

    name: str
    spec_key: str
    title_ru: str
    freq: str
    seasonal_periods: int
    value_col: str
    xgb_target_col: str
    xgb_parquet: Path
    ets_min_train: int
    sarima_min_train: int
    lstm_seq_len: int
    lstm_min_train: int
    mvt_lookback: int
    mvt_horizon: int
    filter_xgb_cluster: bool = True
    #: Каталог под artifacts/ (специфика correlated — обе задачи пишут в один корень).
    artifact_root: str = SPEC_KEY


def paths_for_task(task: TaskSpec):
    root = Path("artifacts") / task.artifact_root
    models = root / "models"
    preds = root / "predictions"
    metrics = root / "metrics"
    return root, models, preds, metrics


def setup_dirs(task: TaskSpec):
    _, models, preds, metrics = paths_for_task(task)
    for d in (models, preds, metrics):
        d.mkdir(parents=True, exist_ok=True)


def series_file_slug(series_id) -> str:
    s = str(series_id)
    if s.isdigit() and len(s) < 24:
        return s
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def rmse(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def wape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100)


def log_result(
    results,
    all_predictions,
    model_name,
    pred_df,
    train_sec,
    infer_sec,
    *,
    tuning_time_sec: float | None = None,
):
    if pred_df is None or len(pred_df) == 0:
        return
    row = {
        "model": model_name,
        "rmse": rmse(pred_df["y_true"], pred_df["y_pred"]),
        "wape": wape(pred_df["y_true"], pred_df["y_pred"]),
        "train_time_sec": float(train_sec),
        "inference_time_sec": float(infer_sec),
    }
    if tuning_time_sec is not None:
        row["tuning_time_sec"] = float(tuning_time_sec)
    results.append(row)
    all_predictions.append(pred_df)


def save_metrics_files(task: TaskSpec, results):
    dfm = pd.DataFrame(results)
    _, _, _, metrics_dir = paths_for_task(task)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(metrics_dir / f"{task.name}_metrics.csv", index=False)
    dfm.to_json(
        metrics_dir / f"{task.name}_metrics.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    return dfm


def _safe_qcut_3(values: pd.Series, labels: list[str]) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    if x.notna().sum() < 3 or x.nunique(dropna=True) < 3:
        r = x.rank(method="first")
        return pd.cut(r, bins=3, labels=labels, include_lowest=True).astype(str)
    try:
        return pd.qcut(x, q=3, labels=labels, duplicates="drop").astype(str)
    except Exception:
        r = x.rank(method="first")
        return pd.cut(r, bins=3, labels=labels, include_lowest=True).astype(str)


def _run_key_model_slice_analysis(task: TaskSpec, train_df: pd.DataFrame, results, all_predictions):
    """Доп. анализ по 3 моделям: XGBoost, лучший MVT, лучший TFT."""
    if not results or not all_predictions:
        return
    _, _, _, metrics_dir = paths_for_task(task)
    dfm = pd.DataFrame(results)
    if len(dfm) == 0:
        return

    xgb_rows = dfm[dfm["model"] == "XGBoost"].sort_values("rmse")
    mvt_rows = dfm[dfm["model"].astype(str).str.startswith("MV-Transformer-")].sort_values("rmse")
    tft_rows = dfm[dfm["model"].astype(str).str.startswith("TFT_")].sort_values("rmse")
    if xgb_rows.empty or mvt_rows.empty or tft_rows.empty:
        print("Slice analysis: не хватает одной из ключевых моделей (XGBoost/MVT/TFT), пропуск.")
        return

    chosen = [xgb_rows.iloc[0]["model"], mvt_rows.iloc[0]["model"], tft_rows.iloc[0]["model"]]
    pred_map: dict[str, pd.DataFrame] = {}
    for r, p in zip(results, all_predictions):
        mn = str(r.get("model", ""))
        if mn in chosen and mn not in pred_map:
            pred_map[mn] = p.copy()
    if any(m not in pred_map for m in chosen):
        print("Slice analysis: не найдены prediction dataframe для части выбранных моделей.")
        return

    # unified preds for 3 selected models
    blocks = []
    for m in chosen:
        dfp = pred_map[m].copy()
        dfp["model_key"] = m
        if "series_id" not in dfp.columns:
            dfp["series_id"] = dfp.get("rest_id", "na").astype(str)
        blocks.append(dfp)
    dfp_all = pd.concat(blocks, ignore_index=True)
    dfp_all["y_true"] = pd.to_numeric(dfp_all["y_true"], errors="coerce")
    dfp_all["y_pred"] = pd.to_numeric(dfp_all["y_pred"], errors="coerce")
    dfp_all = dfp_all.dropna(subset=["y_true", "y_pred", "series_id"])
    if "date_dt" in dfp_all.columns:
        dfp_all["date_dt"] = pd.to_datetime(dfp_all["date_dt"], errors="coerce")

    # 1) series-level comparison
    ser_rows = []
    for (m, sid), g in dfp_all.groupby(["model_key", "series_id"], dropna=False):
        ser_rows.append(
            {
                "model": m,
                "series_id": str(sid),
                "rest_id": g["rest_id"].iloc[0] if "rest_id" in g.columns else np.nan,
                "rmse": rmse(g["y_true"], g["y_pred"]),
                "wape": wape(g["y_true"], g["y_pred"]),
                "n_points": len(g),
            }
        )
    df_series = pd.DataFrame(ser_rows)
    if task.name == "disease" and len(df_series):
        split = df_series["series_id"].astype(str).str.split("__", n=1, expand=True)
        if split.shape[1] == 2:
            df_series["city"] = split[0]
            df_series["age_group"] = split[1]
    winners = (
        df_series.sort_values(["series_id", "rmse"])
        .groupby("series_id", as_index=False)
        .first()[["series_id", "model", "rmse"]]
        .rename(columns={"model": "winner_model", "rmse": "winner_rmse"})
    )
    wins_count = winners["winner_model"].value_counts().to_dict()
    df_series.to_csv(metrics_dir / f"{task.name}_slice_series_comparison.csv", index=False)
    winners.to_csv(metrics_dir / f"{task.name}_slice_series_winners.csv", index=False)

    # level / volatility groups from train series stats
    stats_rows = []
    for sid, g in train_df.groupby("series_id", dropna=False):
        y = pd.to_numeric(g[task.value_col], errors="coerce").dropna()
        if len(y) == 0:
            continue
        mu = float(y.mean())
        sd = float(y.std())
        cv = float(sd / (abs(mu) + 1e-6))
        stats_rows.append({"series_id": str(sid), "series_mean": mu, "series_std": sd, "series_cv": cv})
    df_stats = pd.DataFrame(stats_rows)
    if len(df_stats) == 0:
        return
    df_stats["level_group"] = _safe_qcut_3(df_stats["series_mean"], ["low", "medium", "high"])
    df_stats["vol_group"] = _safe_qcut_3(df_stats["series_cv"], ["low_volatility", "medium_volatility", "high_volatility"])

    dfp_ext = dfp_all.merge(df_stats, on="series_id", how="left")

    def _agg_slice(df: pd.DataFrame, grp_col: str) -> pd.DataFrame:
        rows = []
        for (m, grp), gg in df.groupby(["model_key", grp_col], dropna=False):
            if pd.isna(grp):
                continue
            rows.append(
                {
                    "model": m,
                    grp_col: grp,
                    "rmse": rmse(gg["y_true"], gg["y_pred"]),
                    "wape": wape(gg["y_true"], gg["y_pred"]),
                    "n_points": len(gg),
                }
            )
        return pd.DataFrame(rows)

    df_level = _agg_slice(dfp_ext, "level_group")
    df_level.to_csv(metrics_dir / f"{task.name}_slice_level_comparison.csv", index=False)

    df_vol = _agg_slice(dfp_ext, "vol_group")
    df_vol.to_csv(metrics_dir / f"{task.name}_slice_volatility_comparison.csv", index=False)

    # 4) horizon slices
    hor_blocks = []
    for m in chosen:
        d = pred_map[m].copy()
        d["model_key"] = m
        if "series_id" not in d.columns:
            d["series_id"] = d.get("rest_id", "na").astype(str)
        if "date_dt" in d.columns:
            d["date_dt"] = pd.to_datetime(d["date_dt"], errors="coerce")
        if task.name == "disease" and "horizon_week" in d.columns:
            d["h_idx"] = pd.to_numeric(d["horizon_week"], errors="coerce")
        elif task.name == "disease" and "horizon_weeks" in d.columns:
            d["h_idx"] = pd.to_numeric(d["horizon_weeks"], errors="coerce")
        else:
            d = d.sort_values(["series_id", "date_dt"])
            d["h_idx"] = d.groupby("series_id").cumcount() + 1
        hor_blocks.append(d)
    df_h = pd.concat(hor_blocks, ignore_index=True)
    df_h["h_idx"] = pd.to_numeric(df_h["h_idx"], errors="coerce")
    df_h = df_h.dropna(subset=["h_idx", "y_true", "y_pred"])
    df_h["horizon_group"] = _safe_qcut_3(df_h["h_idx"], ["short_horizon", "medium_horizon", "long_horizon"])
    df_hor = _agg_slice(df_h, "horizon_group")
    df_hor.to_csv(metrics_dir / f"{task.name}_slice_horizon_comparison.csv", index=False)

    # short summary in logs
    chosen_df = dfm[dfm["model"].isin(chosen)].sort_values("rmse")
    best_overall = chosen_df.iloc[0]["model"] if len(chosen_df) else "n/a"
    print(f"\n--- Slice analysis [{task.name}] ---")
    print(f"Лучший по общей RMSE (из 3 ключевых): {best_overall}")
    print(
        "Побед по сериям: "
        + ", ".join([f"{m}={int(wins_count.get(m, 0))}" for m in chosen])
    )
    for nm, dfx, col in [
        ("level", df_level, "level_group"),
        ("volatility", df_vol, "vol_group"),
        ("horizon", df_hor, "horizon_group"),
    ]:
        if len(dfx):
            best_by_slice = (
                dfx.sort_values([col, "rmse"]).groupby(col, as_index=False).first()[[col, "model"]]
            )
            txt = ", ".join([f"{r[col]}={r['model']}" for _, r in best_by_slice.iterrows()])
            print(f"Лучшие по срезам ({nm}): {txt}")

def plot_metrics_dashboard(dfm, title="Специфика: коррелированные ряды", save_path=None, **_kwargs):
    """Строит дашборд и сохраняет в save_path; на экран не выводит (фигура закрывается)."""
    if dfm is None or len(dfm) == 0:
        print("Нет метрик для графиков")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    models = dfm["model"].astype(str).tolist()

    axes[0, 0].bar(models, dfm["rmse"], color="steelblue")
    axes[0, 0].set_title("RMSE (ниже — лучше)")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].bar(models, dfm["wape"], color="darkorange")
    axes[0, 1].set_title("WAPE, % (ниже — лучше)")
    axes[0, 1].tick_params(axis="x", rotation=20)

    axes[1, 0].bar(models, dfm["train_time_sec"], color="seagreen")
    axes[1, 0].set_title("Время обучения, с")
    axes[1, 0].tick_params(axis="x", rotation=20)

    axes[1, 1].bar(models, dfm["inference_time_sec"], color="indianred")
    axes[1, 1].set_title("Время инференса, с")
    axes[1, 1].tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    plt.tight_layout()
    try:
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)


def to_regular_series(part: pd.DataFrame, task: TaskSpec, value_col: str | None = None):
    vc = value_col or task.value_col
    tmp = part[["date_dt", vc]].copy()
    tmp["date_dt"] = pd.to_datetime(tmp["date_dt"], errors="coerce")
    tmp[vc] = pd.to_numeric(tmp[vc], errors="coerce")
    tmp = tmp.dropna(subset=["date_dt", vc])
    # После обновления disease-датасета возможны дубли по неделе: агрегируем безопасно.
    s = tmp.groupby("date_dt", as_index=True)[vc].mean().sort_index()
    return s.asfreq(task.freq).fillna(0)


def test_start_from_df(df: pd.DataFrame, task: TaskSpec):
    max_date = df["date_dt"].max()
    return max_date - pd.DateOffset(years=TEST_HORIZON_YEARS) + pd.Timedelta(days=1)


def compute_val_start(train_df: pd.DataFrame, test_start: pd.Timestamp, task: TaskSpec) -> pd.Timestamp:
    """Граница train_fit | validation: validation заканчивается перед test_start."""
    mn = train_df["date_dt"].min()
    if task.freq == "D":
        cand = test_start - pd.Timedelta(days=VAL_DAYS_BEFORE_TEST)
    else:
        cand = test_start - pd.DateOffset(weeks=VAL_WEEKS_BEFORE_TEST)
    if cand <= mn:
        span = (test_start - mn).total_seconds()
        cand = mn + pd.Timedelta(seconds=max(span * 0.2, 86400 * 7))
    if cand >= test_start:
        cand = mn + (test_start - mn) * 0.7
    return cand


def compute_split_dates_by_ratio(
    df: pd.DataFrame,
    date_col: str,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Возвращает (val_start, test_start) по уникальным датам без утечки во времени."""
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and (train_ratio + val_ratio) < 1.0):
        raise ValueError("Некорректные доли train/val/test")
    dts = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().unique()
    n = len(dts)
    if n < 10:
        raise ValueError("Слишком мало уникальных дат для процентного split")
    i_val = max(1, min(n - 2, int(np.floor(n * train_ratio))))
    i_test = max(i_val + 1, min(n - 1, int(np.floor(n * (train_ratio + val_ratio)))))
    val_start = pd.Timestamp(dts[i_val])
    test_start = pd.Timestamp(dts[i_test])
    return val_start, test_start


def compute_val_start_xgb(df: pd.DataFrame, split_col: str, test_start: pd.Timestamp, task: TaskSpec) -> pd.Timestamp:
    """Граница train_fit | val по той же колонке, что и сплит XGBoost (snapshot_dt / date_dt)."""
    pre = df[df[split_col] < test_start]
    if len(pre) == 0:
        raise ValueError("XGBoost: нет строк до test_start")
    mn = pd.Timestamp(pre[split_col].min())
    if task.freq == "D":
        cand = test_start - pd.Timedelta(days=VAL_DAYS_BEFORE_TEST)
    else:
        cand = test_start - pd.DateOffset(weeks=VAL_WEEKS_BEFORE_TEST)
    if cand <= mn:
        span = (test_start - mn).total_seconds()
        cand = mn + pd.Timedelta(seconds=max(span * 0.2, 86400 * 7))
    if cand >= test_start:
        cand = mn + (test_start - mn) * 0.7
    return cand


def sarima_grid_configs(seasonal_period: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Ограниченный перебор (p,d,q) и (P,D,Q,s), s = сезонность ряда (дневной трафик)."""
    s = seasonal_period
    out: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    for p in (0, 1, 2):
        for q in (0, 1, 2):
            for d in (0, 1):
                for P in (0, 1):
                    for Q in (0, 1):
                        for D in (0, 1):
                            if p == 0 and q == 0 and d == 0:
                                continue
                            out.append(((p, d, q), (P, D, Q, s)))
                            if len(out) >= SARIMA_GRID_MAX:
                                return out
    return out


def sarima_grid_configs_weekly(s: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    """Недельные ряды (заболеваемость): меньше D/P/Q, сильнее (0,0,0,s) — стабильнее MLE."""
    out: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = []
    orders = [
        (1, 0, 0),
        (0, 0, 1),
        (1, 0, 1),
        (2, 0, 0),
        (0, 0, 2),
        (1, 1, 0),
        (0, 1, 1),
        (1, 1, 1),
    ]
    seasonals = [
        (0, 0, 0, s),
        (1, 0, 0, s),
        (0, 0, 1, s),
        (1, 0, 1, s),
    ]
    for order in orders:
        p, d, q = order
        if p == 0 and q == 0 and d == 0:
            continue
        for P, D, Q, ss in seasonals:
            out.append((order, (P, D, Q, ss)))
            if len(out) >= SARIMA_GRID_MAX:
                return out
    return out


def sarima_grid_for_task(task: TaskSpec, seasonal_period: int) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    if task.freq == "W-MON" or task.name == "disease":
        return sarima_grid_configs_weekly(seasonal_period)
    return sarima_grid_configs(seasonal_period)


def _sarima_mle_converged(res) -> bool:
    mr = getattr(res, "mle_retvals", None)
    if not isinstance(mr, dict):
        return True
    c = mr.get("converged")
    return c is None or bool(c)


def _fit_sarima(
    endog: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    *,
    maxiter: int | None = None,
):
    """SARIMAX без exog; устойчивый MLE: несколько optimizers, подавление шумных предупреждений."""
    maxiter = int(maxiter or SARIMA_MAXITER)

    def _spec():
        return SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

    methods = ("lbfgs", "bfgs", "nm")
    last_res = None
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SMConvergenceWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message=".*Maximum Likelihood optimization failed.*",
                    category=Warning,
                )
                res = _spec().fit(disp=False, maxiter=maxiter, method=method)
            last_res = res
            if _sarima_mle_converged(res):
                return res
        except Exception:
            continue
    if last_res is not None:
        return last_res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SMConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        return _spec().fit(disp=False, maxiter=maxiter, method="nm")


def _fit_sarima_fallback(
    endog: pd.Series,
    seasonal_period: int,
    *,
    maxiter: int | None = None,
):
    """Минимальная спецификация при полном отказе сетки."""
    maxiter = int(maxiter or SARIMA_MAXITER)
    for order, seasonal_order in (
        ((0, 1, 1), (0, 0, 0, seasonal_period)),
        ((1, 0, 0), (0, 0, 0, seasonal_period)),
        ((1, 1, 0), (0, 0, 0, seasonal_period)),
    ):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=SMConvergenceWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                return SARIMAX(
                    endog,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=maxiter, method="lbfgs")
        except Exception:
            continue
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SMConvergenceWarning)
        return SARIMAX(
            endog,
            order=(0, 1, 1),
            seasonal_order=(0, 0, 0, seasonal_period),
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=maxiter, method="nm")


def load_correlated_rest_ids(path=CLUSTER_CSV):
    df = pd.read_csv(path)
    if "rest_id" in df.columns:
        ids = df["rest_id"].astype(int).unique().tolist()
    elif "series_id" in df.columns:
        ids = df["series_id"].astype(int).unique().tolist()
    else:
        raise ValueError("В файле кластера нет колонок rest_id/series_id")
    return sorted(ids)


def drop_weather(df):
    cols = [c for c in WEATHER_COLS if c in df.columns]
    return df.drop(columns=cols, errors="ignore")


def load_classic_cluster_split(cluster_ids):
    df = pd.read_parquet(CLASSIC_PARQUET)
    df["date_dt"] = pd.to_datetime(df["date_dt"])
    df = df[df["rest_id"].isin(cluster_ids)].copy()
    df = drop_weather(df)
    df = df.sort_values(["rest_id", "date_dt"]).reset_index(drop=True)
    df["series_id"] = df["rest_id"].astype(str)
    return df


def load_disease_classic_long():
    """Недельный long-формат из prepared sequential parquet: city × age_group -> series."""
    df = pd.read_parquet(DISEASE_SEQUENTIAL_PARQUET).copy()
    required = {"city", "age_group", "target"}
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(
            f"disease_sequential_dataset.parquet: отсутствуют обязательные колонки {miss}"
        )
    if "date_full" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date_full"], errors="coerce")
    elif "date_dt" in df.columns:
        df["date_dt"] = _coerce_year_week_to_datetime(df["date_dt"])
    else:
        raise ValueError("disease_sequential_dataset.parquet: нужна колонка date_full или date_dt")

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["date_dt", "target"])
    df["series_id"] = df["city"].astype(str) + "__" + df["age_group"].astype(str)
    df["rest_id"] = pd.factorize(df["series_id"], sort=True)[0].astype(int)
    df = df.sort_values(["rest_id", "date_dt"]).reset_index(drop=True)
    return df


def _coerce_year_week_to_datetime(s: pd.Series) -> pd.Series:
    """Преобразует date_dt в Timestamp; поддерживает YYYY-WW и обычные datetime-строки."""
    x = s.copy()
    dt = pd.to_datetime(x, errors="coerce")
    mask = dt.isna() & x.notna()
    if bool(mask.any()):
        # Формат YYYY-WW -> YYYY-WW-1 (понедельник недели; %W как в prepare_disease_datasets.py)
        xw = x[mask].astype(str).str.strip()
        xw = xw.str.replace(r"^(\d{4})-(\d{1,2})$", lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-1", regex=True)
        dt2 = pd.to_datetime(xw, format="%Y-%W-%w", errors="coerce")
        dt.loc[mask] = dt2
    return dt


TASK_TRAFFIC = TaskSpec(
    name="traffic",
    spec_key=SPEC_KEY,
    title_ru="Коррелированные ряды: трафик ресторанов",
    freq="D",
    seasonal_periods=7,
    value_col=TARGET_COL,
    xgb_target_col=TARGET_COL,
    xgb_parquet=BOOST_PARQUET,
    ets_min_train=ETS_MIN_TRAIN_DAYS,
    sarima_min_train=SARIMA_MIN_TRAIN_DAYS,
    lstm_seq_len=LSTM_SEQ_LEN,
    lstm_min_train=LSTM_MIN_TRAIN_DAYS,
    mvt_lookback=MVT_LOOKBACK,
    mvt_horizon=MVT_HORIZON,
    filter_xgb_cluster=True,
    artifact_root=SPEC_KEY,
)

TASK_DISEASE = TaskSpec(
    name="disease",
    spec_key="disease",
    title_ru="Заболеваемость: недельные ряды; XGBoost — snapshot (лаги от даты прогноза)",
    freq="W-MON",
    seasonal_periods=52,
    value_col="target",
    xgb_target_col="target",
    xgb_parquet=DISEASE_SNAPSHOT_PARQUET,
    ets_min_train=24,
    sarima_min_train=52,
    lstm_seq_len=12,
    lstm_min_train=24,
    mvt_lookback=52,
    mvt_horizon=1,
    filter_xgb_cluster=False,
    artifact_root=SPEC_KEY,
)


def load_task_data(task: TaskSpec, cluster_ids: list | None = None):
    """
    Возвращает df_full, train_df, test_df, test_start, cluster_ids (int для rest_id).
    Для traffic cluster_ids обязателен; для disease — все города из датасета.
    """
    if task.name == "traffic":
        if not cluster_ids:
            raise ValueError("traffic: нужен cluster_ids")
        df = load_classic_cluster_split(cluster_ids)
    elif task.name == "disease":
        df = load_disease_classic_long()
        cluster_ids = sorted(df["rest_id"].unique().tolist())
    else:
        raise ValueError(f"Неизвестная задача: {task.name}")

    if task.name == "disease":
        # Disease: time split 70/15/15 по уникальным датам.
        _, test_start = compute_split_dates_by_ratio(df, "date_dt", train_ratio=0.70, val_ratio=0.15)
    else:
        test_start = test_start_from_df(df, task)
    train_df = df[df["date_dt"] < test_start].copy()
    test_df = df[df["date_dt"] >= test_start].copy()
    return df, train_df, test_df, test_start, cluster_ids


def run_ets(task: TaskSpec, train_df, test_df, results, all_predictions):
    _, models_dir, preds_dir, _ = paths_for_task(task)
    ets_dir = models_dir / "ets" / task.name
    ets_dir.mkdir(parents=True, exist_ok=True)
    preds_all = []
    t0 = time.perf_counter()
    series_list = test_df["series_id"].unique()
    print(
        f"\n>>> ETS [{task.spec_key}]: ExponentialSmoothing additive seasonal, "
        f"seasonal_periods={task.seasonal_periods}, рядов в тесте: {len(series_list)}"
    )

    for series_id in tqdm(series_list, desc=f"ETS [{task.spec_key}]"):
        tr = train_df[train_df["series_id"] == series_id]
        te = test_df[test_df["series_id"] == series_id]
        if len(tr) < task.ets_min_train or len(te) == 0:
            continue
        train_series = to_regular_series(tr, task)
        test_series = to_regular_series(te, task)
        slug = series_file_slug(series_id)
        model_path = ets_dir / f"{slug}.pkl"
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SMConvergenceWarning)
                    if task.name == "disease":
                        # Для disease упрощаем ETS (без сезонной компоненты), чтобы избежать массовых срывов оптимизации.
                        model = ExponentialSmoothing(
                            train_series,
                            trend=None,
                            seasonal=None,
                            initialization_method="estimated",
                        ).fit(optimized=True, use_brute=False)
                    else:
                        model = ExponentialSmoothing(
                            train_series,
                            trend=None,
                            seasonal="add",
                            seasonal_periods=task.seasonal_periods,
                            initialization_method="estimated",
                        ).fit(optimized=True, use_brute=False)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            p0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SMConvergenceWarning)
                pr = model.forecast(len(test_series))
            pred_time = time.perf_counter() - p0
            temp = te.copy().set_index("date_dt").asfreq(task.freq).reset_index()
            temp["y_true"] = test_series.values
            temp["y_pred"] = pr.values
            temp["model"] = "ets"
            temp["inference_time_sec"] = pred_time / max(len(temp), 1)
            rid = int(tr.iloc[0]["rest_id"])
            temp["rest_id"] = rid
            temp["series_id"] = str(series_id)
            preds_all.append(
                temp[
                    ["rest_id", "series_id", "date_dt", "y_true", "y_pred", "model", "inference_time_sec"]
                ]
            )
        except Exception:
            # Страховка: если ETS не обучился/не предсказал, отдаём наивный прогноз последним train-значением.
            try:
                last_val = float(train_series.iloc[-1]) if len(train_series) else 0.0
                pred_time = 0.0
                temp = te.copy().set_index("date_dt").asfreq(task.freq).reset_index()
                temp["y_true"] = test_series.values
                temp["y_pred"] = np.full(len(test_series), last_val, dtype=float)
                temp["model"] = "ets"
                temp["inference_time_sec"] = pred_time
                rid = int(tr.iloc[0]["rest_id"])
                temp["rest_id"] = rid
                temp["series_id"] = str(series_id)
                preds_all.append(
                    temp[
                        ["rest_id", "series_id", "date_dt", "y_true", "y_pred", "model", "inference_time_sec"]
                    ]
                )
            except Exception:
                continue

    train_time = time.perf_counter() - t0
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    print(
        f"<<< ETS [{task.spec_key}] готово: строк предиктов={len(pred_df)}, "
        f"train+infer {train_time:.1f} с (время в метрике — суммарное по этапу ETS)\n"
    )
    log_result(results, all_predictions, "ETS", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(preds_dir / f"{task.name}_ets_predictions.parquet", index=False)


def run_sarima(
    task: TaskSpec,
    train_fit: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results,
    all_predictions,
):
    """SARIMA без экзогенных регрессоров; подбор порядков по val, финальное обучение на полном train."""
    _, models_dir, preds_dir, _ = paths_for_task(task)
    sdir = models_dir / "sarima" / task.name
    sdir.mkdir(parents=True, exist_ok=True)
    params_cache_path = sdir / "sarima_best_params.json"
    if params_cache_path.exists():
        try:
            with open(params_cache_path, encoding="utf-8") as f:
                params_cache = json.load(f)
        except Exception:
            params_cache = {}
    else:
        params_cache = {}
    params_cache_updated = False
    preds_all = []
    t0 = time.perf_counter()
    series_list = test_df["series_id"].unique()
    m = task.seasonal_periods
    grid = sarima_grid_for_task(task, m)
    mx = SARIMA_MAXITER_DISEASE if task.name == "disease" else SARIMA_MAXITER
    if task.name == "disease":
        mx = min(mx, SARIMA_MAXITER_DISEASE_FAST)
        # Упрощенная SARIMA для disease: фиксированный robust-набор без grid search.
        disease_orders = [((1, 0, 1), (0, 0, 0, m)), ((1, 1, 0), (0, 0, 0, m))]
    else:
        disease_orders = []

    def _fit_predict_one(series_id):
        tr_fit = train_fit[train_fit["series_id"] == series_id]
        te = test_df[test_df["series_id"] == series_id]
        if len(tr_fit) < task.sarima_min_train or len(te) == 0:
            return None
        train_fit_series = to_regular_series(tr_fit, task)
        ts_val = to_regular_series(val_df[val_df["series_id"] == series_id], task) if len(val_df) else None
        tr_full = train_df[train_df["series_id"] == series_id]
        test_series = to_regular_series(te, task)
        slug = series_file_slug(series_id)
        model_path = sdir / f"{slug}.pkl"
        try:
            best_ord: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None
            best_score = float("inf")
            cached = params_cache.get(str(series_id))
            if isinstance(cached, dict) and "order" in cached and "seasonal_order" in cached:
                try:
                    ord_cached = tuple(int(x) for x in cached["order"])
                    seas_cached = tuple(int(x) for x in cached["seasonal_order"])
                    if len(ord_cached) == 3 and len(seas_cached) == 4:
                        best_ord = (ord_cached, seas_cached)
                except Exception:
                    best_ord = None

            if best_ord is None:
                if task.name == "disease":
                    for order, seasonal_order in disease_orders:
                        try:
                            _ = _fit_sarima(train_fit_series, order, seasonal_order, maxiter=mx)
                            best_ord = (order, seasonal_order)
                            break
                        except Exception:
                            continue
                elif ts_val is not None and len(ts_val) > 0:
                    for order, seasonal_order in grid:
                        try:
                            mod = _fit_sarima(
                                train_fit_series, order, seasonal_order, maxiter=mx
                            )
                            fc = mod.forecast(steps=len(ts_val))
                            yv = np.asarray(ts_val.values, dtype=float)
                            fp = np.asarray(fc, dtype=float)
                            if len(fp) != len(yv):
                                continue
                            score = float(np.sqrt(mean_squared_error(yv, fp)))
                            if score < best_score:
                                best_score = score
                                best_ord = (order, seasonal_order)
                        except Exception:
                            continue

            if best_ord is None:
                if task.freq == "W-MON" or task.name == "disease":
                    best_ord = ((1, 0, 1), (0, 0, 0, m))
                else:
                    best_ord = ((1, 1, 1), (1, 0, 1, m))

            train_full_series = to_regular_series(tr_full, task)
            try:
                final_model = _fit_sarima(
                    train_full_series, best_ord[0], best_ord[1], maxiter=mx
                )
            except Exception:
                final_model = _fit_sarima_fallback(train_full_series, m, maxiter=mx)
            with open(model_path, "wb") as f:
                pickle.dump({"model": final_model, "order": best_ord[0], "seasonal_order": best_ord[1]}, f)

            p0 = time.perf_counter()
            pr = final_model.forecast(steps=len(test_series))
            pred_time = time.perf_counter() - p0
            temp = te.copy().set_index("date_dt").asfreq(task.freq).reset_index()
            temp["y_true"] = test_series.values
            temp["y_pred"] = np.asarray(pr)
            temp["model"] = "sarima"
            temp["inference_time_sec"] = pred_time / max(len(temp), 1)
            rid = int(tr_full.iloc[0]["rest_id"])
            temp["rest_id"] = rid
            temp["series_id"] = str(series_id)
            return {
                "pred": temp[
                    ["rest_id", "series_id", "date_dt", "y_true", "y_pred", "model", "inference_time_sec"]
                ],
                "series_id": str(series_id),
                "best_ord": best_ord,
            }
        except Exception:
            return None

    if task.name == "disease":
        max_workers = max(1, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            outputs = list(tqdm(ex.map(_fit_predict_one, series_list), total=len(series_list), desc="SARIMA"))
        for out in outputs:
            if not out:
                continue
            preds_all.append(out["pred"])
            sid = out["series_id"]
            if sid not in params_cache:
                params_cache[sid] = {
                    "order": list(out["best_ord"][0]),
                    "seasonal_order": list(out["best_ord"][1]),
                }
                params_cache_updated = True
    else:
        for series_id in tqdm(series_list, desc="SARIMA"):
            out = _fit_predict_one(series_id)
            if not out:
                continue
            preds_all.append(out["pred"])
            sid = out["series_id"]
            if sid not in params_cache:
                params_cache[sid] = {
                    "order": list(out["best_ord"][0]),
                    "seasonal_order": list(out["best_ord"][1]),
                }
                params_cache_updated = True

    train_time = time.perf_counter() - t0
    if params_cache_updated:
        with open(params_cache_path, "w", encoding="utf-8") as f:
            json.dump(params_cache, f, ensure_ascii=False, indent=2)
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_result(results, all_predictions, "SARIMA", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(preds_dir / f"{task.name}_sarima_predictions.parquet", index=False)


def xgb_drop_columns(df: pd.DataFrame, task: TaskSpec) -> list[str]:
    """Исключаем из X: даты, id, таргет. Не добавляйте в parquet фичи из будущего относительно строки."""
    base = [
        task.xgb_target_col,
        "date_dt",
        "date",
        "rest_id",
        "rest_uk",
        "snapshot_dt",
        "city",
        "age_group",
        "forecast_date",
    ]
    w = [c for c in WEATHER_COLS if c in df.columns]
    return list(dict.fromkeys(base + w))


def run_xgboost(
    task: TaskSpec,
    cluster_ids,
    val_start: pd.Timestamp,
    test_start: pd.Timestamp,
    results,
    all_predictions,
    *,
    xgb_data_parquet: Path | None = None,
    xgb_df: pd.DataFrame | None = None,
    model_label: str = "XGBoost",
    model_file_suffix: str = "",
    predictions_filename: str | None = None,
):
    _, models_dir, preds_dir, _ = paths_for_task(task)
    pred_out = (
        predictions_filename
        if predictions_filename is not None
        else f"{task.name}_xgboost_predictions.parquet"
    )
    t_total = time.perf_counter()
    if xgb_df is not None:
        df = xgb_df.copy()
        pq = "<provided_df>"
    else:
        pq = xgb_data_parquet or task.xgb_parquet
        df = pd.read_parquet(pq)
    if "snapshot_dt" in df.columns:
        df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"])
    if "date_dt" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date_dt"])
    elif "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"])
    elif "forecast_date" in df.columns:
        df["snapshot_dt"] = pd.to_datetime(df["forecast_date"])

    if "snapshot_dt" in df.columns:
        split_col = "snapshot_dt"
    elif "date_dt" in df.columns:
        split_col = "date_dt"
    else:
        raise ValueError(
            f"В {pq} нужны date_dt/date или snapshot_dt (старый формат: forecast_date)"
        )

    if df[split_col].isna().any():
        raise ValueError(
            f"XGBoost: в колонке {split_col} есть NaN — нельзя однозначно разделить train/test."
        )

    if task.filter_xgb_cluster:
        df = df[df["rest_id"].isin(cluster_ids)].copy()
    if "dish_rus_name" in df.columns:
        enc = LabelEncoder()
        df["dish_rus_name_enc"] = enc.fit_transform(df["dish_rus_name"].astype(str))
    if "city" in df.columns and "city_enc" not in df.columns:
        enc_c = LabelEncoder()
        df["city_enc"] = enc_c.fit_transform(df["city"].astype(str))
    if "age_group" in df.columns and "age_group_enc" not in df.columns:
        enc_age = LabelEncoder()
        df["age_group_enc"] = enc_age.fit_transform(df["age_group"].astype(str))

    drop_cols = [c for c in xgb_drop_columns(df, task) if c in df.columns]
    pre_test = df[df[split_col] < test_start].copy()
    train_fit_b = pre_test[pre_test[split_col] < val_start].copy()
    val_b = pre_test[(pre_test[split_col] >= val_start) & (pre_test[split_col] < test_start)].copy()
    test_b = df[df[split_col] >= test_start].copy()

    if len(test_b) == 0:
        print("XGBoost: нет тестовых строк после test_start")
        return

    if len(train_fit_b) < 50 or len(val_b) < 10:
        q = pre_test[split_col].quantile(0.75)
        train_fit_b = pre_test[pre_test[split_col] < q].copy()
        val_b = pre_test[pre_test[split_col] >= q].copy()

    if len(train_fit_b):
        assert bool((train_fit_b[split_col] < test_start).all()), (
            "XGBoost: утечка — train_fit_b содержит даты >= test_start"
        )
    if len(val_b):
        assert bool((val_b[split_col] < test_start).all()), (
            "XGBoost: утечка — val содержит даты >= test_start"
        )
    if len(test_b):
        assert bool((test_b[split_col] >= test_start).all()), (
            "XGBoost: test должен быть только при split_col >= test_start"
        )

    feature_cols = [c for c in train_fit_b.columns if c not in drop_cols]

    def _fill_xy(tb):
        X = tb[feature_cols].copy()
        y = tb[task.xgb_target_col]
        for c in X.columns:
            if X[c].dtype in ("float64", "float32") or X[c].isna().any():
                med = X[c].median()
                X[c] = X[c].fillna(med)
        return X, y

    X_tr, y_tr = _fill_xy(train_fit_b)
    X_va, y_va = _fill_xy(val_b)

    print(
        f"\n>>> XGBoost [{task.spec_key}] «{model_label}»: Optuna подбор гиперпараметров "
        f"({OPTUNA_N_TRIALS} trials), split по «{split_col}»"
    )
    print(
        f"    train_fit={len(train_fit_b)} | val={len(val_b)} | test={len(test_b)} | "
        f"фичей={len(feature_cols)}"
    )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 450),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        }
        m = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        m.fit(X_tr, y_tr)
        pred = m.predict(X_va)
        return float(np.sqrt(mean_squared_error(np.asarray(y_va, dtype=float), pred)))

    def _optuna_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        v = trial.value
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            print(
                f"    Optuna trial {trial.number + 1}/{OPTUNA_N_TRIALS}  val_RMSE={v:.6f}"
            )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        objective,
        n_trials=OPTUNA_N_TRIALS,
        show_progress_bar=False,
        callbacks=[_optuna_callback],
    )

    best_params = study.best_params
    print(f"    лучшие параметры: {best_params}")
    print(f"    лучший val_RMSE={study.best_value:.6f} → финальное обучение на train+val...")
    train_all = pd.concat([train_fit_b, val_b], ignore_index=True)
    X_all, y_all = _fill_xy(train_all)
    X_test, y_test = _fill_xy(test_b)

    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_all, y_all)
    sfx = f"_{model_file_suffix}" if model_file_suffix else ""
    path_model = models_dir / f"xgboost_{task.name}{sfx}.pkl"
    with open(path_model, "wb") as f:
        pickle.dump(model, f)

    train_time = time.perf_counter() - t_total

    p0 = time.perf_counter()
    y_hat = model.predict(X_test)
    inf_time = time.perf_counter() - p0
    print(
        f"<<< XGBoost «{model_label}» готово: train+optuna+fit {train_time:.1f} с, "
        f"инференс тест {inf_time:.2f} с, строк предиктов={len(test_b)}\n"
    )

    if "rest_id" in test_b.columns:
        pred_df = test_b[["rest_id", "date_dt"]].copy()
        pred_df["series_id"] = pred_df["rest_id"].astype(str)
    else:
        _pcols = ["date_dt"]
        if "snapshot_dt" in test_b.columns:
            _pcols = ["snapshot_dt", "date_dt"]
        pred_df = test_b[_pcols].copy()
        if "city" in test_b.columns:
            pred_df["series_id"] = test_b["city"].astype(str).values
            pred_df["rest_id"] = pd.factorize(pred_df["series_id"])[0]
        else:
            pred_df["series_id"] = "na"
            pred_df["rest_id"] = 0
        if "horizon_weeks" in test_b.columns:
            pred_df["horizon_weeks"] = test_b["horizon_weeks"].values
        if "horizon_week" in test_b.columns:
            pred_df["horizon_week"] = test_b["horizon_week"].values

    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_hat
    pred_df["model"] = model_label.lower().replace(" ", "-")
    pred_df["inference_time_sec"] = inf_time / max(len(pred_df), 1)
    pred_df.to_parquet(preds_dir / pred_out, index=False)

    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_result(results, all_predictions, model_label, pred_df, train_time, infer_sum)


class LSTMReg(nn.Module):
    def __init__(self, hidden: int = LSTM_HIDDEN, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            1,
            hidden,
            layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_windows(series_norm, seq_len: int):
    x_list, y_list = [], []
    arr = series_norm.values.astype(np.float32)
    for t in range(seq_len, len(arr)):
        x_list.append(arr[t - seq_len : t].reshape(seq_len, 1))
        y_list.append(arr[t])
    return x_list, y_list


def _lstm_pooled_train_val_split(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    task: TaskSpec,
    seq_len: int,
):
    """Окна по train (target до val_start) и validation (target с val_start)."""
    X_tr, y_tr = [], []
    X_va, y_va = [], []
    series_stats: dict = {}
    vs = pd.Timestamp(val_start)
    for series_id in train_df["series_id"].unique():
        tr_all = train_df[train_df["series_id"] == series_id]
        if len(tr_all) < task.lstm_min_train:
            continue
        full_series = to_regular_series(tr_all, task)
        if len(full_series) <= seq_len:
            continue
        mu = float(full_series.mean())
        sig = float(full_series.std()) + 1e-6
        z = ((full_series - mu) / sig).values.astype(np.float32)
        idx_vs = int(full_series.index.searchsorted(vs))
        if idx_vs == 0:
            continue
        series_stats[series_id] = {"mu": mu, "sig": sig}
        for t in range(seq_len, len(z)):
            win = z[t - seq_len : t].reshape(seq_len, 1)
            if t < idx_vs:
                X_tr.append(win)
                y_tr.append(z[t])
            else:
                X_va.append(win)
                y_va.append(z[t])

    if not X_tr:
        return None, None, None, None, series_stats

    if not X_va:
        n = len(X_tr)
        split = max(1, int(n * (1.0 - LSTM_VAL_FRAC)))
        X_va = X_tr[split:]
        y_va = y_tr[split:]
        X_tr = X_tr[:split]
        y_tr = y_tr[:split]

    X_tr_arr = np.stack(X_tr, axis=0).astype(np.float32)
    y_tr_arr = np.asarray(y_tr, dtype=np.float32).reshape(-1, 1)
    X_va_arr = np.stack(X_va, axis=0).astype(np.float32)
    y_va_arr = np.asarray(y_va, dtype=np.float32).reshape(-1, 1)
    return X_tr_arr, y_tr_arr, X_va_arr, y_va_arr, series_stats


def _lstm_pooled_all_train(train_df: pd.DataFrame, task: TaskSpec, seq_len: int):
    """Все окна перед test (полный train для финального переобучения)."""
    X_all, y_all = [], []
    series_stats: dict = {}
    for series_id in train_df["series_id"].unique():
        tr_all = train_df[train_df["series_id"] == series_id]
        if len(tr_all) < task.lstm_min_train:
            continue
        full_series = to_regular_series(tr_all, task)
        if len(full_series) <= seq_len:
            continue
        mu = float(full_series.mean())
        sig = float(full_series.std()) + 1e-6
        z = ((full_series - mu) / sig).values.astype(np.float32)
        series_stats[series_id] = {"mu": mu, "sig": sig}
        for t in range(seq_len, len(z)):
            X_all.append(z[t - seq_len : t].reshape(seq_len, 1))
            y_all.append(z[t])

    if not X_all:
        return None, None, series_stats

    X_arr = np.stack(X_all, axis=0).astype(np.float32)
    y_arr = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)
    return X_arr, y_arr, series_stats


def _lstm_train_with_es(
    model: LSTMReg,
    loader: torch.utils.data.DataLoader,
    X_va: torch.Tensor,
    y_va: torch.Tensor,
    device: str,
) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.L1Loss()
    best_val = float("inf")
    best_state: dict | None = None
    bad = 0
    y_va_np = y_va.numpy() if not y_va.is_cuda else y_va.cpu().numpy()
    for _ep in range(LSTM_EPOCHS_MAX):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if LSTM_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred_v = model(X_va.to(device))
            val_rmse = float(
                np.sqrt(mean_squared_error(y_va_np, pred_v.cpu().numpy()))
            )
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= LSTM_ES_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def run_lstm(
    task: TaskSpec,
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    test_df: pd.DataFrame,
    results,
    all_predictions,
    *,
    retrain: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    seq_len = task.lstm_seq_len
    _, models_dir, preds_dir, _ = paths_for_task(task)
    path_pt = models_dir / f"lstm_{task.name}.pt"
    path_cfg = models_dir / f"lstm_{task.name}_config.json"

    if not retrain:
        if not path_pt.is_file() or not path_cfg.is_file():
            print(
                f"LSTM [{task.spec_key}]: RETRAIN=False, но нет {path_pt.name} и/или {path_cfg.name} — пропуск."
            )
            return
        with open(path_cfg, encoding="utf-8") as f:
            loaded_cfg = json.load(f)
        full = _lstm_pooled_all_train(train_df, task, seq_len)
        X_all, y_all, series_stats = full
        if X_all is None or len(series_stats) == 0:
            print("LSTM: нет окон для инференса")
            return
        final_model = LSTMReg(
            hidden=int(loaded_cfg["hidden"]),
            layers=int(loaded_cfg["layers"]),
            dropout=float(loaded_cfg["dropout"]),
        ).to(device)
        final_model.load_state_dict(torch.load(path_pt, map_location=device))
        train_time = 0.0
    else:
        t_total = time.perf_counter()
        split = _lstm_pooled_train_val_split(train_df, val_start, task, seq_len)
        X_tr, y_tr, X_va, y_va, series_stats_hp = split
        if X_tr is None or len(series_stats_hp) == 0:
            print("LSTM: нет окон для обучения")
            return

        X_va_t = torch.from_numpy(X_va)
        y_va_t = torch.from_numpy(y_va)

        print(
            f"\n>>> LSTM [{task.spec_key}]: подбор из {len(LSTM_CONFIGS)} конфигураций, "
            f"device={device}, seq_len={seq_len}"
        )
        print(
            f"    окна train={len(X_tr)} | val={len(X_va)} | рядов в пуле={len(series_stats_hp)}"
        )

        best_cfg: dict | None = None
        best_val_rmse = float("inf")

        for i, cfg in enumerate(LSTM_CONFIGS, start=1):
            hid = int(cfg["hidden"])
            lay = int(cfg["layers"])
            dr = float(cfg["dropout"])
            print(
                f"    ▶ конфиг {i}/{len(LSTM_CONFIGS)}: hidden={hid}, layers={lay}, dropout={dr} "
                f"(early stopping, max {LSTM_EPOCHS_MAX} ep)"
            )
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_tr), torch.from_numpy(y_tr)
            )
            loader = torch.utils.data.DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)
            model = LSTMReg(
                hidden=hid,
                layers=lay,
                dropout=dr,
            ).to(device)
            val_rmse = _lstm_train_with_es(model, loader, X_va_t, y_va_t, device)
            print(f"      → val_RMSE={val_rmse:.6f}")
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_cfg = dict(cfg)

        if best_cfg is None:
            print("LSTM: не удалось выбрать конфигурацию")
            return

        print(
            f"    выбрана лучшая по val: hidden={int(best_cfg['hidden'])}, layers={int(best_cfg['layers'])}, "
            f"dropout={float(best_cfg['dropout'])} (val_RMSE={best_val_rmse:.6f})"
        )
        print("    финальное обучение на полном train (train+val окна)...")

        full = _lstm_pooled_all_train(train_df, task, seq_len)
        X_all, y_all, series_stats = full
        if X_all is None:
            print("LSTM: нет окон для финального обучения")
            return

        n = len(X_all)
        if n >= 2:
            X_tr_f, X_va_f, y_tr_f, y_va_f = train_test_split(
                X_all,
                y_all,
                test_size=LSTM_VAL_FRAC,
                random_state=RANDOM_SEED,
                shuffle=True,
            )
        else:
            X_tr_f, y_tr_f = X_all, y_all
            X_va_f, y_va_f = X_all, y_all
        ds_f = torch.utils.data.TensorDataset(
            torch.from_numpy(X_tr_f), torch.from_numpy(y_tr_f)
        )
        loader_f = torch.utils.data.DataLoader(ds_f, batch_size=LSTM_BATCH, shuffle=True)

        final_model = LSTMReg(
            hidden=int(best_cfg["hidden"]),
            layers=int(best_cfg["layers"]),
            dropout=float(best_cfg["dropout"]),
        ).to(device)
        _ = _lstm_train_with_es(
            final_model,
            loader_f,
            torch.from_numpy(X_va_f),
            torch.from_numpy(y_va_f),
            device,
        )

        torch.save(final_model.state_dict(), path_pt)
        cfg_out = {
            "hidden": int(best_cfg["hidden"]),
            "layers": int(best_cfg["layers"]),
            "dropout": float(best_cfg["dropout"]),
        }
        with open(path_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg_out, f, indent=2, ensure_ascii=False)
        train_time = time.perf_counter() - t_total
        print(
            f"<<< LSTM [{task.spec_key}] обучение завершено за {train_time:.1f} с → инференс по тесту\n"
        )

    if not retrain:
        print(
            f"LSTM [{task.spec_key}]: загружены веса из {path_pt.name}, инференс без переобучения\n"
        )

    final_model.eval()
    rows = []
    infer_total = 0.0
    with torch.no_grad():
        for series_id in tqdm(test_df["series_id"].unique(), desc=f"LSTM infer [{task.spec_key}]"):
            if series_id not in series_stats:
                continue
            tr = train_df[train_df["series_id"] == series_id]
            te = test_df[test_df["series_id"] == series_id]
            if len(te) == 0:
                continue
            mu = series_stats[series_id]["mu"]
            sig = series_stats[series_id]["sig"]
            train_series = to_regular_series(tr, task)
            test_series = to_regular_series(te, task)
            z_hist = ((train_series - mu) / sig).values.astype(np.float32).tolist()
            p0 = time.perf_counter()
            preds = []
            for _ in range(len(test_series)):
                win = np.array(z_hist[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
                zn = (
                    final_model(torch.from_numpy(win).to(device))
                    .cpu()
                    .numpy()
                    .reshape(-1)[0]
                )
                preds.append(zn)
                z_hist.append(float(zn))
            dt = time.perf_counter() - p0
            infer_total += dt

            temp = pd.DataFrame(
                {
                    "date_dt": test_series.index,
                    "y_true": test_series.values,
                    "y_pred": np.array(preds) * sig + mu,
                }
            )
            temp["rest_id"] = int(tr.iloc[0]["rest_id"])
            temp["series_id"] = str(series_id)
            temp["model"] = "lstm"
            temp["inference_time_sec"] = dt / max(len(temp), 1)
            rows.append(temp)

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(pred_df) == 0:
        print("LSTM: нет прогнозов")
        return
    pred_df.to_parquet(preds_dir / f"{task.name}_lstm_predictions.parquet", index=False)
    log_result(results, all_predictions, "LSTM", pred_df, train_time, infer_total)


# ═══════════════════════ Temporal Fusion Transformer (compact) ═══════════════════════

def _tft_known_from_dates(dti: pd.DatetimeIndex) -> np.ndarray:
    w = dti.isocalendar().week.astype(int).to_numpy()
    wd = dti.weekday.to_numpy()
    mo = dti.month.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * w / 52.0),
            np.cos(2 * np.pi * w / 52.0),
            wd.astype(np.float32) / 6.0,
            (mo.astype(np.float32) - 1.0) / 11.0,
        ]
    ).astype(np.float32)


def _tft_static_cols(task: TaskSpec, df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    if task.name == "disease":
        for c in ("Population", "Density", "Latitude", "Longitude", "Area"):
            if c in df.columns:
                cols.append(c)
        if "age_group" in df.columns:
            cols.append("age_group")
    else:
        # Для traffic минимум: идентификатор ряда как статический признак.
        cols.append("rest_id")
    return cols


def _tft_series_static_map(df: pd.DataFrame, static_cols: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    df_u = df.sort_values(["series_id", "date_dt"])
    if "age_group" in static_cols:
        enc_age = LabelEncoder()
        df_u = df_u.copy()
        df_u["age_group"] = enc_age.fit_transform(df_u["age_group"].astype(str))
    for sid, grp in df_u.groupby("series_id"):
        row = grp.iloc[0]
        vals = []
        for c in static_cols:
            vals.append(float(pd.to_numeric(row[c], errors="coerce")) if c in row else 0.0)
        out[str(sid)] = np.array(vals, dtype=np.float32)
    return out


class TFTReg(nn.Module):
    def __init__(self, obs_dim: int, known_dim: int, static_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, hidden)
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.known_proj = nn.Linear(known_dim, hidden)
        self.lstm = nn.LSTM(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=float(dropout) if layers > 1 else 0.0,
        )
        n_heads = 4 if hidden >= 64 else 2
        self.attn = nn.MultiheadAttention(hidden, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, past_obs: torch.Tensor, known_fut: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        xo = self.obs_proj(past_obs)
        out, _ = self.lstm(xo)
        q = self.known_proj(known_fut).unsqueeze(1)
        att, _ = self.attn(q, out, out)
        s = self.static_proj(static_x)
        z = torch.cat([out[:, -1, :], att.squeeze(1), s], dim=-1)
        h = self.fuse(z)
        return self.head(h)


def _tft_make_windows(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    task: TaskSpec,
    lookback: int,
):
    static_cols = _tft_static_cols(task, train_df)
    static_map = _tft_series_static_map(train_df, static_cols)
    X_tr_p, X_tr_k, X_tr_s, y_tr = [], [], [], []
    X_va_p, X_va_k, X_va_s, y_va = [], [], [], []
    series_stats: dict[str, dict[str, float | np.ndarray]] = {}
    vs = pd.Timestamp(val_start)

    for series_id in train_df["series_id"].unique():
        tr = train_df[train_df["series_id"] == series_id]
        if len(tr) < max(task.lstm_min_train, lookback + 1):
            continue
        s = to_regular_series(tr, task)
        if len(s) <= lookback:
            continue
        mu = float(s.mean())
        sig = float(s.std()) + 1e-6
        z = ((s - mu) / sig).values.astype(np.float32)
        # Для disease при наличии готовых календарных признаков используем их из датасета.
        if task.name == "disease" and all(c in tr.columns for c in ("week_sin", "week_cos", "month_sin", "month_cos")):
            ksrc = (
                tr.drop_duplicates("date_dt")
                .set_index("date_dt")[["week_sin", "week_cos", "month_sin", "month_cos"]]
                .reindex(s.index)
                .astype(np.float32)
            )
            known_all = ksrc.fillna(0.0).values.astype(np.float32)
        else:
            known_all = _tft_known_from_dates(s.index)
        sid = str(series_id)
        svec = static_map.get(sid)
        if svec is None:
            continue
        series_stats[sid] = {"mu": mu, "sig": sig, "svec": svec}
        for t in range(lookback, len(z)):
            win = z[t - lookback : t].reshape(lookback, 1)
            kf = known_all[t]
            yt = z[t]
            if s.index[t] < vs:
                X_tr_p.append(win)
                X_tr_k.append(kf)
                X_tr_s.append(svec)
                y_tr.append(yt)
            else:
                X_va_p.append(win)
                X_va_k.append(kf)
                X_va_s.append(svec)
                y_va.append(yt)

    if not X_tr_p:
        return None
    if not X_va_p:
        n = len(X_tr_p)
        split = max(1, int(n * (1.0 - TFT_VAL_FRAC)))
        X_va_p, X_va_k, X_va_s, y_va = X_tr_p[split:], X_tr_k[split:], X_tr_s[split:], y_tr[split:]
        X_tr_p, X_tr_k, X_tr_s, y_tr = X_tr_p[:split], X_tr_k[:split], X_tr_s[:split], y_tr[:split]

    return {
        "X_tr_p": np.stack(X_tr_p).astype(np.float32),
        "X_tr_k": np.stack(X_tr_k).astype(np.float32),
        "X_tr_s": np.stack(X_tr_s).astype(np.float32),
        "y_tr": np.asarray(y_tr, dtype=np.float32).reshape(-1, 1),
        "X_va_p": np.stack(X_va_p).astype(np.float32),
        "X_va_k": np.stack(X_va_k).astype(np.float32),
        "X_va_s": np.stack(X_va_s).astype(np.float32),
        "y_va": np.asarray(y_va, dtype=np.float32).reshape(-1, 1),
        "series_stats": series_stats,
        "static_dim": len(static_cols),
    }


def _tft_train_with_es(model: TFTReg, loader, Xv_p, Xv_k, Xv_s, yv, device: str) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=TFT_LR, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    best_val = float("inf")
    best_state = None
    bad = 0
    yv_np = yv.numpy()
    for _ in range(TFT_EPOCHS_MAX):
        model.train()
        for pb, kb, sb, yb in loader:
            pb, kb, sb, yb = pb.to(device), kb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(pb, kb, sb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if TFT_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), TFT_GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv_p.to(device), Xv_k.to(device), Xv_s.to(device)).cpu().numpy()
            vrmse = float(np.sqrt(mean_squared_error(yv_np, pv)))
        if vrmse < best_val:
            best_val = vrmse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= TFT_ES_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def run_tft(
    task: TaskSpec,
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    test_df: pd.DataFrame,
    results,
    all_predictions,
    *,
    retrain: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, models_dir, preds_dir, _ = paths_for_task(task)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    lookback = max(TFT_LOOKBACK, task.lstm_seq_len)

    packed = _tft_make_windows(train_df, val_start, task, lookback)
    if packed is None:
        print("TFT: нет окон для обучения")
        return
    X_tr_p = packed["X_tr_p"]
    X_tr_k = packed["X_tr_k"]
    X_tr_s = packed["X_tr_s"]
    y_tr = packed["y_tr"]
    X_va_p = torch.from_numpy(packed["X_va_p"])
    X_va_k = torch.from_numpy(packed["X_va_k"])
    X_va_s = torch.from_numpy(packed["X_va_s"])
    y_va = torch.from_numpy(packed["y_va"])
    series_stats = packed["series_stats"]
    static_dim = int(packed["static_dim"])

    print(
        f"\n>>> TFT [{task.spec_key}]: {len(TFT_CONFIGS)} конфигурации, device={device}, lookback={lookback}"
    )

    for cfg in TFT_CONFIGS:
        name = str(cfg["name"])
        hidden = int(cfg["hidden"])
        layers = int(cfg["layers"])
        dropout = float(cfg["dropout"])
        slug = name.lower()
        model_path = models_dir / f"tft_{task.name}_{slug}.pt"
        t0 = time.perf_counter()

        model = TFTReg(
            obs_dim=1,
            known_dim=4,
            static_dim=static_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
        ).to(device)
        if retrain:
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(X_tr_p),
                torch.from_numpy(X_tr_k),
                torch.from_numpy(X_tr_s),
                torch.from_numpy(y_tr),
            )
            loader = torch.utils.data.DataLoader(ds, batch_size=TFT_BATCH, shuffle=True)
            val_rmse = _tft_train_with_es(model, loader, X_va_p, X_va_k, X_va_s, y_va, device)
            torch.save(model.state_dict(), model_path)
            print(f"    {name}: val_RMSE={val_rmse:.6f}")
        else:
            if not model_path.is_file():
                print(f"    {name}: нет весов {model_path.name}, пропуск.")
                continue
            model.load_state_dict(torch.load(model_path, map_location=device))

        train_time = time.perf_counter() - t0 if retrain else 0.0
        model.eval()
        rows = []
        infer_total = 0.0
        with torch.no_grad():
            for sid in test_df["series_id"].unique():
                sid_s = str(sid)
                if sid_s not in series_stats:
                    continue
                tr = train_df[train_df["series_id"] == sid]
                te = test_df[test_df["series_id"] == sid]
                if len(te) == 0:
                    continue
                st = series_stats[sid_s]
                mu = float(st["mu"])
                sig = float(st["sig"])
                svec = np.asarray(st["svec"], dtype=np.float32).reshape(1, -1)
                train_s = to_regular_series(tr, task)
                test_s = to_regular_series(te, task)
                z_hist = ((train_s - mu) / sig).values.astype(np.float32).tolist()
                p0 = time.perf_counter()
                preds = []
                for dt in test_s.index:
                    win = np.asarray(z_hist[-lookback:], dtype=np.float32).reshape(1, lookback, 1)
                    kf = _tft_known_from_dates(pd.DatetimeIndex([dt])).reshape(1, 4)
                    zn = (
                        model(
                            torch.from_numpy(win).to(device),
                            torch.from_numpy(kf).to(device),
                            torch.from_numpy(svec).to(device),
                        )
                        .cpu()
                        .numpy()
                        .reshape(-1)[0]
                    )
                    preds.append(float(zn))
                    z_hist.append(float(zn))
                dt_inf = time.perf_counter() - p0
                infer_total += dt_inf
                tmp = pd.DataFrame(
                    {
                        "date_dt": test_s.index,
                        "y_true": test_s.values,
                        "y_pred": np.asarray(preds, dtype=np.float32) * sig + mu,
                    }
                )
                tmp["rest_id"] = int(tr.iloc[0]["rest_id"])
                tmp["series_id"] = sid_s
                tmp["model"] = name
                tmp["inference_time_sec"] = dt_inf / max(len(tmp), 1)
                rows.append(tmp)

        pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(pred_df) == 0:
            print(f"    {name}: нет прогнозов")
            continue
        pred_df.to_parquet(preds_dir / f"{task.name}_{slug}_predictions.parquet", index=False)
        log_result(results, all_predictions, name, pred_df, train_time, infer_total)


# ═══════════════════════ Multivariate Cross-Series Transformer ═══════════════════════

def _default_cal_from_index(all_dates: pd.DatetimeIndex) -> np.ndarray:
    dd = all_dates
    cal_df = pd.DataFrame(
        {
            "weekday_num": dd.weekday.astype(np.float32),
            "month_num": dd.month.astype(np.float32),
            "day_num": dd.day.astype(np.float32),
            "holiday_flg": np.zeros(len(dd), dtype=np.float32),
            "weekend_day": (dd.weekday >= 5).astype(np.float32),
            "working_day": (dd.weekday < 5).astype(np.float32),
        },
        index=all_dates,
    )
    return cal_df[list(MVT_CAL_COLS)].values.astype(np.float32)


def _build_mv_panel(df, cluster_ids, test_start, task: TaskSpec):
    """Панель target [T,S] + календарь; сетка task.freq.
    Нормализация target и cal только по train (до train_end); окна обучения не используют test.
    Календарь: без bfill из будущего (только ffill + 0), чтобы не подмешивать значения с теста."""
    val_col = task.value_col
    freq = task.freq
    df = df.sort_values(["rest_id", "date_dt"]).copy()
    all_dates = pd.date_range(df["date_dt"].min(), df["date_dt"].max(), freq=freq)
    ordered_ids = sorted(cluster_ids)
    S = len(ordered_ids)
    T = len(all_dates)

    target_mat = np.zeros((T, S), dtype=np.float32)
    for j, rid in enumerate(ordered_ids):
        grp = df[df["rest_id"] == rid].set_index("date_dt")[val_col].reindex(all_dates).fillna(0)
        target_mat[:, j] = grp.values.astype(np.float32)

    ts = pd.Timestamp(test_start)
    train_end = int((all_dates < ts).sum())
    mu = target_mat[:train_end].mean(axis=0, keepdims=True)
    sig = target_mat[:train_end].std(axis=0, keepdims=True) + 1e-6
    target_z = ((target_mat - mu) / sig).astype(np.float32)

    cal_cols_present = [c for c in MVT_CAL_COLS if c in df.columns]
    if len(cal_cols_present) == len(MVT_CAL_COLS):
        n_cal = len(MVT_CAL_COLS)
        cal_mat = np.zeros((T, n_cal), dtype=np.float32)
        ref = df.drop_duplicates("date_dt").sort_values("date_dt").set_index("date_dt")
        ref = ref.reindex(all_dates)
        for j, c in enumerate(MVT_CAL_COLS):
            cal_mat[:, j] = ref[c].ffill().fillna(0).values.astype(np.float32)
        cal_mu = cal_mat[:train_end].mean(axis=0, keepdims=True)
        cal_sig = cal_mat[:train_end].std(axis=0, keepdims=True)
        cal_sig[cal_sig < 1e-6] = 1.0
        cal_mat = ((cal_mat - cal_mu) / cal_sig).astype(np.float32)
    else:
        cal_mat = _default_cal_from_index(all_dates)
        n_cal = cal_mat.shape[1]
        cal_mu = cal_mat[:train_end].mean(axis=0, keepdims=True)
        cal_sig = cal_mat[:train_end].std(axis=0, keepdims=True)
        cal_sig[cal_sig < 1e-6] = 1.0
        cal_mat = ((cal_mat - cal_mu) / cal_sig).astype(np.float32)

    return {
        "dates": all_dates,
        "ids": ordered_ids,
        "target": target_mat,
        "target_z": target_z,
        "cal": cal_mat,
        "mu": mu.flatten(),
        "sig": sig.flatten(),
        "train_end": train_end,
        "n_cal": n_cal,
    }


class MVGroupDataset(torch.utils.data.Dataset):
    """Each sample: full group snapshot at a time window.
    Returns past_x [T,S,F], past_cal [T,n_cal], future_cal [H,n_cal], target_y [H,S]."""

    def __init__(self, panel, windows, lookback, horizon):
        self.p = panel
        self.wins = windows
        self.L = lookback
        self.H = horizon

    def __len__(self):
        return len(self.wins)

    def __getitem__(self, idx):
        t = self.wins[idx]
        pe = t + self.L
        fe = pe + self.H
        past_z = self.p["target_z"][t:pe]          # [L,S]
        future_y = self.p["target_z"][pe:fe]        # [H,S]
        past_cal = self.p["cal"][t:pe]              # [L,n_cal]
        future_cal = self.p["cal"][pe:fe]           # [H,n_cal]
        return (
            torch.from_numpy(past_z),
            torch.from_numpy(past_cal),
            torch.from_numpy(future_cal),
            torch.from_numpy(future_y),
        )


def _make_mv_windows(panel, lookback: int, horizon: int):
    mx = panel["train_end"] - lookback - horizon
    if mx < 0:
        return []
    return list(range(0, mx + 1))


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2 + d % 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def _mvt_encoder_layer(d: int, heads: int, ff: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d,
        nhead=heads,
        dim_feedforward=ff,
        dropout=dropout,
        batch_first=True,
        activation="gelu",
        norm_first=True,
    )


class TemporalConvStem(nn.Module):
    """Локальный контекст по времени (до глобального self-attention)."""

    def __init__(self, d: int, kernel: int = 5, drop: float = 0.05):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(d, d, kernel, padding=pad, groups=1),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.net(y)
        y = y.transpose(1, 2)
        return x + y


class TemporalBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout, n_layers):
        super().__init__()
        layer = _mvt_encoder_layer(d, heads, ff, dropout)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(x)


class CrossSeriesBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout, n_layers):
        super().__init__()
        layer = _mvt_encoder_layer(d, heads, ff, dropout)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x):
        return self.enc(x)


class MVCrossSeriesTransformer(nn.Module):
    def __init__(
        self,
        n_series,
        n_cal,
        lookback=MVT_LOOKBACK,
        horizon=MVT_HORIZON,
        d=MVT_HIDDEN,
        ff=MVT_FF_DIM,
        heads_t=MVT_HEADS_TIME,
        heads_s=MVT_HEADS_SERIES,
        n_temp=MVT_N_TEMP_LAYERS,
        n_cross=MVT_N_SERIES_LAYERS,
        dropout=MVT_DROPOUT,
    ):
        super().__init__()
        self.n_series = n_series
        self.lookback = lookback
        self.horizon = horizon
        self.d = d

        self.series_emb = nn.Embedding(n_series, d)
        self.input_proj = nn.Linear(1 + n_cal, d)
        self.temporal_stem = TemporalConvStem(d, kernel=5, drop=min(0.1, dropout + 0.02))
        self.pos_enc = PositionalEncoding(d, max_len=lookback + horizon + 64)

        self.temporal = TemporalBlock(d, heads_t, ff, dropout, n_temp)
        self.cross = CrossSeriesBlock(d, heads_s, ff, dropout, n_cross)

        self.fusion_norm = nn.LayerNorm(d)
        self.fusion_ff = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d),
        )
        self.fusion_norm2 = nn.LayerNorm(d)
        self.ctx_fuse = nn.Linear(2 * d, d)

        self.future_proj = nn.Linear(n_cal, d)
        self.dec_temporal = TemporalBlock(d, heads_t, ff, dropout, 1)
        self.dec_cross = CrossSeriesBlock(d, heads_s, ff, dropout, 1)

        self.head = nn.Linear(d, 1)

    def forward(self, past_z, past_cal, future_cal):
        B, T, S = past_z.shape
        H = future_cal.size(1)

        cal_exp = past_cal.unsqueeze(2).expand(B, T, S, -1)
        x = torch.cat([past_z.unsqueeze(-1), cal_exp], dim=-1)
        x = self.input_proj(x)

        se = self.series_emb(
            torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        )
        x = x + se.unsqueeze(1)

        x_flat = x.permute(0, 2, 1, 3).reshape(B * S, T, self.d)
        x_flat = self.temporal_stem(x_flat)
        x_flat = self.pos_enc(x_flat)
        x_flat = self.temporal(x_flat)
        x = x_flat.reshape(B, S, T, self.d).permute(0, 2, 1, 3)

        x_flat = x.reshape(B * T, S, self.d)
        x_flat = self.cross(x_flat)
        x = x_flat.reshape(B, T, S, self.d)

        x = self.fusion_norm(x)
        x = x + self.fusion_ff(x)
        x = self.fusion_norm2(x)

        ctx = self.ctx_fuse(torch.cat([x[:, -1, :, :], x.mean(dim=1)], dim=-1))

        fc_exp = future_cal.unsqueeze(2).expand(B, H, S, -1)
        dec = self.future_proj(fc_exp)
        dec = dec + se.unsqueeze(1) + ctx.unsqueeze(1)

        dec_flat = dec.permute(0, 2, 1, 3).reshape(B * S, H, self.d)
        dec_flat = self.pos_enc(dec_flat)
        dec_flat = self.dec_temporal(dec_flat)
        dec = dec_flat.reshape(B, S, H, self.d).permute(0, 2, 1, 3)

        dec_flat = dec.reshape(B * H, S, self.d)
        dec_flat = self.dec_cross(dec_flat)
        dec = dec_flat.reshape(B, H, S, self.d)

        out = self.head(dec).squeeze(-1)
        return out


def _train_mv_transformer(
    model,
    tr_loader,
    device,
    *,
    run_label: str = "",
    verbose: bool = True,
) -> float:
    """Обучение только по train-окнам; OneCycleLR (разогрев + косинус), AdamW с weight decay."""
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=MVT_LR,
        weight_decay=MVT_WEIGHT_DECAY,
    )
    loss_fn = nn.SmoothL1Loss()
    steps_per_epoch = max(len(tr_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=MVT_LR,
        epochs=MVT_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=MVT_ONECYCLE_PCT_START,
        div_factor=25.0,
        final_div_factor=1e4,
    )
    t0 = time.perf_counter()
    prefix = f"{run_label} " if run_label else ""

    for ep in range(1, MVT_EPOCHS + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for pz, pc, fc, y in tr_loader:
            pz, pc, fc, y = pz.to(device), pc.to(device), fc.to(device), y.to(device)
            pred = model(pz, pc, fc)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            if MVT_GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), MVT_GRAD_CLIP)
            opt.step()
            scheduler.step()
            bs = pz.size(0)
            tr_sum += loss.item() * bs
            tr_n += bs
        tr_loss = tr_sum / max(tr_n, 1)
        if verbose and (ep % 5 == 0 or ep == 1 or ep == MVT_EPOCHS):
            lr_cur = opt.param_groups[0]["lr"]
            print(
                f"  {prefix}MVT ep {ep:3d}/{MVT_EPOCHS} train={tr_loss:.5f} lr={lr_cur:.2e}"
            )

    return time.perf_counter() - t0


def _infer_mv_transformer(model, panel, device, lookback: int, horizon: int):
    model.to(device)
    model.eval()
    L = lookback
    H = horizon
    T_total = len(panel["dates"])
    te = panel["train_end"]
    ids = panel["ids"]
    mu = panel["mu"]
    sig = panel["sig"]
    S = len(ids)

    target_z = panel["target_z"].copy()
    cal = panel["cal"]
    dates = panel["dates"]
    target_raw = panel["target"]

    rows = []
    infer_total = 0.0
    pos = te

    with torch.no_grad():
        while pos < T_total:
            ah = min(H, T_total - pos)
            start = max(0, pos - L)
            actual_L = pos - start
            pz = target_z[start:pos]
            if actual_L < L:
                pad = np.zeros((L - actual_L, S), dtype=np.float32)
                pz = np.vstack([pad, pz])
                pc_raw = cal[start:pos]
                pc = np.vstack([np.zeros((L - actual_L, cal.shape[1]), dtype=np.float32), pc_raw])
            else:
                pc = cal[start:pos]

            if pos + H <= T_total:
                fc = cal[pos : pos + H]
            else:
                fc = np.zeros((H, cal.shape[1]), dtype=np.float32)
                fc[:ah] = cal[pos : pos + ah]

            b_pz = torch.from_numpy(pz).unsqueeze(0).to(device)
            b_pc = torch.from_numpy(pc).unsqueeze(0).to(device)
            b_fc = torch.from_numpy(fc).unsqueeze(0).to(device)

            t1 = time.perf_counter()
            pred_z = model(b_pz, b_pc, b_fc)[0].cpu().numpy()
            dt = time.perf_counter() - t1
            infer_total += dt

            pred_z = pred_z[:ah]
            target_z[pos : pos + ah] = pred_z

            y_pred = pred_z * sig + mu
            y_true = target_raw[pos : pos + ah]
            step_dates = dates[pos : pos + ah]

            for j, rid in enumerate(ids):
                temp = pd.DataFrame({
                    "date_dt": step_dates,
                    "y_true": y_true[:, j],
                    "y_pred": y_pred[:, j],
                })
                temp["rest_id"] = rid
                temp["series_id"] = str(rid)
                temp["model"] = "mv_transformer"
                temp["inference_time_sec"] = dt / max(ah * S, 1)
                rows.append(temp)
            pos += H

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pred_df, infer_total


def run_mv_transformer(
    task: TaskSpec,
    df_full,
    cluster_ids,
    test_start,
    results,
    all_predictions,
    *,
    retrain: bool = True,
):
    device = MVT_DEVICE
    H = task.mvt_horizon
    _, models_dir, preds_dir, metrics_dir = paths_for_task(task)

    if task.name == "traffic":
        config_list = MVT_CONFIGS_TRAFFIC
    elif task.name == "disease":
        config_list = MVT_CONFIGS_DISEASE
    else:
        print(f"MVT: задача {task.name!r} без набора конфигов — пропуск.")
        return

    mode = "обучение + инференс" if retrain else "только инференс (веса из папки)"
    print(f"\n--- MV Cross-Series Transformer [{task.spec_key}] ({mode}, device: {device}) ---")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    panel = _build_mv_panel(df_full, cluster_ids, test_start, task)
    S = len(panel["ids"])
    n_cal = panel["n_cal"]
    print(f"Рядов: {S}, дат: {len(panel['dates'])}, train_end idx: {panel['train_end']}")
    print(f"Calendar features: {n_cal}, horizon (фикс.): {H}")

    dl_kw = dict(num_workers=0, pin_memory=(device == "cuda"))
    search_rows: list[dict] = []
    mvt_logged: list[tuple[str, pd.DataFrame, float]] = []
    train_time_total = 0.0

    for cfg in config_list:
        cfg_name = str(cfg["name"])
        L = int(cfg["lookback"])
        d_model = int(cfg["hidden"])
        ff_dim = int(cfg["ff_dim"])
        n_temp = int(cfg["n_temp_layers"])
        n_cross = int(cfg["n_series_layers"])
        drop = float(cfg["dropout"])
        file_slug = cfg_name.lower().replace(" ", "_")
        path_ckpt = models_dir / f"mv_transformer_{task.name}_{file_slug}.pt"

        windows = _make_mv_windows(panel, L, H)
        if retrain:
            if len(windows) < 1:
                print(f"MVT [{cfg_name}]: нет train окон (L={L}), пропуск.")
                continue
        elif not path_ckpt.is_file():
            print(
                f"MVT [{cfg_name}]: RETRAIN=False, нет файла {path_ckpt.name} — пропуск "
                f"(положите веса или выставьте RETRAIN=True)."
            )
            continue

        if retrain:
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(windows)
            print(
                f"MVT [{cfg_name}] L={L} d={d_model} ff={ff_dim} "
                f"n_temp={n_temp} n_cross={n_cross} drop={drop} | "
                f"окна (все на обучение, без val): {len(windows)}"
            )

            tr_ds = MVGroupDataset(panel, windows, L, H)
            tr_loader = torch.utils.data.DataLoader(
                tr_ds, batch_size=MVT_BATCH, shuffle=True, **dl_kw
            )

            model = MVCrossSeriesTransformer(
                n_series=S,
                n_cal=n_cal,
                lookback=L,
                horizon=H,
                d=d_model,
                ff=ff_dim,
                n_temp=n_temp,
                n_cross=n_cross,
                dropout=drop,
            )
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  параметров: {n_params:,}")

            train_t = _train_mv_transformer(
                model,
                tr_loader,
                device,
                run_label=f"[{cfg_name}]",
                verbose=True,
            )
            train_time_total += train_t
        else:
            print(
                f"MVT [{cfg_name}] L={L} d={d_model}: загрузка {path_ckpt.name} (без обучения)"
            )
            model = MVCrossSeriesTransformer(
                n_series=S,
                n_cal=n_cal,
                lookback=L,
                horizon=H,
                d=d_model,
                ff=ff_dim,
                n_temp=n_temp,
                n_cross=n_cross,
                dropout=drop,
            )
            model.load_state_dict(torch.load(path_ckpt, map_location=device))

        pred_df, infer_t = _infer_mv_transformer(model, panel, device, L, H)
        if len(pred_df) == 0:
            print(f"MVT [{cfg_name}]: нет прогнозов на тесте.")
            continue

        pred_df["model"] = f"mv_transformer_{file_slug}"
        if retrain:
            torch.save(model.state_dict(), path_ckpt)
        pred_df.to_parquet(
            preds_dir / f"{task.name}_mv_transformer_{file_slug}_predictions.parquet",
            index=False,
        )

        test_rmse = rmse(pred_df["y_true"], pred_df["y_pred"])
        test_wape = wape(pred_df["y_true"], pred_df["y_pred"])
        search_rows.append(
            {
                "task": task.spec_key,
                "config_name": cfg_name,
                "rmse": test_rmse,
                "wape": test_wape,
                "train_time_total_sec": train_time_total if retrain else 0.0,
                "inference_time_sec": infer_t,
            }
        )
        mvt_logged.append((cfg_name, pred_df, infer_t))

    if not search_rows:
        print(
            "MVT: ни одна конфигурация не дала результатов "
            f"({'обучение' if retrain else 'инференс'})."
        )
        return

    df_search = pd.DataFrame(search_rows)
    df_search.to_csv(
        metrics_dir / f"{task.name}_mv_transformer_configs.csv",
        index=False,
    )
    print("\nMVT — все конфигурации (test, без выбора по val):")
    print(df_search.to_string(index=False))

    for cfg_name, pred_df, infer_t in mvt_logged:
        log_result(
            results,
            all_predictions,
            f"MV-Transformer-{cfg_name}",
            pred_df,
            train_time_total,
            infer_t,
        )

    if retrain:
        print(
            f"\nMVT: суммарное время обучения всех конфигов = {train_time_total:.1f} с "
            f"(одинаковое в таблице метрик для каждой архитектуры)"
        )


def run_task_experiment(
    task: TaskSpec,
    cluster_ids: list | None = None,
    do_ets=True,
    do_sarima=True,
    do_xgb=True,
    do_lstm=True,
    do_tft=True,
    do_mv_transformer=True,
    show_plot=True,
    retrain: bool | None = None,
    **kwargs,
):
    """Аргумент show_plot оставлен для совместимости; график метрик только сохраняется в файл."""
    _ = show_plot
    if "do_sarimax" in kwargs:
        do_sarima = bool(kwargs.pop("do_sarimax"))
    if kwargs:
        raise TypeError(f"run_task_experiment: неизвестные аргументы {set(kwargs)!r}")

    _retrain = RETRAIN if retrain is None else retrain

    setup_dirs(task)
    results = []
    all_predictions = []

    if task.name == "traffic" and cluster_ids is None:
        cluster_ids = load_correlated_rest_ids(CLUSTER_CSV)
    df_full, train_df, test_df, test_start, cluster_ids = load_task_data(task, cluster_ids)

    print("=" * 60)
    print(task.title_ru)
    if task.name == "traffic":
        print(f"Кластер: {CLUSTER_CSV}")
    print(f"Сущностей (рядов): {len(cluster_ids)}")
    print(f"Train (классика): {train_df.shape} | Test: {test_df.shape}")
    print(f"Test start (последний год): {test_start.date()}")
    print("=" * 60)

    if len(cluster_ids) == 0 or len(test_df) == 0:
        print("Нет данных для прогона — остановка.")
        return pd.DataFrame(), []

    if task.name == "disease":
        val_start, test_start = compute_split_dates_by_ratio(
            df_full, "date_dt", train_ratio=0.70, val_ratio=0.15
        )
        train_df = df_full[df_full["date_dt"] < test_start].copy()  # 85%
        test_df = df_full[df_full["date_dt"] >= test_start].copy()  # 15%
    else:
        val_start = compute_val_start(train_df, test_start, task)
    train_fit_df = train_df[train_df["date_dt"] < val_start].copy()
    val_df = train_df[train_df["date_dt"] >= val_start].copy()
    if len(train_fit_df) == 0 or len(val_df) == 0:
        q = train_df["date_dt"].quantile(0.75)
        val_start = pd.Timestamp(q)
        train_fit_df = train_df[train_df["date_dt"] < val_start].copy()
        val_df = train_df[train_df["date_dt"] >= val_start].copy()
    print(
        f"Validation (классика date_dt): val_start={val_start.date()} | "
        f"train_fit: {train_fit_df['date_dt'].min().date()}..{train_fit_df['date_dt'].max().date()} | "
        f"val: {val_df['date_dt'].min().date()}..{val_df['date_dt'].max().date()}"
    )

    if do_ets:
        run_ets(task, train_df, test_df, results, all_predictions)
    if do_sarima:
        run_sarima(
            task,
            train_fit_df,
            val_df,
            train_df,
            test_df,
            results,
            all_predictions,
        )
    if do_xgb:
        if task.name == "disease":
            if not task.xgb_parquet.exists():
                print(f"Пропуск XGBoost: нет файла {task.xgb_parquet}")
            else:
                df_x = pd.read_parquet(task.xgb_parquet)
                if "snapshot_dt" not in df_x.columns:
                    raise ValueError(
                        f"В {task.xgb_parquet} нужна колонка snapshot_dt для split"
                    )
                df_x["snapshot_dt"] = pd.to_datetime(df_x["snapshot_dt"])
                val_xgb, test_xgb = compute_split_dates_by_ratio(
                    df_x, "snapshot_dt", train_ratio=0.70, val_ratio=0.15
                )
                train_snap = df_x[df_x["snapshot_dt"] < test_xgb].copy()
                test_snap = df_x[df_x["snapshot_dt"] >= test_xgb].copy()
                if "date_dt" not in train_snap.columns or "date_dt" not in test_snap.columns:
                    print("XGBoost disease: отсутствует date_dt в snapshot-датасете, безопасный split невозможен, пропуск.")
                else:
                    train_snap["date_dt"] = _coerce_year_week_to_datetime(train_snap["date_dt"])
                    test_snap["date_dt"] = _coerce_year_week_to_datetime(test_snap["date_dt"])
                    test_dates = set(test_snap["date_dt"].dropna().unique().tolist())
                    before_n = len(train_snap)
                    train_snap = train_snap[~train_snap["date_dt"].isin(test_dates)].copy()
                    cut_n = before_n - len(train_snap)
                    if cut_n > 0:
                        print(
                            f"XGBoost disease: удалено {cut_n} train-строк по anti-leakage date_dt (даты тестового snapshot)."
                        )
                    if len(train_snap) == 0 or len(test_snap) == 0:
                        print("XGBoost disease: после split нет train/test строк, пропуск.")
                    else:
                        df_x_single = pd.concat([train_snap, test_snap], ignore_index=True)
                        print(
                            f"XGBoost disease: split by snapshot_dt | "
                            f"val_start={pd.Timestamp(val_xgb).date()} | "
                            f"test_start={pd.Timestamp(test_xgb).date()} | "
                            f"train_snap={len(train_snap)} | test_snap={len(test_snap)} | "
                            f"test snapshots={int(test_snap['snapshot_dt'].nunique())}"
                        )
                        run_xgboost(
                            task,
                            cluster_ids,
                            val_xgb,
                            pd.Timestamp(test_xgb),
                            results,
                            all_predictions,
                            xgb_df=df_x_single,
                        )
        else:
            # Traffic: one-snapshot логика (если parquet snapshot-формата).
            df_x = pd.read_parquet(task.xgb_parquet)
            if "snapshot_dt" in df_x.columns:
                df_x["snapshot_dt"] = pd.to_datetime(df_x["snapshot_dt"])
                snap_anchor = df_x.loc[df_x["snapshot_dt"] < test_start, "snapshot_dt"].max()
                if pd.isna(snap_anchor):
                    print("XGBoost traffic: нет snapshot_dt перед началом тестового периода, пропуск.")
                else:
                    train_snap = df_x[df_x["snapshot_dt"] < snap_anchor].copy()
                    test_snap = df_x[df_x["snapshot_dt"] == snap_anchor].copy()
                    if "date_dt" in train_snap.columns and "date_dt" in test_snap.columns:
                        train_snap["date_dt"] = pd.to_datetime(train_snap["date_dt"])
                        test_snap["date_dt"] = pd.to_datetime(test_snap["date_dt"])
                        test_dates = set(test_snap["date_dt"].dropna().unique().tolist())
                        before_n = len(train_snap)
                        train_snap = train_snap[~train_snap["date_dt"].isin(test_dates)].copy()
                        cut_n = before_n - len(train_snap)
                        if cut_n > 0:
                            print(
                                f"XGBoost traffic: удалено {cut_n} train-строк по anti-leakage date_dt (даты тестового snapshot)."
                            )
                    if len(train_snap) == 0 or len(test_snap) == 0:
                        print("XGBoost traffic: после one-snapshot split нет train/test строк, пропуск.")
                    else:
                        val_xgb = compute_val_start_xgb(train_snap, "snapshot_dt", snap_anchor, task)
                        df_x_single = pd.concat([train_snap, test_snap], ignore_index=True)
                        print(
                            f"XGBoost traffic: anchor_snapshot={pd.Timestamp(snap_anchor).date()} | "
                            f"train_snap={len(train_snap)} | test_snap={len(test_snap)} | "
                            f"val_start={pd.Timestamp(val_xgb).date()}"
                        )
                        run_xgboost(
                            task,
                            cluster_ids,
                            val_xgb,
                            pd.Timestamp(snap_anchor),
                            results,
                            all_predictions,
                            xgb_df=df_x_single,
                        )
            else:
                run_xgboost(task, cluster_ids, val_start, test_start, results, all_predictions)
    if do_lstm:
        run_lstm(
            task, train_df, val_start, test_df, results, all_predictions, retrain=_retrain
        )
    if do_tft:
        run_tft(
            task, train_df, val_start, test_df, results, all_predictions, retrain=_retrain
        )
    if do_mv_transformer:
        run_mv_transformer(
            task,
            df_full,
            cluster_ids,
            test_start,
            results,
            all_predictions,
            retrain=_retrain,
        )

    dfm = pd.DataFrame(results)
    _, _, _, metrics_dir = paths_for_task(task)
    if len(results):
        dfm = save_metrics_files(task, results)
        _run_key_model_slice_analysis(task, train_df, results, all_predictions)
        fig_path = metrics_dir / f"{task.name}_metrics_plot.png"
        plot_metrics_dashboard(
            dfm,
            title=f"{task.title_ru} — RMSE, WAPE, время",
            save_path=fig_path,
        )
        print(f"\n=== Сводка метрик ({task.name} / {task.spec_key}) ===")
        print(dfm.to_string(index=False))
        print(
            "\nФайлы:",
            metrics_dir / f"{task.name}_metrics.csv",
            "| график (без показа на экране):",
            fig_path,
        )
    else:
        print(f"\n({task.name}: нет метрик для сохранения)")

    return dfm, all_predictions


def run_correlated_experiment(**kwargs):
    """Обратная совместимость: только трафик."""
    return run_task_experiment(TASK_TRAFFIC, **kwargs)


def run_all_tasks(
    do_ets=True,
    do_sarima=True,
    do_xgb=True,
    do_lstm=True,
    do_tft=True,
    do_mv_transformer=True,
    show_plot=True,
    retrain: bool | None = None,
    **kwargs,
):
    """Трафик + заболеваемость; графики и метрики в artifacts/correlated/metrics/."""
    if "do_sarimax" in kwargs:
        do_sarima = bool(kwargs.pop("do_sarimax"))
    if kwargs:
        raise TypeError(f"run_all_tasks: неизвестные аргументы {set(kwargs)!r}")

    _retrain = RETRAIN if retrain is None else retrain

    out = []
    task_plan: list[tuple[TaskSpec, dict]] = []
    if ENABLE_TASK_TRAFFIC:
        task_plan.append((TASK_TRAFFIC, {}))
    if ENABLE_TASK_DISEASE:
        task_plan.append((TASK_DISEASE, {}))
    if not task_plan:
        print("run_all_tasks: все задачи отключены (ENABLE_TASK_TRAFFIC/ENABLE_TASK_DISEASE=False).")
        return out
    for task, kw_over in task_plan:
        if task.name == "disease" and not task.xgb_parquet.exists():
            print(
                f"Предупреждение ({task.spec_key}): нет snapshot parquet {task.xgb_parquet} — XGBoost snapshot пропущен"
            )
        try:
            kw = dict(
                do_ets=do_ets,
                do_sarima=do_sarima,
                do_xgb=do_xgb,
                do_lstm=do_lstm,
                do_tft=do_tft,
                do_mv_transformer=do_mv_transformer,
                show_plot=show_plot,
                retrain=_retrain,
            )
            kw.update(kw_over)
            dfm, _ = run_task_experiment(task, **kw)
            out.append((task, dfm))
        except FileNotFoundError as e:
            print(f"Пропуск {task.spec_key}: {e}")
    return out


if __name__ == "__main__":
    run_all_tasks(retrain=RETRAIN)

