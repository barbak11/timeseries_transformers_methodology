# -*- coding: utf-8 -*-
"""
Базовый прогон для специфики «разреженные данные»: ETS, SARIMAX, XGBoost, LSTM, PatchTST.
Метрики: MAE (по неокруглённым предсказаниям) и accuracy класса «0 / не 0»
(после округления предсказаний до целых, неотрицательных).

В ноутбуке:  from nir_sparse_baseline import run_sparse_experiment
            run_sparse_experiment()

CLI:  python nir_sparse_baseline.py

Артефакты: artifacts/<SPEC_KEY>/models|predictions|metrics
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tqdm.auto import tqdm
from xgboost import XGBRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from functions.metrics import (
    mae,
    round_pred_counts,
    accuracy_zero_nonzero,
    precision_zero_nonzero,
)

# --- настройки (для других специфик поменяйте SPEC_KEY и пути к parquet) ---
SC = cfg.SPARSE_CONFIG
SPEC_KEY = SC["SPEC_KEY"]
DISEASE_SPEC_KEY = SC["DISEASE_SPEC_KEY"]
RANDOM_SEED = cfg.RANDOM_SEED
DATA_DIR = cfg.DATA_DIR
CLASSIC_PARQUET = cfg.SPARSE_CLASSIC_PARQUET
BOOST_PARQUET = cfg.SPARSE_BOOST_PARQUET
PCR_CLASSIC_PARQUET = cfg.PCR_CLASSIC_PARQUET
PCR_BOOST_PARQUET = cfg.PCR_BOOST_PARQUET

# Какие задачи запускать при прямом запуске файла
RUN_SPARSE_TASK = SC["RUN_SPARSE_TASK"]
RUN_DISEASE_TASK = SC["RUN_DISEASE_TASK"]

_artifacts = cfg.artifact_dirs(SPEC_KEY)
ARTIFACTS_DIR = _artifacts["base"]
MODELS_DIR = _artifacts["models"]
PREDS_DIR = _artifacts["predictions"]
METRICS_DIR = _artifacts["metrics"]

# LSTM
SEQ_LEN = cfg.LSTM_SEQ_LEN
LSTM_HIDDEN = cfg.LSTM_HIDDEN
LSTM_EPOCHS = SC["LSTM_EPOCHS"]
LSTM_BATCH = cfg.LSTM_BATCH
LSTM_LR = cfg.LSTM_LR
LSTM_GRAD_CLIP = cfg.LSTM_GRAD_CLIP
PATCHTST_SEQ_LEN = SC["PATCHTST_SEQ_LEN"]
PATCHTST_PATCH = SC["PATCHTST_PATCH"]
PATCHTST_D_MODEL = SC["PATCHTST_D_MODEL"]
PATCHTST_HEADS = SC["PATCHTST_HEADS"]
PATCHTST_LAYERS = SC["PATCHTST_LAYERS"]
PATCHTST_EPOCHS = SC["PATCHTST_EPOCHS"]
PATCHTST_BATCH = SC["PATCHTST_BATCH"]
PATCHTST_LR = SC["PATCHTST_LR"]
PATCHTST_CONFIGS = [
    {
        **c,
        "epochs": PATCHTST_EPOCHS,
        "batch_size": PATCHTST_BATCH,
        "lr": PATCHTST_LR,
    }
    for c in SC["PATCHTST_CONFIGS"]
]


def setup_dirs():
    """Создать каталоги артефактов для sparse-специфики."""
    for d in (MODELS_DIR, PREDS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def setup_dirs_for(models_dir, preds_dir, metrics_dir):
    """Создать каталоги артефактов

    Args:
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
        metrics_dir: Каталог метрик.
    """
    for d in (models_dir, preds_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)


def to_daily_series(part, value_col="cnt_dishes"):
    """Привести часть ряда к дневной частоте, заполнив пропуски нулями.

    Args:
        part: DataFrame с колонками `date_dt` и целевым столбцом.
        value_col: Имя целевого столбца.

    Returns:
        Pandas Series на дневной сетке.
    """
    return part.set_index("date_dt")[value_col].asfreq("D").fillna(0)


def log_model_result(results, all_predictions, model_name, pred_df, train_sec, infer_sec):
    """Добавить строку результатов модели и сохранить предсказания в список.

    Args:
        results: Список словарей с метриками.
        all_predictions: Список DataFrame с предсказаниями.
        model_name: Человекочитаемое имя модели.
        pred_df: Таблица предсказаний с колонками `y_true`, `y_pred`.
        train_sec: Время обучения (сек).
        infer_sec: Время инференса (сек).
    """
    if pred_df is None or len(pred_df) == 0:
        return
    results.append(
        {
            "model": model_name,
            "mae": mae(pred_df["y_true"], pred_df["y_pred"]),
            "accuracy": accuracy_zero_nonzero(pred_df["y_true"], pred_df["y_pred"]),
            "precision": precision_zero_nonzero(pred_df["y_true"], pred_df["y_pred"]),
            "train_time_sec": float(train_sec),
            "inference_time_sec": float(infer_sec),
        }
    )
    all_predictions.append(pred_df)


def save_final_test_predictions(pred_df, preds_dir, file_stem):
    """Сохранить финальные тестовые предсказания с округлением до целых.

    Args:
        pred_df: Таблица предсказаний.
        preds_dir: Каталог для сохранения.
        file_stem: Префикс имени файла.
    """
    if pred_df is None or len(pred_df) == 0:
        return
    out = pred_df.copy()
    out["y_pred_final"] = round_pred_counts(out["y_pred"])
    out.to_parquet(preds_dir / f"{file_stem}_final_test_predictions.parquet", index=False)


def save_metrics_files(results, spec_key=SPEC_KEY, metrics_dir=METRICS_DIR):
    """Сохранить метрики в CSV и JSON.

    Args:
        results: Список словарей метрик.
        spec_key: Префикс имени файла метрик.
        metrics_dir: Каталог для сохранения.

    Returns:
        DataFrame с метриками.
    """
    dfm = pd.DataFrame(results)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / f"{spec_key}_metrics.csv"
    json_path = metrics_dir / f"{spec_key}_metrics.json"
    dfm.to_csv(csv_path, index=False)
    dfm.to_json(json_path, orient="records", force_ascii=False, indent=2)
    return dfm


def plot_metrics_dashboard(dfm, title="Специфика: разреженные данные", save_path=None, show=True):
    """Построить простой дашборд метрик (MAE/Accuracy/Precision).

    Args:
        dfm: Таблица метрик.
        title: Заголовок графика.
        save_path: Путь для сохранения изображения.
        show: Показать график.
    """
    if dfm is None or len(dfm) == 0:
        print("Нет метрик для графиков")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    models = dfm["model"].astype(str).tolist()

    axes[0].bar(models, dfm["mae"], color="steelblue")
    axes[0].set_title("MAE (ниже — лучше)")
    axes[0].tick_params(axis="x", rotation=20)

    acc_pct = dfm["accuracy"].astype(float) * 100
    axes[1].bar(models, acc_pct, color="darkorange")
    axes[1].set_title("Accuracy 0/≠0, % (выше — лучше)")
    axes[1].tick_params(axis="x", rotation=20)

    prec_pct = dfm["precision"].astype(float) * 100 if "precision" in dfm.columns else np.zeros(len(dfm))
    axes[2].bar(models, prec_pct, color="mediumpurple")
    axes[2].set_title("Precision (класс ≠0), %")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def load_classic_train_test():
    """Загрузить sparse classic dataset и выполнить time split train/test.

    Returns:
        (train_df, test_df, test_start).
    """
    df = pd.read_parquet(CLASSIC_PARQUET)
    df = df.sort_values(["rest_id", "dish_rus_name", "date_dt"]).reset_index(drop=True)
    df["series_id"] = df["rest_id"].astype(str) + "__" + df["dish_rus_name"].astype(str)

    max_date = df["date_dt"].max()
    test_start = max_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    train_df = df[df["date_dt"] < test_start].copy()
    test_df = df[df["date_dt"] >= test_start].copy()
    return train_df, test_df, test_start


def run_ets(train_df, test_df, results, all_predictions):
    """Запустить ETS по каждому ряду и сохранить предсказания.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
    """
    ets_train = train_df[
        ["rest_id", "dish_rus_name", "series_id", "date_dt", "cnt_dishes"]
    ].copy()
    ets_test = test_df[
        ["rest_id", "dish_rus_name", "series_id", "date_dt", "cnt_dishes"]
    ].copy()

    ets_dir = MODELS_DIR / "ets"
    ets_dir.mkdir(parents=True, exist_ok=True)

    preds_all = []
    t0 = time.perf_counter()
    series_list = ets_test["series_id"].unique()

    for i, series_id in enumerate(series_list):
        if i % 500 == 0:
            print(f"ETS: рядов {i} / {len(series_list)}")

        train_part = ets_train[ets_train["series_id"] == series_id].copy()
        test_part = ets_test[ets_test["series_id"] == series_id].copy()
        if len(train_part) < 30 or len(test_part) == 0:
            continue

        train_series = to_daily_series(train_part)
        test_series = to_daily_series(test_part)
        safe_id = series_id.replace("/", "_")
        model_path = ets_dir / f"{safe_id}.pkl"

        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = ExponentialSmoothing(
                    train_series,
                    trend=None,
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit()
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            p0 = time.perf_counter()
            pr = model.forecast(len(test_series))
            pred_time = time.perf_counter() - p0

            temp = test_part.copy().set_index("date_dt").asfreq("D").reset_index()
            temp["series_id"] = series_id
            temp["y_true"] = test_series.values
            temp["y_pred"] = pr.values
            temp["model"] = "ets"
            n = max(len(temp), 1)
            temp["inference_time_sec"] = pred_time / n
            preds_all.append(
                temp[
                    [
                        "rest_id",
                        "dish_rus_name",
                        "series_id",
                        "date_dt",
                        "y_true",
                        "y_pred",
                        "model",
                        "inference_time_sec",
                    ]
                ]
            )
        except Exception:
            continue

    train_time = time.perf_counter() - t0
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_model_result(results, all_predictions, "ETS", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(PREDS_DIR / "ets_predictions.parquet", index=False)
        save_final_test_predictions(pred_df, PREDS_DIR, "ets")
        print(
            "ETS готово — MAE:",
            results[-1]["mae"],
            "accuracy:",
            f"{results[-1]['accuracy']*100:.2f}%",
        )
    else:
        print("ETS: нет прогнозов")


def run_sarimax(train_df, test_df, results, all_predictions):
    """Запустить SARIMAX по каждому ряду и сохранить предсказания.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
    """
    sdir = MODELS_DIR / "sarimax"
    sdir.mkdir(parents=True, exist_ok=True)

    preds_all = []
    t0 = time.perf_counter()
    series_list = test_df["series_id"].unique()

    for series_id in tqdm(series_list, desc="SARIMAX"):
        train_part = train_df[train_df["series_id"] == series_id].copy()
        test_part = test_df[test_df["series_id"] == series_id].copy()
        if len(train_part) < 40 or len(test_part) == 0:
            continue

        train_series = to_daily_series(train_part)
        test_series = to_daily_series(test_part)
        safe_id = series_id.replace("/", "_")
        model_path = sdir / f"{safe_id}.pkl"

        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = SARIMAX(
                    train_series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            p0 = time.perf_counter()
            pr = model.forecast(steps=len(test_series))
            pred_time = time.perf_counter() - p0

            temp = test_part.copy().set_index("date_dt").asfreq("D").reset_index()
            temp["series_id"] = series_id
            temp["y_true"] = test_series.values
            temp["y_pred"] = np.asarray(pr)
            temp["model"] = "sarimax"
            n = max(len(temp), 1)
            temp["inference_time_sec"] = pred_time / n
            preds_all.append(
                temp[
                    [
                        "rest_id",
                        "dish_rus_name",
                        "series_id",
                        "date_dt",
                        "y_true",
                        "y_pred",
                        "model",
                        "inference_time_sec",
                    ]
                ]
            )
        except Exception:
            continue

    train_time = time.perf_counter() - t0
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_model_result(results, all_predictions, "SARIMAX", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(PREDS_DIR / "sarimax_predictions.parquet", index=False)
        save_final_test_predictions(pred_df, PREDS_DIR, "sarimax")
        print(
            "SARIMAX готово — MAE:",
            results[-1]["mae"],
            "accuracy:",
            f"{results[-1]['accuracy']*100:.2f}%",
        )
    else:
        print("SARIMAX: нет прогнозов")


def run_xgboost(results, all_predictions):
    """Обучить и применить XGBoost на sparse boosting-таблице.

    Args:
        results: Список метрик.
        all_predictions: Список предсказаний.
    """
    df = pd.read_parquet(BOOST_PARQUET)
    enc = LabelEncoder()
    df["dish_rus_name_enc"] = enc.fit_transform(df["dish_rus_name"].astype(str))

    max_date = df["date_dt"].max()
    test_start = max_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    train_b = df[df["date_dt"] < test_start].copy()
    test_b = df[df["date_dt"] >= test_start].copy()

    drop_cols = [
        "dish_rus_name",
        "cnt_dishes",
        "date_dt",
        "rest_id",
        "rest_uk",
        "snapshot_dt",
    ]
    X_train = train_b.drop(drop_cols, axis=1)
    y_train = train_b["cnt_dishes"]
    X_test = test_b.drop(drop_cols, axis=1)
    y_test = test_b["cnt_dishes"]

    path_model = MODELS_DIR / "xgboost_model.pkl"
    t0 = time.perf_counter()
    if path_model.exists():
        with open(path_model, "rb") as f:
            model = pickle.load(f)
    else:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        with open(path_model, "wb") as f:
            pickle.dump(model, f)
    train_time = time.perf_counter() - t0

    p0 = time.perf_counter()
    y_hat = model.predict(X_test)
    inf_time = time.perf_counter() - p0

    pred_df = test_b[["rest_id", "dish_rus_name", "date_dt"]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_hat
    pred_df["model"] = "xgboost"
    pred_df["inference_time_sec"] = inf_time / max(len(pred_df), 1)

    pred_df.to_parquet(PREDS_DIR / "xgboost_predictions.parquet", index=False)
    save_final_test_predictions(pred_df, PREDS_DIR, "xgboost")
    log_model_result(results, all_predictions, "XGBoost", pred_df, train_time, inf_time)
    print(
        "XGBoost готово — MAE:",
        results[-1]["mae"],
        "accuracy:",
        f"{results[-1]['accuracy']*100:.2f}%",
    )


class LSTMReg(nn.Module):
    """Простая LSTM-регрессия для одномерного ряда."""

    def __init__(self, hidden=LSTM_HIDDEN, layers=1):
        """Инициализировать LSTM-регрессор.

        Args:
            hidden: Размер скрытого состояния.
            layers: Число LSTM-слоёв.
        """
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        """Прямой проход модели.

        Args:
            x: Окна входа shape (B, T, 1).

        Returns:
            Прогноз shape (B, 1).
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _build_windows(train_series_norm, seq_len):
    """Построить окна (lookback, target) для одномерного ряда.

    Args:
        train_series_norm: Нормированный ряд (Series).
        seq_len: Длина окна.

    Returns:
        (x_list, y_list).
    """
    x_list, y_list = [], []
    arr = train_series_norm.values.astype(np.float32)
    for t in range(seq_len, len(arr)):
        x_list.append(arr[t - seq_len : t].reshape(seq_len, 1))
        y_list.append(arr[t])
    return x_list, y_list


def run_lstm(classic_train_df, classic_test_df, results, all_predictions):
    """Обучить LSTM на pooled-окнах и выполнить прогноз по тесту.

    Args:
        classic_train_df: Train-таблица classic.
        classic_test_df: Test-таблица classic.
        results: Список метрик.
        all_predictions: Список предсказаний.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("LSTM device:", device)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    Xw, yw = [], []
    series_stats = {}

    for series_id in classic_test_df["series_id"].unique():
        tr = classic_train_df[classic_train_df["series_id"] == series_id]
        te = classic_test_df[classic_test_df["series_id"] == series_id]
        if len(tr) < 30 or len(te) == 0:
            continue
        train_series = to_daily_series(tr)
        if len(train_series) <= SEQ_LEN:
            continue
        mu = float(train_series.mean())
        sig = float(train_series.std()) + 1e-6
        z = (train_series - mu) / sig
        xs, ys = _build_windows(z, SEQ_LEN)
        Xw.extend(xs)
        yw.extend(ys)
        series_stats[series_id] = {"mu": mu, "sig": sig}

    if not Xw:
        print("LSTM: нет окон — пропуск")
        return

    X_arr = np.stack(Xw, axis=0).astype(np.float32)
    y_arr = np.asarray(yw, dtype=np.float32).reshape(-1, 1)

    tensor_x = torch.from_numpy(X_arr).to(device)
    tensor_y = torch.from_numpy(y_arr).to(device)
    ds = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

    model = LSTMReg().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    # L1 в z-пространстве ближе к оптимизации MAE на масштабах ряда, чем MSE
    loss_fn = nn.L1Loss()

    t0 = time.perf_counter()
    model.train()
    for ep in range(LSTM_EPOCHS):
        tot = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if LSTM_GRAD_CLIP is not None and LSTM_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"LSTM epoch {ep + 1}/{LSTM_EPOCHS} loss={tot / len(ds):.6f}")

    lstm_path = MODELS_DIR / "lstm_sparse.pt"
    torch.save(model.state_dict(), lstm_path)
    train_time = time.perf_counter() - t0

    model.eval()
    rows = []
    infer_total = 0.0

    with torch.no_grad():
        for series_id in tqdm(classic_test_df["series_id"].unique(), desc="LSTM infer"):
            if series_id not in series_stats:
                continue
            tr = classic_train_df[classic_train_df["series_id"] == series_id]
            te = classic_test_df[classic_test_df["series_id"] == series_id]
            if len(te) == 0:
                continue

            mu = series_stats[series_id]["mu"]
            sig = series_stats[series_id]["sig"]
            train_series = to_daily_series(tr)
            test_series = to_daily_series(te)
            z_hist = ((train_series - mu) / sig).values.astype(np.float32).tolist()

            t1 = time.perf_counter()
            preds = []
            for _ in range(len(test_series)):
                win = np.array(z_hist[-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, 1)
                w = torch.from_numpy(win).to(device)
                zn = model(w).cpu().numpy().reshape(-1)[0]
                preds.append(zn)
                z_hist.append(float(zn))
            dt = time.perf_counter() - t1
            infer_total += dt

            y_pred = np.array(preds) * sig + mu
            y_true = test_series.values
            temp = pd.DataFrame(
                {"date_dt": test_series.index, "y_true": y_true, "y_pred": y_pred}
            )
            temp["rest_id"] = tr["rest_id"].iloc[0]
            temp["dish_rus_name"] = tr["dish_rus_name"].iloc[0]
            temp["series_id"] = series_id
            temp["model"] = "lstm"
            n = max(len(temp), 1)
            temp["inference_time_sec"] = dt / n
            rows.append(temp)

    pred_df = pd.concat(rows, ignore_index=True)
    log_model_result(results, all_predictions, "LSTM", pred_df, train_time, infer_total)
    pred_df.to_parquet(PREDS_DIR / "lstm_predictions.parquet", index=False)
    save_final_test_predictions(pred_df, PREDS_DIR, "lstm")
    print(
        "LSTM готово — MAE:",
        results[-1]["mae"],
        "accuracy:",
        f"{results[-1]['accuracy']*100:.2f}%",
        "инференс, с:",
        infer_total,
    )


class PatchTSTReg(nn.Module):
    """Упрощённая PatchTST-регрессия для унивариантного ряда."""

    def __init__(self, seq_len=28, patch_len=7, d_model=64, n_heads=4, n_layers=2):
        """Инициализировать PatchTSTReg.

        Args:
            seq_len: Длина lookback.
            patch_len: Длина патча.
            d_model: Размерность эмбеддинга.
            n_heads: Число голов внимания.
            n_layers: Число слоёв encoder.
        """
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = max(1, seq_len // patch_len)
        self.in_proj = nn.Linear(patch_len, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """Прямой проход модели.

        Args:
            x: Вход shape (B, T).

        Returns:
            Прогноз shape (B, 1).
        """
        b = x.shape[0]
        x = x[:, -self.seq_len :]
        x = x.reshape(b, self.n_patches, self.patch_len)
        z = self.in_proj(x)
        z = self.encoder(z)
        z = z[:, -1, :]
        return self.head(z)


def _build_patch_windows(series_norm, seq_len, horizon=1):
    """Построить окна для PatchTST.

    Args:
        series_norm: Нормированный ряд (1D array).
        seq_len: Длина lookback-окна.
        horizon: Горизонт прогноза.

    Returns:
        (X_windows, y_targets).
    """
    X, y = [], []
    for i in range(seq_len, len(series_norm) - horizon + 1):
        X.append(series_norm[i - seq_len : i])
        y.append(series_norm[i + horizon - 1])
    return X, y


def run_patchtst_univariate(
    train_df,
    test_df,
    results,
    all_predictions,
    models_dir,
    preds_dir,
    series_col="series_id",
    date_col="date_dt",
    target_col="cnt_dishes",
    id_cols=("rest_id", "dish_rus_name"),
    model_name="PatchTST",
    pred_model_tag="patchtst",
    seq_len=28,
    epochs=PATCHTST_EPOCHS,
    batch_size=PATCHTST_BATCH,
    lr=PATCHTST_LR,
    random_seed=RANDOM_SEED,
    configs=None,
):
    """Обучить PatchTST на наборе рядов и получить прогнозы на test.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
        series_col: Колонка идентификатора ряда.
        date_col: Колонка даты.
        target_col: Целевая колонка.
        id_cols: Доп. идентификаторы, которые нужно сохранить в предиктах.
        model_name: Имя модели для таблицы метрик.
        pred_model_tag: Тег модели в предиктах и именах файлов.
        seq_len: Длина окна.
        epochs: Число эпох.
        batch_size: Размер батча.
        lr: Learning rate.
        random_seed: Seed.
        configs: Набор конфигураций.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    cfgs = configs or [{"name": "base", "seq_len": seq_len, "patch_len": PATCHTST_PATCH, "d_model": PATCHTST_D_MODEL, "n_heads": PATCHTST_HEADS, "n_layers": PATCHTST_LAYERS, "epochs": epochs, "batch_size": batch_size, "lr": lr}]

    best = None
    best_mae = np.inf
    infer_total = 0.0
    train_time_best = 0.0
    best_tag = "base"

    for c in cfgs:
        s_len = int(c["seq_len"])
        patch_len = int(c["patch_len"])
        d_model = int(c["d_model"])
        n_heads = int(c["n_heads"])
        n_layers = int(c["n_layers"])
        c_epochs = int(c.get("epochs", epochs))
        c_batch = int(c.get("batch_size", batch_size))
        c_lr = float(c.get("lr", lr))
        if s_len % patch_len != 0:
            continue

        Xw, yw = [], []
        stats = {}
        for sid in test_df[series_col].unique():
            tr = train_df[train_df[series_col] == sid]
            te = test_df[test_df[series_col] == sid]
            if len(tr) <= s_len or len(te) == 0:
                continue
            tr_s = tr.sort_values(date_col).set_index(date_col)[target_col].asfreq("D").fillna(0)
            mu = float(tr_s.mean())
            sig = float(tr_s.std()) + 1e-6
            z = ((tr_s - mu) / sig).values.astype(np.float32)
            xs, ys = _build_patch_windows(z, s_len, horizon=1)
            Xw.extend(xs)
            yw.extend(ys)
            stats[sid] = {"mu": mu, "sig": sig}
        if not Xw:
            continue

        X_arr = np.stack(Xw).astype(np.float32)
        y_arr = np.asarray(yw, dtype=np.float32).reshape(-1, 1)
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        dl = torch.utils.data.DataLoader(ds, batch_size=c_batch, shuffle=True)

        model = PatchTSTReg(seq_len=s_len, patch_len=patch_len, d_model=d_model, n_heads=n_heads, n_layers=n_layers).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c_lr)
        loss_fn = nn.L1Loss()
        t0 = time.perf_counter()
        model.train()
        for _ in range(c_epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
                opt.step()
        train_time = time.perf_counter() - t0

        rows = []
        infer_sum = 0.0
        model.eval()
        with torch.no_grad():
            for sid in test_df[series_col].unique():
                if sid not in stats:
                    continue
                tr = train_df[train_df[series_col] == sid].sort_values(date_col)
                te = test_df[test_df[series_col] == sid].sort_values(date_col)
                tr_s = tr.set_index(date_col)[target_col].asfreq("D").fillna(0)
                te_s = te.set_index(date_col)[target_col].asfreq("D").fillna(0)
                mu = stats[sid]["mu"]
                sig = stats[sid]["sig"]
                z_hist = ((tr_s - mu) / sig).values.astype(np.float32).tolist()
                t1 = time.perf_counter()
                preds = []
                for _ in range(len(te_s)):
                    win = np.array(z_hist[-s_len:], dtype=np.float32).reshape(1, s_len)
                    p = model(torch.from_numpy(win).to(device)).cpu().numpy().reshape(-1)[0]
                    preds.append(p)
                    z_hist.append(float(p))
                dt = time.perf_counter() - t1
                infer_sum += dt
                y_pred = np.array(preds) * sig + mu
                temp = pd.DataFrame({date_col: te_s.index, "y_true": te_s.values, "y_pred": y_pred})
                for col in id_cols:
                    if col in te.columns:
                        temp[col] = te[col].iloc[0]
                temp[series_col] = sid
                temp["model"] = pred_model_tag
                temp["inference_time_sec"] = dt / max(len(temp), 1)
                rows.append(temp)
        if not rows:
            continue
        pred_df = pd.concat(rows, ignore_index=True)
        cur_mae = mae(pred_df["y_true"], pred_df["y_pred"])
        if cur_mae < best_mae:
            best_mae = cur_mae
            best = pred_df
            infer_total = infer_sum
            train_time_best = train_time
            best_tag = c["name"]
            torch.save(model.state_dict(), models_dir / f"{pred_model_tag}_{best_tag}.pt")

    if best is None:
        print(f"{model_name}: не удалось получить прогнозы")
        return
    best.to_parquet(preds_dir / f"{pred_model_tag}_predictions.parquet", index=False)
    save_final_test_predictions(best, preds_dir, pred_model_tag)
    log_model_result(results, all_predictions, model_name, best, train_time_best, infer_total)


def run_patchtst(results, all_predictions, finalize_metrics=False, show_plot=False):
    """Запустить PatchTST для sparse и добавить результаты в общий список.

    Args:
        results: Список метрик.
        all_predictions: Список предсказаний.
        finalize_metrics: Сохранить метрики/график внутри функции.
        show_plot: Показывать график при finalize_metrics=True.
    """
    train_df, test_df, _ = load_classic_train_test()
    run_patchtst_univariate(
        train_df=train_df,
        test_df=test_df,
        results=results,
        all_predictions=all_predictions,
        models_dir=MODELS_DIR,
        preds_dir=PREDS_DIR,
        series_col="series_id",
        date_col="date_dt",
        target_col="cnt_dishes",
        id_cols=("rest_id", "dish_rus_name"),
        model_name="PatchTST",
        pred_model_tag="patchtst",
        seq_len=PATCHTST_SEQ_LEN,
        epochs=PATCHTST_EPOCHS,
        batch_size=PATCHTST_BATCH,
        lr=PATCHTST_LR,
        random_seed=RANDOM_SEED,
        configs=PATCHTST_CONFIGS,
    )
    if finalize_metrics:
        dfm = save_metrics_files(results, SPEC_KEY)
        fig_path = METRICS_DIR / f"{SPEC_KEY}_metrics_plot.png"
        plot_metrics_dashboard(dfm, save_path=fig_path, show=show_plot)


def run_sparse_experiment(
    do_ets=True,
    do_sarimax=True,
    do_xgb=True,
    do_lstm=True,
    do_patchtst=True,
    show_plot=True,
):
    """Запустить полный эксперимент sparse.

    Args:
        do_ets: Запуск ETS.
        do_sarimax: Запуск SARIMAX.
        do_xgb: Запуск XGBoost.
        do_lstm: Запуск LSTM.
        do_patchtst: Запуск PatchTST.
        show_plot: Показать итоговый график метрик.

    Returns:
        (df_metrics, all_predictions).
    """
    setup_dirs()
    results = []
    all_predictions = []

    train_df, test_df, test_start = load_classic_train_test()
    print("Тест с:", test_start.date(), "| train", train_df.shape, "| test", test_df.shape)

    print("\n[sparse] Старт эксперимента")
    if do_ets:
        print("[sparse][ETS] запуск...")
        run_ets(train_df, test_df, results, all_predictions)
    if do_sarimax:
        print("[sparse][SARIMAX] запуск...")
        run_sarimax(train_df, test_df, results, all_predictions)

    classic_train = train_df.copy()
    classic_test = test_df.copy()

    if do_xgb:
        print("[sparse][XGBoost] запуск...")
        run_xgboost(results, all_predictions)
    if do_lstm:
        print("[sparse][LSTM] запуск...")
        run_lstm(classic_train, classic_test, results, all_predictions)

    if do_patchtst:
        print("[sparse][PatchTST] запуск...")
        run_patchtst(
            results,
            all_predictions,
            finalize_metrics=False,
            show_plot=False,
        )
        print("[sparse][PatchTST] завершён")

    dfm = save_metrics_files(results, SPEC_KEY)
    fig_path = METRICS_DIR / f"{SPEC_KEY}_metrics_plot.png"
    plot_metrics_dashboard(dfm, save_path=fig_path, show=show_plot)

    print("\n=== Сводка метрик (sparse) ===")
    disp = dfm.copy()
    if "accuracy" in disp.columns:
        disp["accuracy_pct"] = (disp["accuracy"] * 100).round(2)
        disp_show = disp.drop(columns=["accuracy"]).rename(columns={"accuracy_pct": "accuracy_%"})
    else:
        disp_show = disp
    if "precision" in disp_show.columns:
        disp_show["precision_%"] = (disp_show["precision"] * 100).round(2)
        disp_show = disp_show.drop(columns=["precision"])
    print(disp_show.to_string(index=False))
    print("\nФайлы:", METRICS_DIR / f"{SPEC_KEY}_metrics.csv")
    return dfm, all_predictions


def _split_by_date_85_15(df, date_col):
    """Разделить даты на train/test в пропорции 85/15 по уникальным датам.

    Args:
        df: Таблица с колонкой дат.
        date_col: Имя колонки дат.

    Returns:
        Cutoff дата для train (включительно).
    """
    uniq_dates = np.sort(df[date_col].dropna().unique())
    if len(uniq_dates) < 2:
        raise ValueError(f"Недостаточно дат для split по {date_col}: {len(uniq_dates)}")
    split_idx = max(0, min(len(uniq_dates) - 2, int(len(uniq_dates) * 0.85) - 1))
    cutoff = pd.Timestamp(uniq_dates[split_idx])
    return cutoff


def load_pcr_train_test():
    """Загрузить PCR sequential dataset и выполнить split по датам 85/15.

    Returns:
        (train_df, test_df, cutoff).
    """
    df = pd.read_parquet(PCR_CLASSIC_PARQUET)
    df = df.sort_values(["city", "pcr_type", "date_dt"]).reset_index(drop=True)
    df["series_id"] = df["city"].astype(str) + "__" + df["pcr_type"].astype(str)
    cutoff = _split_by_date_85_15(df, "date_dt")
    train_df = df[df["date_dt"] <= cutoff].copy()
    test_df = df[df["date_dt"] > cutoff].copy()
    return train_df, test_df, cutoff


def run_ets_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir):
    """Запустить ETS для PCR-задачи.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
    """
    print("[disease][ETS_PCR] старт")
    ets_dir = models_dir / "ets"
    ets_dir.mkdir(parents=True, exist_ok=True)
    preds_all = []
    t0 = time.perf_counter()

    for i, series_id in enumerate(test_df["series_id"].unique()):
        if i % 200 == 0:
            print(f"ETS PCR: рядов {i} / {test_df['series_id'].nunique()}")
        tr = train_df[train_df["series_id"] == series_id].copy()
        te = test_df[test_df["series_id"] == series_id].copy()
        if len(tr) < 30 or len(te) == 0:
            continue
        train_series = to_daily_series(tr, value_col="target")
        test_series = to_daily_series(te, value_col="target")
        safe_id = series_id.replace("/", "_")
        model_path = ets_dir / f"{safe_id}.pkl"
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = ExponentialSmoothing(
                    train_series,
                    trend=None,
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit()
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            p0 = time.perf_counter()
            pr = model.forecast(len(test_series))
            pred_time = time.perf_counter() - p0
            temp = te.copy().set_index("date_dt").asfreq("D").reset_index()
            temp["series_id"] = series_id
            temp["y_true"] = test_series.values
            temp["y_pred"] = pr.values
            temp["model"] = "ets_pcr"
            temp["inference_time_sec"] = pred_time / max(len(temp), 1)
            preds_all.append(
                temp[
                    [
                        "city",
                        "pcr_type",
                        "series_id",
                        "date_dt",
                        "y_true",
                        "y_pred",
                        "model",
                        "inference_time_sec",
                    ]
                ]
            )
        except Exception:
            continue

    train_time = time.perf_counter() - t0
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_model_result(results, all_predictions, "ETS_PCR", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(preds_dir / "ets_pcr_predictions.parquet", index=False)
        save_final_test_predictions(pred_df, preds_dir, "ets_pcr")
        print(f"[disease][ETS_PCR] готово: {len(pred_df)} строк прогнозов")
    else:
        print("[disease][ETS_PCR] нет прогнозов")


def run_sarimax_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir):
    """Запустить SARIMAX для PCR-задачи.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
    """
    print("[disease][SARIMAX_PCR] старт")
    sdir = models_dir / "sarimax"
    sdir.mkdir(parents=True, exist_ok=True)
    preds_all = []
    t0 = time.perf_counter()

    for series_id in tqdm(test_df["series_id"].unique(), desc="SARIMAX PCR"):
        tr = train_df[train_df["series_id"] == series_id].copy()
        te = test_df[test_df["series_id"] == series_id].copy()
        if len(tr) < 40 or len(te) == 0:
            continue
        train_series = to_daily_series(tr, value_col="target")
        test_series = to_daily_series(te, value_col="target")
        safe_id = series_id.replace("/", "_")
        model_path = sdir / f"{safe_id}.pkl"
        try:
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                model = SARIMAX(
                    train_series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            p0 = time.perf_counter()
            pr = model.forecast(steps=len(test_series))
            pred_time = time.perf_counter() - p0
            temp = te.copy().set_index("date_dt").asfreq("D").reset_index()
            temp["series_id"] = series_id
            temp["y_true"] = test_series.values
            temp["y_pred"] = np.asarray(pr)
            temp["model"] = "sarimax_pcr"
            temp["inference_time_sec"] = pred_time / max(len(temp), 1)
            preds_all.append(
                temp[
                    [
                        "city",
                        "pcr_type",
                        "series_id",
                        "date_dt",
                        "y_true",
                        "y_pred",
                        "model",
                        "inference_time_sec",
                    ]
                ]
            )
        except Exception:
            continue

    train_time = time.perf_counter() - t0
    pred_df = pd.concat(preds_all, ignore_index=True) if preds_all else pd.DataFrame()
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_model_result(results, all_predictions, "SARIMAX_PCR", pred_df, train_time, infer_sum)
    if len(pred_df):
        pred_df.to_parquet(preds_dir / "sarimax_pcr_predictions.parquet", index=False)
        save_final_test_predictions(pred_df, preds_dir, "sarimax_pcr")
        print(f"[disease][SARIMAX_PCR] готово: {len(pred_df)} строк прогнозов")
    else:
        print("[disease][SARIMAX_PCR] нет прогнозов")


def run_xgboost_pcr(results, all_predictions, cutoff, models_dir, preds_dir):
    """Запустить XGBoost на PCR snapshot boosting-таблице.

    Args:
        results: Список метрик.
        all_predictions: Список предсказаний.
        cutoff: Дата разделения.
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
    """
    print("[disease][XGBoost_PCR] старт")
    df = pd.read_parquet(PCR_BOOST_PARQUET).copy()
    enc_city = LabelEncoder()
    enc_type = LabelEncoder()
    df["city_enc"] = enc_city.fit_transform(df["city"].astype(str))
    df["pcr_type_enc"] = enc_type.fit_transform(df["pcr_type"].astype(str))

    # Без утечек: train ограничиваем одновременно по snapshot_dt и forecast_dt.
    train_b = df[(df["snapshot_dt"] <= cutoff) & (df["forecast_dt"] <= cutoff)].copy()
    test_b = df[(df["snapshot_dt"] > cutoff) & (df["forecast_dt"] > cutoff)].copy()
    if len(train_b) == 0 or len(test_b) == 0:
        print("XGBoost PCR: пустой train/test после антиутечки split — пропуск")
        return
    print(f"[disease][XGBoost_PCR] train={train_b.shape}, test={test_b.shape}")

    drop_cols = ["city", "pcr_type", "snapshot_dt", "forecast_dt", "target"]
    X_train = train_b.drop(drop_cols, axis=1)
    y_train = train_b["target"].values
    X_test = test_b.drop(drop_cols, axis=1)
    y_test = test_b["target"].values

    # XGBoost не принимает object dtype: кодируем категориальные признаки
    # единым mapping на объединении train/test, чтобы не получить несовпадение кодов.
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        combo = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        categories = pd.Index(sorted(combo.unique()))
        mapping = {v: i for i, v in enumerate(categories)}
        X_train[col] = X_train[col].astype(str).map(mapping).astype(np.int32)
        X_test[col] = X_test[col].astype(str).map(mapping).astype(np.int32)

    path_model = models_dir / "xgboost_pcr_model.pkl"
    t0 = time.perf_counter()
    if path_model.exists():
        with open(path_model, "rb") as f:
            model = pickle.load(f)
    else:
        model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)
        with open(path_model, "wb") as f:
            pickle.dump(model, f)
    train_time = time.perf_counter() - t0

    p0 = time.perf_counter()
    y_hat = model.predict(X_test)
    inf_time = time.perf_counter() - p0

    pred_df = test_b[["city", "pcr_type", "forecast_dt"]].copy()
    pred_df = pred_df.rename(columns={"forecast_dt": "date_dt"})
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_hat
    pred_df["model"] = "xgboost_pcr"
    pred_df["inference_time_sec"] = inf_time / max(len(pred_df), 1)
    pred_df.to_parquet(preds_dir / "xgboost_pcr_predictions.parquet", index=False)
    save_final_test_predictions(pred_df, preds_dir, "xgboost_pcr")
    log_model_result(results, all_predictions, "XGBoost_PCR", pred_df, train_time, inf_time)
    print(f"[disease][XGBoost_PCR] готово: {len(pred_df)} строк прогнозов")


def run_lstm_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir):
    """Запустить LSTM для PCR-задачи.

    Args:
        train_df: Train-таблица.
        test_df: Test-таблица.
        results: Список метрик.
        all_predictions: Список предсказаний.
        models_dir: Каталог моделей.
        preds_dir: Каталог предсказаний.
    """
    print("[disease][LSTM_PCR] старт")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    Xw, yw = [], []
    series_stats = {}

    for series_id in test_df["series_id"].unique():
        tr = train_df[train_df["series_id"] == series_id]
        te = test_df[test_df["series_id"] == series_id]
        if len(tr) < 30 or len(te) == 0:
            continue
        train_series = to_daily_series(tr, value_col="target")
        if len(train_series) <= SEQ_LEN:
            continue
        mu = float(train_series.mean())
        sig = float(train_series.std()) + 1e-6
        z = (train_series - mu) / sig
        xs, ys = _build_windows(z, SEQ_LEN)
        Xw.extend(xs)
        yw.extend(ys)
        series_stats[series_id] = {"mu": mu, "sig": sig}

    if not Xw:
        print("LSTM PCR: нет окон — пропуск")
        return
    print(f"[disease][LSTM_PCR] train окон: {len(Xw)}")

    X_arr = np.stack(Xw, axis=0).astype(np.float32)
    y_arr = np.asarray(yw, dtype=np.float32).reshape(-1, 1)
    tensor_x = torch.from_numpy(X_arr).to(device)
    tensor_y = torch.from_numpy(y_arr).to(device)
    ds = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(ds, batch_size=LSTM_BATCH, shuffle=True)

    model = LSTMReg().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.L1Loss()
    t0 = time.perf_counter()
    model.train()
    for _ in range(LSTM_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if LSTM_GRAD_CLIP is not None and LSTM_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
            opt.step()
    torch.save(model.state_dict(), models_dir / "lstm_pcr.pt")
    train_time = time.perf_counter() - t0

    model.eval()
    rows = []
    infer_total = 0.0
    with torch.no_grad():
        for series_id in test_df["series_id"].unique():
            if series_id not in series_stats:
                continue
            tr = train_df[train_df["series_id"] == series_id]
            te = test_df[test_df["series_id"] == series_id]
            if len(te) == 0:
                continue
            mu = series_stats[series_id]["mu"]
            sig = series_stats[series_id]["sig"]
            train_series = to_daily_series(tr, value_col="target")
            test_series = to_daily_series(te, value_col="target")
            z_hist = ((train_series - mu) / sig).values.astype(np.float32).tolist()
            t1 = time.perf_counter()
            preds = []
            for _ in range(len(test_series)):
                win = np.array(z_hist[-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, 1)
                w = torch.from_numpy(win).to(device)
                zn = model(w).cpu().numpy().reshape(-1)[0]
                preds.append(zn)
                z_hist.append(float(zn))
            dt = time.perf_counter() - t1
            infer_total += dt
            y_pred = np.array(preds) * sig + mu
            temp = pd.DataFrame(
                {"date_dt": test_series.index, "y_true": test_series.values, "y_pred": y_pred}
            )
            temp["city"] = tr["city"].iloc[0]
            temp["pcr_type"] = tr["pcr_type"].iloc[0]
            temp["series_id"] = series_id
            temp["model"] = "lstm_pcr"
            temp["inference_time_sec"] = dt / max(len(temp), 1)
            rows.append(temp)

    pred_df = pd.concat(rows, ignore_index=True)
    pred_df.to_parquet(preds_dir / "lstm_pcr_predictions.parquet", index=False)
    save_final_test_predictions(pred_df, preds_dir, "lstm_pcr")
    log_model_result(results, all_predictions, "LSTM_PCR", pred_df, train_time, infer_total)
    print(f"[disease][LSTM_PCR] готово: {len(pred_df)} строк прогнозов")


def run_disease_experiment(
    do_ets=True,
    do_sarimax=True,
    do_xgb=True,
    do_lstm=True,
    do_patchtst=True,
    show_plot=True,
):
    """Запустить полный эксперимент disease (PCR).

    Args:
        do_ets: Запуск ETS.
        do_sarimax: Запуск SARIMAX.
        do_xgb: Запуск XGBoost.
        do_lstm: Запуск LSTM.
        do_patchtst: Запуск PatchTST.
        show_plot: Показать итоговый график метрик.

    Returns:
        (df_metrics, all_predictions).
    """
    _artifacts = cfg.artifact_dirs(DISEASE_SPEC_KEY)
    artifacts_dir = _artifacts["base"]
    models_dir = _artifacts["models"]
    preds_dir = _artifacts["predictions"]
    metrics_dir = _artifacts["metrics"]
    setup_dirs_for(models_dir, preds_dir, metrics_dir)

    results = []
    all_predictions = []
    train_df, test_df, cutoff = load_pcr_train_test()
    print(
        "PCR test с:",
        (cutoff + pd.Timedelta(days=1)).date(),
        "| train",
        train_df.shape,
        "| test",
        test_df.shape,
    )

    print("\n[disease] Старт эксперимента")
    if do_ets:
        print("[disease][ETS_PCR] запуск...")
        run_ets_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir)
    if do_sarimax:
        print("[disease][SARIMAX_PCR] запуск...")
        run_sarimax_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir)
    if do_xgb:
        print("[disease][XGBoost_PCR] запуск...")
        run_xgboost_pcr(results, all_predictions, cutoff, models_dir, preds_dir)
    if do_lstm:
        print("[disease][LSTM_PCR] запуск...")
        run_lstm_pcr(train_df, test_df, results, all_predictions, models_dir, preds_dir)
    if do_patchtst:
        print("[disease][PatchTST_PCR] запуск...")
        run_patchtst_univariate(
            train_df=train_df,
            test_df=test_df,
            results=results,
            all_predictions=all_predictions,
            models_dir=models_dir,
            preds_dir=preds_dir,
            series_col="series_id",
            date_col="date_dt",
            target_col="target",
            id_cols=("city", "pcr_type"),
            model_name="PatchTST_PCR",
            pred_model_tag="patchtst_pcr",
            seq_len=PATCHTST_SEQ_LEN,
            patch_len=PATCHTST_PATCH,
            d_model=PATCHTST_D_MODEL,
            n_heads=PATCHTST_HEADS,
            n_layers=PATCHTST_LAYERS,
            epochs=PATCHTST_EPOCHS,
            batch_size=PATCHTST_BATCH,
            lr=PATCHTST_LR,
            random_seed=RANDOM_SEED,
            log_prefix="[disease][PatchTST_PCR]",
            configs=[
                {
                    "name": c["name"],
                    "seq_len": c["lookback"],
                    "patch_len": c["patch_len"],
                    "d_model": c["d_model"],
                    "n_heads": c["n_heads"],
                    "n_layers": c["n_enc_layers"],
                    "epochs": PATCHTST_EPOCHS,
                    "batch_size": PATCHTST_BATCH,
                    "lr": PATCHTST_LR,
                }
                for c in PATCHTST_CONFIGS
            ],
        )

    dfm = save_metrics_files(results, DISEASE_SPEC_KEY, metrics_dir=metrics_dir)
    fig_path = metrics_dir / f"{DISEASE_SPEC_KEY}_metrics_plot.png"
    plot_metrics_dashboard(
        dfm,
        title="Задача ПЦР",
        save_path=fig_path,
        show=show_plot,
    )
    print("\n=== Сводка метрик (disease) ===")
    disp = dfm.copy()
    if "accuracy" in disp.columns:
        disp["accuracy_pct"] = (disp["accuracy"] * 100).round(2)
        disp_show = disp.drop(columns=["accuracy"]).rename(columns={"accuracy_pct": "accuracy_%"})
    else:
        disp_show = disp
    if "precision" in disp_show.columns:
        disp_show["precision_%"] = (disp_show["precision"] * 100).round(2)
        disp_show = disp_show.drop(columns=["precision"])
    print(disp_show.to_string(index=False))
    print("\nФайлы:", metrics_dir / f"{DISEASE_SPEC_KEY}_metrics.csv")
    return dfm, all_predictions


if __name__ == "__main__":
    if RUN_SPARSE_TASK:
        run_sparse_experiment()
    if RUN_DISEASE_TASK:
        run_disease_experiment()
