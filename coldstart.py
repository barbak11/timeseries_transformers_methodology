# -*- coding: utf-8 -*-
"""
Специфика «короткие временные ряды»: traffic_classic / traffic_boosting, цель checks_cnt.

Отбор рядов: open_days на первый день теста в (SHORT_OPEN_DAYS_MIN, SHORT_OPEN_DAYS_MAX).
Погодные признаки не используются (нет на будущее).

Метрики для этой специфики: RMSE, WAPE, % (+ время обучения и инференса).
Считаются только по коротким рядам в тесте.
Используются только XGBoost и TFT.

Запуск:  python nir_short_baseline.py
Или:     from nir_short_baseline import run_short_experiment; run_short_experiment()
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from common import short_shared as short_common

# ───────────────────────── Константы специфики ─────────────────────────
SC = cfg.COLDSTART_CONFIG
SPEC_KEY = SC["SPEC_KEY"]
RETRAIN = SC["RETRAIN"]
RUN_TRAFFIC_TASK = SC["RUN_TRAFFIC_TASK"]
RUN_DISEASE_TASK = SC["RUN_DISEASE_TASK"]
PRINT_METRICS_TO_STDOUT = SC["PRINT_METRICS_TO_STDOUT"]
RANDOM_SEED = cfg.RANDOM_SEED
DATA_DIR = cfg.DATA_DIR
CLASSIC_PARQUET = cfg.TRAFFIC_CLASSIC_PARQUET
BOOST_PARQUET = cfg.TRAFFIC_BOOST_PARQUET
DISEASE_SEQUENTIAL_PARQUET = cfg.DISEASE_SEQUENTIAL_PARQUET
DISEASE_SNAPSHOT_PARQUET = cfg.DISEASE_SNAPSHOT_PARQUET

# Горизонт теста (как в sparse)
TEST_HORIZON_YEARS = cfg.TEST_HORIZON_YEARS

# Короткая история: open_days в первый день теста (строго больше 7 и строго меньше 90)
SHORT_OPEN_DAYS_MIN = SC["SHORT_OPEN_DAYS_MIN"]   # условие: open_days > этого значения  → минимум 8 дней
SHORT_OPEN_DAYS_MAX = SC["SHORT_OPEN_DAYS_MAX"]  # условие: open_days < этого значения → меньше 3 мес. (≈90 дн.)

# Минимум наблюдений в train для классики (снижены под короткие ряды)
XGB_N_ESTIMATORS = cfg.XGB_N_ESTIMATORS
XGB_MAX_DEPTH = cfg.XGB_MAX_DEPTH
XGB_LEARNING_RATE = cfg.XGB_LEARNING_RATE
OPTUNA_N_TRIALS = SC["OPTUNA_N_TRIALS"]

DISEASE_SHORT_SHARE = SC["DISEASE_SHORT_SHARE"]
DISEASE_MIN_WEEKS = SC["DISEASE_MIN_WEEKS"]
DISEASE_MAX_WEEKS = SC["DISEASE_MAX_WEEKS"]

_artifacts = cfg.artifact_dirs(SPEC_KEY)
ARTIFACTS_DIR = _artifacts["base"]
MODELS_DIR = _artifacts["models"]
PREDS_DIR = _artifacts["predictions"]
METRICS_DIR = _artifacts["metrics"]

# Признаки погоды — не используем (в т.ч. в бустинге)
WEATHER_COLS = cfg.WEATHER_COLS
TARGET_COL = cfg.TARGET_COL

# ───────────────────────── TFT ─────────────────────────
TFT_LOOKBACK = SC["TFT_LOOKBACK"]
TFT_HORIZON = SC["TFT_HORIZON"]
TFT_HIDDEN = SC["TFT_HIDDEN"]
TFT_LSTM_LAYERS = SC["TFT_LSTM_LAYERS"]
TFT_ATTN_HEADS = SC["TFT_ATTN_HEADS"]
TFT_DROPOUT = SC["TFT_DROPOUT"]
TFT_EPOCHS = SC["TFT_EPOCHS"]
TFT_BATCH = SC["TFT_BATCH"]
TFT_LR = SC["TFT_LR"]
TFT_GRAD_CLIP = SC["TFT_GRAD_CLIP"]
TFT_PATIENCE = SC["TFT_PATIENCE"]
TFT_TRAIN_STRIDE = SC["TFT_TRAIN_STRIDE"]
TFT_VAL_FRAC = SC["TFT_VAL_FRAC"]

TFT_KNOWN_FUTURE_COLS = SC["TFT_KNOWN_FUTURE_COLS"]

# В cold-start постановке не опираемся на past observed context:
# у нового ряда не должно быть обязательной зависимости от исторических наблюдений.
TFT_OBSERVED_PAST_COLS = SC["TFT_OBSERVED_PAST_COLS"]

TFT_STATIC_CAT_COLS = SC["TFT_STATIC_CAT_COLS"]

TFT_STATIC_NUM_COLS = SC["TFT_STATIC_NUM_COLS"]

TFT_CAT_EMB_DIM = SC["TFT_CAT_EMB_DIM"]
TFT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TFT_ARCH_CONFIGS = SC["TFT_ARCH_CONFIGS"]
TFT_SCALE_GROUP_LEVELS = SC["TFT_SCALE_GROUP_LEVELS"]
TFT_SCALE_MIN_SAMPLES = SC["TFT_SCALE_MIN_SAMPLES"]


def setup_dirs():
    for d in (MODELS_DIR, PREDS_DIR, METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def task_artifact_paths(task_key: str):
    base = ARTIFACTS_DIR / str(task_key)
    return {
        "base_dir": base,
        "models_dir": base / "models",
        "preds_dir": base / "predictions",
        "metrics_dir": base / "metrics",
    }


def setup_task_dirs(paths: dict):
    for d in (paths["models_dir"], paths["preds_dir"], paths["metrics_dir"]):
        d.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def wape(y_true, y_pred):
    """Σ|y−ŷ| / Σ|y| · 100."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100)


def log_short_result(results, all_predictions, model_name, pred_df, train_sec, infer_sec):
    if pred_df is None or len(pred_df) == 0:
        return
    m = np.isfinite(pd.to_numeric(pred_df["y_true"], errors="coerce")) & np.isfinite(
        pd.to_numeric(pred_df["y_pred"], errors="coerce")
    )
    if not bool(m.all()):
        bad = int((~m).sum())
        print(f"{model_name}: отброшено строк с NaN/inf перед метриками: {bad}")
    pred_df = pred_df.loc[m].copy()
    if len(pred_df) == 0:
        print(f"{model_name}: после фильтра NaN/inf не осталось строк для метрик")
        return
    results.append(
        {
            "model": model_name,
            "rmse": rmse(pred_df["y_true"], pred_df["y_pred"]),
            "wape": wape(pred_df["y_true"], pred_df["y_pred"]),
            "train_time_sec": float(train_sec),
            "inference_time_sec": float(infer_sec),
        }
    )
    all_predictions.append(pred_df)


def plot_short_metrics_dashboard(dfm, title="Специфика: короткие ряды", save_path=None, show=True):
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
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_metrics_files(results, spec_key=SPEC_KEY, metrics_dir=METRICS_DIR):
    dfm = pd.DataFrame(results)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    dfm.to_csv(metrics_dir / f"{spec_key}_metrics.csv", index=False)
    dfm.to_json(
        metrics_dir / f"{spec_key}_metrics.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    return dfm


def to_daily_series(part, value_col=TARGET_COL):
    return part.set_index("date_dt")[value_col].asfreq("D").fillna(0)


def test_start_from_df(df):
    max_date = df["date_dt"].max()
    return max_date - pd.DateOffset(years=TEST_HORIZON_YEARS) + pd.Timedelta(days=1)


def compute_test_start_by_ratio(df, date_col="date_dt", train_ratio=0.70, val_ratio=0.15):
    dts = np.sort(pd.to_datetime(df[date_col]).dropna().unique())
    n = len(dts)
    if n < 10:
        raise ValueError("Слишком мало уникальных дат для ratio split")
    i_test = max(1, min(n - 1, int(np.floor(n * (train_ratio + val_ratio)))))
    return pd.Timestamp(dts[i_test])


def select_shortest_share_ids(train_df, share=0.10, id_col="rest_id"):
    lengths = train_df.groupby(id_col)["date_dt"].nunique().sort_values()
    if len(lengths) == 0:
        return []
    k = max(1, int(np.ceil(len(lengths) * float(share))))
    return lengths.head(k).index.astype(int).tolist()


def truncate_disease_train_random_weeks(train_df, short_ids, min_weeks=50, max_weeks=150, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    out = []
    short_set = set(int(x) for x in short_ids)
    for rid, grp in train_df.groupby("rest_id", sort=False):
        g = grp.sort_values("date_dt").copy()
        if int(rid) not in short_set:
            out.append(g)
            continue
        keep = int(rng.integers(min_weeks, max_weeks + 1))
        if len(g) > keep:
            g = g.iloc[-keep:].copy()
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else train_df.copy()


def coerce_year_week_to_datetime(s: pd.Series) -> pd.Series:
    x = s.copy()
    dt = pd.to_datetime(x, errors="coerce")
    mask = dt.isna() & x.notna()
    if bool(mask.any()):
        xw = x[mask].astype(str).str.strip()
        xw = xw.str.replace(
            r"^(\d{4})-(\d{1,2})$",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-1",
            regex=True,
        )
        dt2 = pd.to_datetime(xw, format="%Y-%W-%w", errors="coerce")
        dt.loc[mask] = dt2
    return dt


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
        df["date_dt"] = coerce_year_week_to_datetime(df["date_dt"])
    else:
        raise ValueError("disease_sequential_dataset.parquet: нужна колонка date_full или date_dt")

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["date_dt", "target"])
    df["series_id"] = df["city"].astype(str) + "__" + df["age_group"].astype(str)
    df["rest_id"] = pd.factorize(df["series_id"], sort=True)[0].astype(int)
    df = df.sort_values(["rest_id", "date_dt"]).reset_index(drop=True)
    return df


def rest_open_days_on_first_test_day(df, test_start):
    """open_days на первый календарный день теста (если строки нет — ближайший следующий день в данных)."""
    ts = pd.Timestamp(test_start)
    on_day = df[df["date_dt"] == ts][["rest_id", "open_days"]].drop_duplicates("rest_id")
    if len(on_day) == len(df["rest_id"].unique()):
        return on_day
    have = set(on_day["rest_id"].unique())
    need = df["rest_id"].unique()
    extra = []
    for rid in need:
        if rid in have:
            continue
        sub = df[(df["rest_id"] == rid) & (df["date_dt"] >= ts)].sort_values("date_dt")
        if len(sub) == 0:
            continue
        r0 = sub.iloc[0]
        extra.append({"rest_id": rid, "open_days": r0["open_days"]})
    if extra:
        on_day = pd.concat([on_day, pd.DataFrame(extra)], ignore_index=True)
    return on_day.drop_duplicates("rest_id")


def select_short_rest_ids(open_days_df):
    """open_days > SHORT_OPEN_DAYS_MIN и open_days < SHORT_OPEN_DAYS_MAX."""
    m = (open_days_df["open_days"] > SHORT_OPEN_DAYS_MIN) & (
        open_days_df["open_days"] < SHORT_OPEN_DAYS_MAX
    )
    ids = open_days_df.loc[m, "rest_id"].astype(int).unique().tolist()
    return sorted(ids)


def load_classic_split(df, test_start):
    df = df.sort_values(["rest_id", "date_dt"]).reset_index(drop=True)
    df["series_id"] = df["rest_id"].astype(str)
    train_df = df[df["date_dt"] < test_start].copy()
    test_df = df[df["date_dt"] >= test_start].copy()
    return train_df, test_df


def xgboost_drop_columns(df):
    base = [
        TARGET_COL,
        "date_dt",
        "rest_id",
        "rest_uk",
        "snapshot_dt",
        "avg_checks_month",
        "avg_day_checks_month",
        "max_checks_month",
        "min_checks_month",
    ]
    w = [c for c in WEATHER_COLS if c in df.columns]
    return base + w


def _history_feature_columns(columns):
    """Эвристика: признаки, которые обычно содержат историю таргета."""
    pats = ("lag", "rolling", "roll_", "ewm", "ema", "ma_", "window", "history", "hist", "target_prev")
    out = []
    for c in columns:
        lc = str(c).lower()
        if any(p in lc for p in pats):
            out.append(c)
    return out


def run_xgboost_all_eval_short(
    short_rest_ids,
    test_start,
    results,
    all_predictions,
    *,
    retrain=True,
    models_dir=MODELS_DIR,
    preds_dir=PREDS_DIR,
):
    df = pd.read_parquet(BOOST_PARQUET)
    if "dish_rus_name" in df.columns:
        enc = LabelEncoder()
        df["dish_rus_name_enc"] = enc.fit_transform(df["dish_rus_name"].astype(str))

    drop_cols = [c for c in xgboost_drop_columns(df) if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    hist_cols = _history_feature_columns(feature_cols)
    if hist_cols:
        feature_cols = [c for c in feature_cols if c not in set(hist_cols)]
        print(f"XGBoost: исключены history-фичи из обучения/инференса: {len(hist_cols)} колонок")

    short_set = set(int(x) for x in short_rest_ids)
    train_b = df[(df["date_dt"] < test_start) & (~df["rest_id"].isin(short_set))].copy()
    test_b = df[df["date_dt"] >= test_start].copy()
    if len(train_b) == 0:
        print("XGBoost: пустой train после удаления истории cold-start рядов, пропуск.")
        return

    X_train = train_b[feature_cols].copy()
    X_test = test_b[feature_cols].copy()
    for c in X_train.columns:
        if X_train[c].dtype in ("float64", "float32") or X_train[c].isna().any():
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            X_test[c] = X_test[c].fillna(med)

    y_train = train_b[TARGET_COL]
    y_test = test_b[TARGET_COL]

    path_model = models_dir / "xgboost_coldstart_2_global.pkl"
    path_params = models_dir / "xgboost_coldstart_2_global_params.json"
    val_start = pd.Timestamp(test_start) - pd.Timedelta(days=28)
    tr_fit = train_b[train_b["date_dt"] < val_start].copy()
    val_b = train_b[train_b["date_dt"] >= val_start].copy()
    if len(tr_fit) < 50 or len(val_b) < 10:
        q = train_b["date_dt"].quantile(0.8)
        tr_fit = train_b[train_b["date_dt"] < q].copy()
        val_b = train_b[train_b["date_dt"] >= q].copy()
    if len(tr_fit) < 20 or len(val_b) < 5:
        tr_fit = train_b.copy()
        val_b = train_b.sample(min(len(train_b), 2000), random_state=RANDOM_SEED)

    X_tr = tr_fit[feature_cols].copy()
    X_va = val_b[feature_cols].copy()
    for c in X_tr.columns:
        if X_tr[c].dtype in ("float64", "float32") or X_tr[c].isna().any() or X_va[c].isna().any():
            med = X_tr[c].median()
            X_tr[c] = X_tr[c].fillna(med)
            X_va[c] = X_va[c].fillna(med)
    y_tr = tr_fit[TARGET_COL].copy()
    y_va = val_b[TARGET_COL].copy()

    best_params = None
    if (not retrain) and path_params.exists():
        try:
            with open(path_params, encoding="utf-8") as f:
                best_params = json.load(f)
        except Exception:
            best_params = None
    if best_params is None:
        print(
            f"\n>>> XGBoost [short traffic]: Optuna подбор ({OPTUNA_N_TRIALS} trials), "
            f"train_fit={len(tr_fit)} | val={len(val_b)} | test={len(test_b)} | фичей={len(feature_cols)}"
        )

        def objective(trial):
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

        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        st.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
        best_params = st.best_params
        with open(path_params, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)
        print(f"    лучшие параметры: {best_params} | val_RMSE={st.best_value:.6f}")

    t0 = time.perf_counter()
    if path_model.exists() and not retrain:
        with open(path_model, "rb") as f:
            model = pickle.load(f)
    else:
        model = XGBRegressor(
            **best_params,
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

    pred_all = test_b[["rest_id", "date_dt"]].copy()
    pred_all["y_true"] = y_test.values
    pred_all["y_pred"] = y_hat
    pred_all["model"] = "xgboost"
    pred_all["inference_time_sec"] = inf_time / max(len(pred_all), 1)

    pred_short = pred_all[pred_all["rest_id"].isin(short_set)].copy()
    pred_short["series_id"] = pred_short["rest_id"].astype(str)

    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_all.to_parquet(preds_dir / "coldstart_2_xgboost_predictions_all.parquet", index=False)
    pred_short.to_parquet(preds_dir / "coldstart_2_xgboost_predictions_short_only.parquet", index=False)

    infer_short = float(pred_short["inference_time_sec"].sum()) if len(pred_short) else 0.0
    log_short_result(results, all_predictions, "XGBoost", pred_short, train_time, infer_short)
    print(
        "XGBoost (оценка только короткие ряды) — RMSE:",
        results[-1]["rmse"],
        "WAPE:",
        f"{results[-1]['wape']:.2f}%",
        f"| строк: {len(pred_short)}",
    )


# ═══════════════════════════ TFT: данные ═══════════════════════════

def _norm_static_value(val):
    if pd.isna(val):
        return "__na__"
    if isinstance(val, (np.floating, float)):
        return int(val) if float(val).is_integer() else float(val)
    return val


def _group_key_from_row(row, cols):
    return tuple(_norm_static_value(row.get(c, np.nan)) for c in cols)


def _build_group_scalers(df, test_start):
    train_part = df[df["date_dt"] < test_start].copy()
    if len(train_part) == 0:
        train_part = df.copy()
    glob_mu = float(train_part[TARGET_COL].mean()) if len(train_part) else 0.0
    glob_sig = float(train_part[TARGET_COL].std(ddof=0)) if len(train_part) else 1.0
    if not np.isfinite(glob_mu):
        glob_mu = 0.0
    if (not np.isfinite(glob_sig)) or glob_sig <= 1e-8:
        glob_sig = 1.0

    scalers = {}
    for cols in TFT_SCALE_GROUP_LEVELS:
        cols = tuple(cols)
        work = train_part[list(cols) + [TARGET_COL]].copy()
        for c in cols:
            work[c] = work[c].map(_norm_static_value)
        agg = (
            work.groupby(list(cols), dropna=False)[TARGET_COL]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        level_map = {}
        for _, row in agg.iterrows():
            count = int(row["count"])
            if count < TFT_SCALE_MIN_SAMPLES:
                continue
            mu = float(row["mean"])
            sigma = float(row["std"])
            if not np.isfinite(mu):
                mu = glob_mu
            if (not np.isfinite(sigma)) or sigma <= 1e-8:
                sigma = glob_sig
            key = tuple(row[c] for c in cols)
            level_map[key] = {
                "mu": mu,
                "sigma": sigma,
                "count": count,
                "source": "+".join(cols),
            }
        scalers[cols] = level_map
    global_stats = {"mu": glob_mu, "sigma": glob_sig, "count": len(train_part), "source": "global"}
    return scalers, global_stats


def _resolve_group_scaler(row, scalers, global_stats):
    for cols in TFT_SCALE_GROUP_LEVELS:
        key = _group_key_from_row(row, cols)
        hit = scalers.get(tuple(cols), {}).get(key)
        if hit is not None:
            return hit
    return global_stats


def prepare_tft_series(df, test_start):
    df = df.sort_values(["rest_id", "date_dt"]).copy()
    rest_ids = sorted(df["rest_id"].unique())

    cat_maps = {}
    for col in TFT_STATIC_CAT_COLS:
        ser = df[col].map(_norm_static_value)
        cat_maps[col] = {v: i for i, v in enumerate(sorted(ser.unique().tolist(), key=str))}
    cat_dims = [len(cat_maps[col]) for col in TFT_STATIC_CAT_COLS]

    norm_base = df[df["date_dt"] < test_start].copy()
    if len(norm_base) == 0:
        norm_base = df.copy()
    kf_mean = norm_base[TFT_KNOWN_FUTURE_COLS].mean().fillna(0.0).astype(np.float32)
    kf_std = norm_base[TFT_KNOWN_FUTURE_COLS].std(ddof=0).replace(0, 1).fillna(1.0).astype(np.float32)
    sn_mean = norm_base[TFT_STATIC_NUM_COLS].mean().fillna(0.0).astype(np.float32)
    sn_std = norm_base[TFT_STATIC_NUM_COLS].std(ddof=0).replace(0, 1).fillna(1.0).astype(np.float32)
    group_scalers, global_stats = _build_group_scalers(df, test_start)

    series_data = {}
    for rid in rest_ids:
        grp = df[df["rest_id"] == rid].sort_values("date_dt").copy()
        dates = pd.date_range(grp["date_dt"].min(), grp["date_dt"].max(), freq="D")
        daily = grp.set_index("date_dt").reindex(dates)

        daily[TARGET_COL] = daily[TARGET_COL].fillna(0.0)
        target = daily[TARGET_COL].values.astype(np.float32)
        te_idx = int((dates < test_start).sum())
        row0 = grp.iloc[0]
        scaler = _resolve_group_scaler(row0, group_scalers, global_stats)
        mu = float(scaler["mu"])
        sigma = float(scaler["sigma"])
        if not np.isfinite(mu):
            mu = 0.0
        if (not np.isfinite(sigma)) or sigma <= 1e-8:
            sigma = 1.0
        target_z = ((target - mu) / sigma).astype(np.float32)

        for c in TFT_KNOWN_FUTURE_COLS:
            if c in daily.columns:
                daily[c] = daily[c].ffill().bfill().fillna(0)
            else:
                daily[c] = 0.0
        kf = daily[TFT_KNOWN_FUTURE_COLS].values.astype(np.float32)
        kf_n = ((kf - kf_mean.values) / kf_std.values).astype(np.float32)

        scat = []
        for col in TFT_STATIC_CAT_COLS:
            val = _norm_static_value(row0.get(col, np.nan))
            scat.append(cat_maps[col].get(val, 0))
        scat = np.array(scat, dtype=np.int64)

        sn_raw = np.array([float(row0.get(c, 0.0)) for c in TFT_STATIC_NUM_COLS], dtype=np.float32)
        sn_raw = np.nan_to_num(sn_raw, 0.0)
        sn_n = ((sn_raw - sn_mean.values) / sn_std.values).astype(np.float32)

        series_data[str(rid)] = {
            "dates": dates,
            "target": target,
            "target_z": target_z,
            "kf": kf_n,
            "scat": scat,
            "snum": sn_n,
            "mu": mu,
            "sigma": sigma,
            "train_end": te_idx,
            "rest_id": int(rid),
            "scale_source": str(scaler["source"]),
            "scale_count": int(scaler["count"]),
        }
    return series_data, cat_dims, len(rest_ids)


class TFTWindowDataset(torch.utils.data.Dataset):
    def __init__(self, series_data, windows, lookback, horizon):
        self.sd = series_data
        self.win = windows
        self.L = lookback
        self.H = horizon

    def __len__(self):
        return len(self.win)

    def __getitem__(self, i):
        sid, start = self.win[i]
        s = self.sd[sid]
        pe = start + self.L
        fe = pe + self.H
        return (
            torch.from_numpy(s["scat"]),
            torch.from_numpy(s["snum"]),
            torch.from_numpy(s["kf"][start:pe]),
            torch.from_numpy(s["kf"][pe:fe]),
            torch.from_numpy(s["target_z"][pe:fe]),
        )


def make_tft_windows(series_data, lookback, horizon, stride=1):
    wins = []
    for sid, s in series_data.items():
        mx = s["train_end"] - lookback - horizon
        if mx < 0:
            continue
        for t in range(0, mx + 1, stride):
            wins.append((sid, t))
    return wins


# ═══════════════════════════ TFT: модель ═══════════════════════════

class GRN(nn.Module):
    def __init__(self, d_in, d_h, d_out=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_out)
        self.gate = nn.Linear(d_h, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None

    def forward(self, x):
        r = self.skip(x) if self.skip is not None else x
        h = F.elu(self.fc1(x))
        h = self.drop(h)
        return self.norm(r + torch.sigmoid(self.gate(h)) * self.fc2(h))


class TFTModel(nn.Module):
    def __init__(
        self,
        cat_dims,
        n_snum,
        n_kf,
        lookback,
        horizon,
        d=TFT_HIDDEN,
        lstm_layers=TFT_LSTM_LAYERS,
        n_heads=TFT_ATTN_HEADS,
        drop=TFT_DROPOUT,
    ):
        super().__init__()
        self.d = d
        self.n_layers = lstm_layers
        self.horizon = horizon

        self.cat_embs = nn.ModuleList([nn.Embedding(cd + 1, TFT_CAT_EMB_DIM) for cd in cat_dims])
        cat_total = len(cat_dims) * TFT_CAT_EMB_DIM

        static_in = cat_total + n_snum
        self.static_grn = GRN(static_in, d, d, drop)
        self.ctx_h = nn.Linear(d, d * lstm_layers)
        self.ctx_c = nn.Linear(d, d * lstm_layers)
        self.ctx_enrich = nn.Linear(d, d)

        self.past_proj = nn.Linear(n_kf, d)
        self.future_proj = nn.Linear(n_kf, d)

        self.enc_lstm = nn.LSTM(
            d, d, lstm_layers, batch_first=True,
            dropout=drop if lstm_layers > 1 else 0,
        )
        self.dec_lstm = nn.LSTM(
            d, d, lstm_layers, batch_first=True,
            dropout=drop if lstm_layers > 1 else 0,
        )

        self.enrich_grn = GRN(d * 2, d, d, drop)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=drop, batch_first=True)
        self.attn_grn = GRN(d, d, d, drop)
        self.out_grn = GRN(d, d, d, drop)
        self.fc_out = nn.Linear(d, 1)

    def forward(self, scat, snum, past_kf, fut_kf):
        B = scat.size(0)
        cats = [emb(scat[:, i]) for i, emb in enumerate(self.cat_embs)]
        ctx = self.static_grn(torch.cat(cats + [snum], dim=1))

        h0 = self.ctx_h(ctx).view(B, self.n_layers, self.d).permute(1, 0, 2).contiguous()
        c0 = self.ctx_c(ctx).view(B, self.n_layers, self.d).permute(1, 0, 2).contiguous()
        enrich = self.ctx_enrich(ctx).unsqueeze(1)

        enc_in = self.past_proj(past_kf)
        enc_out, (hn, cn) = self.enc_lstm(enc_in, (h0, c0))

        dec_in = self.future_proj(fut_kf)
        dec_out, _ = self.dec_lstm(dec_in, (hn, cn))

        dec_e = self.enrich_grn(torch.cat([dec_out, enrich.expand_as(dec_out)], dim=2))
        attn_out, _ = self.attn(dec_e, enc_out, enc_out)
        h = self.attn_grn(attn_out + dec_e)
        h = self.out_grn(h)
        return self.fc_out(h).squeeze(-1)


# ═══════════════════════════ TFT: обучение ═══════════════════════════

def train_tft(model, tr_loader, va_loader, epochs, lr, grad_clip, patience, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.SmoothL1Loss(beta=0.5)

    best_val, best_st, wait = float("inf"), None, 0
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        s_loss, n = 0.0, 0
        for batch in tr_loader:
            sc, sn, pk, fk, tz = [b.to(device) for b in batch]
            pred = model(sc, sn, pk, fk)
            loss = loss_fn(pred, tz)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            bs = sc.size(0)
            s_loss += loss.item() * bs
            n += bs
        s_loss /= max(n, 1)

        model.eval()
        v_loss, vn = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                sc, sn, pk, fk, tz = [b.to(device) for b in batch]
                pred = model(sc, sn, pk, fk)
                v_loss += loss_fn(pred, tz).item() * sc.size(0)
                vn += sc.size(0)
        v_loss /= max(vn, 1)
        sched.step(v_loss)

        improved = v_loss < best_val
        if improved:
            best_val = v_loss
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if ep % 5 == 0 or ep == 1 or improved:
            mark = " *" if improved else ""
            print(
                f"  TFT ep {ep:3d}/{epochs}  train={s_loss:.5f}  "
                f"val={v_loss:.5f}  lr={opt.param_groups[0]['lr']:.1e}{mark}"
            )

        if wait >= patience:
            print(f"  Early stop ep {ep} (patience {patience})")
            break

    if best_st is not None:
        model.load_state_dict(best_st)
    return time.perf_counter() - t0


# ═══════════════════════════ TFT: инференс ═══════════════════════════

def infer_tft(model, series_data, short_rest_ids, lookback, horizon, device):
    model.eval()
    model.to(device)
    short_set = set(str(r) for r in short_rest_ids)
    rows = []
    infer_total = 0.0

    def _take_past_with_left_pad(arr2d, end_pos, win_len):
        st = end_pos - win_len
        if st >= 0:
            return arr2d[st:end_pos].astype(np.float32)
        need = -st
        pad = np.zeros((need, arr2d.shape[1]), dtype=np.float32)
        body = arr2d[:end_pos].astype(np.float32)
        return np.concatenate([pad, body], axis=0)

    with torch.no_grad():
        for sid in tqdm(series_data, desc="TFT infer"):
            if sid not in short_set:
                continue
            s = series_data[sid]
            T = len(s["target"])
            pos = int(s["train_end"])
            mu, sigma = float(s["mu"]), float(s["sigma"])

            while pos < T:
                ah = min(horizon, T - pos)
                pk = _take_past_with_left_pad(s["kf"], pos, lookback)

                if pos + horizon <= T:
                    fk = s["kf"][pos : pos + horizon].astype(np.float32)
                else:
                    fk = np.zeros((horizon, s["kf"].shape[1]), dtype=np.float32)
                    fk[:ah] = s["kf"][pos : pos + ah]
                    if ah > 0:
                        fk[ah:] = fk[ah - 1]

                b_sc = torch.from_numpy(s["scat"]).unsqueeze(0).to(device)
                b_sn = torch.from_numpy(s["snum"]).unsqueeze(0).to(device)
                b_pk = torch.from_numpy(pk).unsqueeze(0).to(device)
                b_fk = torch.from_numpy(fk).unsqueeze(0).to(device)

                t1 = time.perf_counter()
                pred_z = model(b_sc, b_sn, b_pk, b_fk)
                dt = time.perf_counter() - t1
                infer_total += dt

                yp = pred_z[0, :ah].cpu().numpy().astype(np.float32) * sigma + mu
                yp = np.clip(yp, a_min=0.0, a_max=None)
                yt = s["target"][pos : pos + ah]
                dt_arr = s["dates"][pos : pos + ah]

                tmp = pd.DataFrame({"date_dt": dt_arr, "y_true": yt, "y_pred": yp})
                tmp["rest_id"] = s["rest_id"]
                tmp["series_id"] = sid
                tmp["model"] = "tft"
                tmp["inference_time_sec"] = dt / max(ah, 1)
                rows.append(tmp)
                pos += horizon

    pred_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return pred_df, infer_total


# ═══════════════════════════ TFT: точка входа ═══════════════════════════

def run_tft_short(
    short_rest_ids,
    results,
    all_predictions,
    *,
    retrain=True,
    models_dir=MODELS_DIR,
    preds_dir=PREDS_DIR,
):
    print("\n--- TFT: cold-start Temporal Fusion Transformer ---")
    print(f"Device: {TFT_DEVICE}")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    df = pd.read_parquet(CLASSIC_PARQUET)
    df["date_dt"] = pd.to_datetime(df["date_dt"])
    test_start = test_start_from_df(df)

    short_set = set(int(x) for x in short_rest_ids)
    keep_mask = (df["date_dt"] >= test_start) | (~df["rest_id"].isin(short_set))
    df_tft = df.loc[keep_mask].copy()
    print("Подготовка данных TFT (cold-start: без истории target у целевых рядов)...")
    sd, cat_dims, n_series = prepare_tft_series(df_tft, test_start)
    print(f"Рядов: {n_series}, cat_dims: {cat_dims}")

    n_kf = len(TFT_KNOWN_FUTURE_COLS)
    n_sn = len(TFT_STATIC_NUM_COLS)

    for cfg in TFT_ARCH_CONFIGS:
        name = str(cfg["name"])
        lookback = int(cfg["lookback"])
        horizon = int(cfg["horizon"])
        d = int(cfg["d"])
        layers = int(cfg["lstm_layers"])
        heads = int(cfg["n_heads"])
        drop = float(cfg["drop"])
        slug = name.lower().replace(" ", "_").replace("/", "_")

        all_win = make_tft_windows(sd, lookback, horizon, TFT_TRAIN_STRIDE)
        if len(all_win) < 2:
            print(f"TFT [{name}]: слишком мало окон ({len(all_win)}), пропуск")
            continue
        np.random.shuffle(all_win)
        sp = max(1, min(int(len(all_win) * (1 - TFT_VAL_FRAC)), len(all_win) - 1))
        tr_win, va_win = all_win[:sp], all_win[sp:]
        print(f"\n[TFT] Конфиг: {name} | lookback={lookback} horizon={horizon} d={d} layers={layers} heads={heads} drop={drop}")
        print(f"[TFT] Окон: train={len(tr_win)}, val={len(va_win)}")

        tr_ds = TFTWindowDataset(sd, tr_win, lookback, horizon)
        va_ds = TFTWindowDataset(sd, va_win, lookback, horizon)
        kw = dict(num_workers=0, pin_memory=(TFT_DEVICE == "cuda"))
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=TFT_BATCH, shuffle=True, **kw)
        va_loader = torch.utils.data.DataLoader(va_ds, batch_size=TFT_BATCH, shuffle=False, **kw)

        model = TFTModel(
            cat_dims=cat_dims, n_snum=n_sn, n_kf=n_kf,
            lookback=lookback, horizon=horizon,
            d=d, lstm_layers=layers, n_heads=heads, drop=drop,
        )
        print(f"[TFT] Параметров [{name}]: {sum(p.numel() for p in model.parameters()):,}")

        mpath = models_dir / f"tft_coldstart_2_{slug}.pt"
        loaded = False
        if (not retrain) and mpath.exists():
            print(f"[TFT] Загружаю сохранённые веса: {mpath.name}")
            try:
                try:
                    st = torch.load(mpath, map_location=TFT_DEVICE, weights_only=True)
                except TypeError:
                    st = torch.load(mpath, map_location=TFT_DEVICE)
                model.load_state_dict(st)
                loaded = True
                train_time = 0.0
            except Exception as exc:
                print(f"[TFT] Сохр. веса несовместимы с cold-start архитектурой, переобучаю: {exc}")
        if not loaded:
            print(f"[TFT] Обучение [{name}]...")
            train_time = train_tft(
                model, tr_loader, va_loader,
                TFT_EPOCHS, TFT_LR, TFT_GRAD_CLIP, TFT_PATIENCE, TFT_DEVICE,
            )
            torch.save(model.state_dict(), mpath)
            print(f"[TFT] Обучение [{name}] завершено за {train_time:.1f} с")

        pred_df, infer_total = infer_tft(
            model, sd, short_rest_ids, lookback, horizon, TFT_DEVICE,
        )
        if len(pred_df) == 0:
            print(f"TFT [{name}]: нет прогнозов по коротким рядам")
            continue
        yt_abs = float(np.nanmean(np.abs(pd.to_numeric(pred_df["y_true"], errors="coerce"))))
        yp_abs = float(np.nanmean(np.abs(pd.to_numeric(pred_df["y_pred"], errors="coerce"))))
        print(f"[TFT] {name} sanity: mean|y_true|={yt_abs:.4f} mean|y_pred|={yp_abs:.4f}")

        pred_df["model"] = f"tft/{name}"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(preds_dir / f"coldstart_2_tft_predictions_{slug}.parquet", index=False)
        log_short_result(results, all_predictions, f"TFT/{name}", pred_df, train_time, infer_total)
        print(
            f"TFT [{name}] — RMSE:",
            results[-1]["rmse"],
            "WAPE:",
            f"{results[-1]['wape']:.2f}%",
            f"инференс: {infer_total:.1f} с",
        )


def run_xgboost_disease_short_snapshot(
    short_ids,
    test_start_snap,
    results,
    all_predictions,
    *,
    retrain=True,
    models_dir=MODELS_DIR,
    preds_dir=PREDS_DIR,
):
    if not DISEASE_SNAPSHOT_PARQUET.exists():
        print(f"Disease XGBoost: нет snapshot parquet {DISEASE_SNAPSHOT_PARQUET}, пропуск.")
        return
    df = pd.read_parquet(DISEASE_SNAPSHOT_PARQUET).copy()
    if "snapshot_dt" not in df.columns or "date_dt" not in df.columns or "target" not in df.columns:
        print("Disease XGBoost: в snapshot parquet нужны snapshot_dt/date_dt/target, пропуск.")
        return
    df["snapshot_dt"] = pd.to_datetime(df["snapshot_dt"])
    df["date_dt"] = coerce_year_week_to_datetime(df["date_dt"])
    short_set = set(int(x) for x in short_ids)
    train_b = df[(df["snapshot_dt"] < test_start_snap) & (~df["rest_id"].isin(short_set))].copy()
    test_b = df[(df["snapshot_dt"] >= test_start_snap) & (df["rest_id"].isin(short_set))].copy()
    if len(train_b) == 0 or len(test_b) == 0:
        print("Disease XGBoost: пустые train/test после split, пропуск.")
        return
    test_dates = set(test_b["date_dt"].dropna().unique().tolist())
    train_b = train_b[~train_b["date_dt"].isin(test_dates)].copy()
    drop_cols = [c for c in ["target", "date_dt", "rest_id", "snapshot_dt", "city", "age_group"] if c in df.columns]
    feature_cols = [c for c in train_b.columns if c not in drop_cols]
    if not feature_cols:
        print("Disease XGBoost: нет фичей после drop_cols, пропуск.")
        return

    def _fill_xy(tb):
        X = tb[feature_cols].copy()
        y = tb["target"].copy()
        for c in X.columns:
            if X[c].dtype in ("float64", "float32") or X[c].isna().any():
                med = X[c].median()
                X[c] = X[c].fillna(med)
        return X, y

    val_start = pd.Timestamp(test_start_snap) - pd.DateOffset(weeks=8)
    tr_fit = train_b[train_b["snapshot_dt"] < val_start].copy()
    val_b = train_b[train_b["snapshot_dt"] >= val_start].copy()
    if len(tr_fit) < 50 or len(val_b) < 10:
        q = train_b["snapshot_dt"].quantile(0.8)
        tr_fit = train_b[train_b["snapshot_dt"] < q].copy()
        val_b = train_b[train_b["snapshot_dt"] >= q].copy()
    X_tr, y_tr = _fill_xy(tr_fit)
    X_va, y_va = _fill_xy(val_b)
    X_all, y_all = _fill_xy(train_b)
    X_te, y_te = _fill_xy(test_b)

    path_model = models_dir / "xgboost_coldstart_2_disease.pkl"
    path_params = models_dir / "xgboost_coldstart_2_disease_params.json"
    best_params = None
    if (not retrain) and path_params.exists():
        try:
            with open(path_params, encoding="utf-8") as f:
                best_params = json.load(f)
        except Exception:
            best_params = None
    if best_params is None:
        def objective(trial):
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
            p = m.predict(X_va)
            return float(np.sqrt(mean_squared_error(np.asarray(y_va, dtype=float), p)))
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
        st.optimize(objective, n_trials=OPTUNA_N_TRIALS, show_progress_bar=False)
        best_params = st.best_params
        with open(path_params, "w", encoding="utf-8") as f:
            json.dump(best_params, f, ensure_ascii=False, indent=2)

    t0 = time.perf_counter()
    if path_model.exists() and not retrain:
        with open(path_model, "rb") as f:
            model = pickle.load(f)
    else:
        model = XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
        model.fit(X_all, y_all)
        with open(path_model, "wb") as f:
            pickle.dump(model, f)
    train_time = time.perf_counter() - t0
    p0 = time.perf_counter()
    y_hat = model.predict(X_te)
    inf_time = time.perf_counter() - p0
    pred_df = test_b[["rest_id", "date_dt"]].copy()
    pred_df["series_id"] = pred_df["rest_id"].astype(str)
    pred_df["y_true"] = y_te.values
    pred_df["y_pred"] = y_hat
    pred_df["model"] = "xgboost_disease"
    pred_df["inference_time_sec"] = inf_time / max(len(pred_df), 1)
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(preds_dir / "coldstart_2_xgboost_predictions_disease.parquet", index=False)
    infer_sum = float(pred_df["inference_time_sec"].sum()) if len(pred_df) else 0.0
    log_short_result(results, all_predictions, "XGBoost(disease)", pred_df, train_time, infer_sum)


def run_tft_disease_short(
    train_df,
    test_df,
    short_ids,
    results,
    all_predictions,
    *,
    retrain=True,
    models_dir=MODELS_DIR,
    preds_dir=PREDS_DIR,
):
    # Упрощенный TFT-проход по 3 архитектурам для disease (weekly).
    if len(train_df) == 0 or len(test_df) == 0:
        return
    full = pd.concat([train_df.copy(), test_df.copy()], ignore_index=True).sort_values(["rest_id", "date_dt"])
    full["checks_cnt"] = full["target"].astype(float)
    full["weekday_num"] = full["date_dt"].dt.weekday
    full["month_num"] = full["date_dt"].dt.month
    full["week_num"] = full["date_dt"].dt.isocalendar().week.astype(int)
    full["day_num"] = full["date_dt"].dt.day
    full["year_num"] = full["date_dt"].dt.year
    for c in TFT_KNOWN_FUTURE_COLS:
        if c not in full.columns:
            full[c] = 0
    for c in TFT_OBSERVED_PAST_COLS:
        if c not in full.columns:
            full[c] = 0.0
    for c in TFT_STATIC_CAT_COLS:
        if c not in full.columns:
            full[c] = "__na__"
    for c in TFT_STATIC_NUM_COLS:
        if c not in full.columns:
            full[c] = 0.0
    test_start = pd.Timestamp(test_df["date_dt"].min())
    short_set = set(int(x) for x in short_ids)
    keep_mask = (full["date_dt"] >= test_start) | (~full["rest_id"].isin(short_set))
    full_cold = full.loc[keep_mask].copy()
    sd, cat_dims, n_series = prepare_tft_series(full_cold, test_start)
    for cfg in TFT_ARCH_CONFIGS:
        wins = make_tft_windows(sd, int(cfg["lookback"]), int(cfg["horizon"]), TFT_TRAIN_STRIDE)
        if len(wins) < 2:
            continue
        np.random.shuffle(wins)
        sp = max(1, min(int(len(wins) * (1 - TFT_VAL_FRAC)), len(wins) - 1))
        tr_ds = TFTWindowDataset(sd, wins[:sp], int(cfg["lookback"]), int(cfg["horizon"]))
        va_ds = TFTWindowDataset(sd, wins[sp:], int(cfg["lookback"]), int(cfg["horizon"]))
        kw = dict(num_workers=0, pin_memory=(TFT_DEVICE == "cuda"))
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=TFT_BATCH, shuffle=True, **kw)
        va_loader = torch.utils.data.DataLoader(va_ds, batch_size=TFT_BATCH, shuffle=False, **kw)
        m = TFTModel(
            cat_dims=cat_dims, n_snum=len(TFT_STATIC_NUM_COLS), n_kf=len(TFT_KNOWN_FUTURE_COLS),
            lookback=int(cfg["lookback"]), horizon=int(cfg["horizon"]),
            d=int(cfg["d"]), lstm_layers=int(cfg["lstm_layers"]), n_heads=int(cfg["n_heads"]), drop=float(cfg["drop"]),
        )
        slug = str(cfg["name"]).lower().replace(" ", "_")
        p = models_dir / f"tft_coldstart_2_disease_{slug}.pt"
        loaded = False
        if (not retrain) and p.exists():
            try:
                st = torch.load(p, map_location=TFT_DEVICE)
                m.load_state_dict(st)
                loaded = True
                train_time = 0.0
            except Exception as exc:
                print(f"[TFT][disease] Сохр. веса несовместимы, переобучаю: {exc}")
        if not loaded:
            train_time = train_tft(m, tr_loader, va_loader, TFT_EPOCHS, TFT_LR, TFT_GRAD_CLIP, TFT_PATIENCE, TFT_DEVICE)
            torch.save(m.state_dict(), p)
        pred_df, infer_total = infer_tft(m, sd, short_ids, int(cfg["lookback"]), int(cfg["horizon"]), TFT_DEVICE)
        if len(pred_df) == 0:
            continue
        pred_df["model"] = f"tft_disease_{slug}"
        preds_dir.mkdir(parents=True, exist_ok=True)
        pred_df.to_parquet(preds_dir / f"coldstart_2_tft_predictions_disease_{slug}.parquet", index=False)
        log_short_result(results, all_predictions, f"disease/{cfg['name']}", pred_df, train_time, infer_total)


# ═══════════════════════════════════════════════════════════════════

def drop_weather_from_classic(df):
    cols = [c for c in WEATHER_COLS if c in df.columns]
    return df.drop(columns=cols, errors="ignore")


setup_dirs = lambda: short_common.setup_dirs(MODELS_DIR, PREDS_DIR, METRICS_DIR)
task_artifact_paths = lambda task_key: short_common.task_artifact_paths(ARTIFACTS_DIR, task_key)
setup_task_dirs = short_common.setup_task_dirs
rmse = short_common.rmse
wape = short_common.wape
compute_test_start_by_ratio = short_common.compute_test_start_by_ratio
select_shortest_share_ids = short_common.select_shortest_share_ids
coerce_year_week_to_datetime = short_common.coerce_year_week_to_datetime
rest_open_days_on_first_test_day = short_common.rest_open_days_on_first_test_day
load_classic_split = short_common.load_classic_split
drop_weather_from_classic = lambda df: short_common.drop_weather_from_classic(df, WEATHER_COLS)


def truncate_disease_train_random_weeks(train_df, short_ids, min_weeks=50, max_weeks=150, seed=RANDOM_SEED):
    return short_common.truncate_disease_train_random_weeks(
        train_df=train_df,
        short_ids=short_ids,
        min_weeks=min_weeks,
        max_weeks=max_weeks,
        seed=seed,
    )


def select_short_rest_ids(open_days_df):
    return short_common.select_short_rest_ids(
        open_days_df=open_days_df,
        min_open_days=SHORT_OPEN_DAYS_MIN,
        max_open_days=SHORT_OPEN_DAYS_MAX,
    )


def run_short_experiment(
    do_xgb=True,
    do_tft=True,
    do_disease=True,
    run_traffic_task=True,
    run_disease_task=True,
    show_plot=False,
    retrain: bool | None = None,
):
    traffic_paths = task_artifact_paths("traffic")
    disease_paths = task_artifact_paths("disease")
    print("\n" + "#" * 70)
    print("PIPELINE START: short baseline")
    print(
        f"Флаги: retrain={RETRAIN if retrain is None else bool(retrain)}, "
        f"run_traffic_task={run_traffic_task}, run_disease_task={run_disease_task}, do_disease={do_disease}"
    )
    print(f"Traffic artifacts: {traffic_paths['base_dir']}")
    print(f"Disease artifacts: {disease_paths['base_dir']}")
    print("#" * 70)
    if run_traffic_task:
        setup_task_dirs(traffic_paths)
        print(f"[INIT] Подготовлены папки traffic: models={traffic_paths['models_dir']}, preds={traffic_paths['preds_dir']}, metrics={traffic_paths['metrics_dir']}")
    if do_disease and run_disease_task:
        setup_task_dirs(disease_paths)
        print(f"[INIT] Подготовлены папки disease: models={disease_paths['models_dir']}, preds={disease_paths['preds_dir']}, metrics={disease_paths['metrics_dir']}")
    _retrain = RETRAIN if retrain is None else bool(retrain)
    results_traffic = []
    preds_traffic = []
    results_disease = []
    preds_disease = []

    if run_traffic_task:
        print("\n[TRAFFIC] Этап 1/4: загрузка и подготовка данных")
        df_c = pd.read_parquet(CLASSIC_PARQUET)
        df_c["date_dt"] = pd.to_datetime(df_c["date_dt"])
        df_c = drop_weather_from_classic(df_c)

        test_start = test_start_from_df(df_c)
        train_df, test_df = load_classic_split(df_c, test_start)

        od = rest_open_days_on_first_test_day(df_c, test_start)
        short_ids = select_short_rest_ids(od)

        n_all_rest = df_c["rest_id"].nunique()
        print("=" * 60)
        print("Специфика: короткие временные ряды")
        print(
            f"Порог open_days: ({SHORT_OPEN_DAYS_MIN}, {SHORT_OPEN_DAYS_MAX}) "
            f"на первый день теста {test_start.date()}"
        )
        print(f"Всего ресторанов в классике: {n_all_rest}")
        print(f"Коротких рядов (отбор): {len(short_ids)}")
        if short_ids:
            preview = short_ids[:25]
            more = " ..." if len(short_ids) > 25 else ""
            print(f"rest_id (первые 25): {preview}{more}")
        print(f"Train: {train_df.shape} | Test: {test_df.shape}")
        print("=" * 60)

        if len(short_ids) == 0:
            print("Traffic: нет рядов, удовлетворяющих условию — пропуск задачи.")
        else:
            short_in_test = test_df[test_df["rest_id"].isin(short_ids)]["rest_id"].nunique()
            print(f"Коротких ресторанов с данными в тесте: {short_in_test}")
            print("[TRAFFIC] Этап 2/4: запуск моделей")

            if do_xgb:
                print("[TRAFFIC][MODEL] XGBoost: старт")
                run_xgboost_all_eval_short(
                    short_ids, test_start, results_traffic, preds_traffic,
                    retrain=_retrain,
                    models_dir=traffic_paths["models_dir"], preds_dir=traffic_paths["preds_dir"]
                )
                print("[TRAFFIC][MODEL] XGBoost: завершен")
            if do_tft:
                print("[TRAFFIC][MODEL] TFT: старт")
                run_tft_short(
                    short_ids, results_traffic, preds_traffic,
                    retrain=_retrain,
                    models_dir=traffic_paths["models_dir"], preds_dir=traffic_paths["preds_dir"]
                )
                print("[TRAFFIC][MODEL] TFT: завершен")
            print("[TRAFFIC] Этап 3/4: модели завершены")

    if do_disease and run_disease_task:
        print("\n[DISEASE] Этап 1/4: загрузка и подготовка данных")
        print("\n" + "=" * 60)
        print("Специфика short для disease: 10% самых коротких + обрезка train до 50..150 недель")
        df_d = load_disease_classic_long()
        df_d["date_dt"] = pd.to_datetime(df_d["date_dt"])
        test_start_d = compute_test_start_by_ratio(df_d, "date_dt", train_ratio=0.70, val_ratio=0.15)
        train_d = df_d[df_d["date_dt"] < test_start_d].copy()
        test_d = df_d[df_d["date_dt"] >= test_start_d].copy()
        short_d = select_shortest_share_ids(train_d, share=DISEASE_SHORT_SHARE, id_col="rest_id")
        train_d_cut = truncate_disease_train_random_weeks(
            train_d,
            short_d,
            min_weeks=DISEASE_MIN_WEEKS,
            max_weeks=DISEASE_MAX_WEEKS,
            seed=RANDOM_SEED,
        )
        snap_start_d = compute_test_start_by_ratio(
            pd.read_parquet(DISEASE_SNAPSHOT_PARQUET) if DISEASE_SNAPSHOT_PARQUET.exists() else train_d,
            "snapshot_dt" if DISEASE_SNAPSHOT_PARQUET.exists() else "date_dt",
            train_ratio=0.70,
            val_ratio=0.15,
        )
        if do_xgb:
            print("[DISEASE][MODEL] XGBoost(snapshot): старт")
            run_xgboost_disease_short_snapshot(
                short_d, snap_start_d, results_disease, preds_disease, retrain=_retrain,
                models_dir=disease_paths["models_dir"], preds_dir=disease_paths["preds_dir"]
            )
            print("[DISEASE][MODEL] XGBoost(snapshot): завершен")
        if do_tft:
            print("[DISEASE][MODEL] TFT: старт")
            run_tft_disease_short(
                train_d_cut, test_d, short_d, results_disease, preds_disease, retrain=_retrain,
                models_dir=disease_paths["models_dir"], preds_dir=disease_paths["preds_dir"]
            )
            print("[DISEASE][MODEL] TFT: завершен")
        print("[DISEASE] Этап 3/4: модели завершены")

    if run_traffic_task:
        print("[TRAFFIC] Этап 4/4: сохранение метрик и графиков")
        dfm_traffic = save_metrics_files(
            results_traffic, f"{SPEC_KEY}_traffic", metrics_dir=traffic_paths["metrics_dir"]
        )
        fig_path_traffic = traffic_paths["metrics_dir"] / f"{SPEC_KEY}_traffic_metrics_plot.png"
        plot_short_metrics_dashboard(
            dfm_traffic,
            title="Специфика: короткие ряды (traffic: RMSE, WAPE, время)",
            save_path=fig_path_traffic,
            show=show_plot,
        )
        print("\nTraffic метрики сохранены:", traffic_paths["metrics_dir"] / f"{SPEC_KEY}_traffic_metrics.csv")
        print(f"Traffic predictions dir: {traffic_paths['preds_dir']}")
        print(f"Traffic models dir: {traffic_paths['models_dir']}")

    if do_disease and run_disease_task:
        print("[DISEASE] Этап 4/4: сохранение метрик и графиков")
        dfm_disease = save_metrics_files(
            results_disease, f"{SPEC_KEY}_disease", metrics_dir=disease_paths["metrics_dir"]
        )
        fig_path_disease = disease_paths["metrics_dir"] / f"{SPEC_KEY}_disease_metrics_plot.png"
        plot_short_metrics_dashboard(
            dfm_disease,
            title="Специфика: короткие ряды (disease: RMSE, WAPE, время)",
            save_path=fig_path_disease,
            show=show_plot,
        )
        print("Disease метрики сохранены:", disease_paths["metrics_dir"] / f"{SPEC_KEY}_disease_metrics.csv")
        print(f"Disease predictions dir: {disease_paths['preds_dir']}")
        print(f"Disease models dir: {disease_paths['models_dir']}")

    print("#" * 70)
    print("PIPELINE END: short baseline")
    print("#" * 70)

    if run_traffic_task and (do_disease and run_disease_task):
        return {"traffic": dfm_traffic, "disease": dfm_disease}, {"traffic": preds_traffic, "disease": preds_disease}
    if run_traffic_task:
        return dfm_traffic, preds_traffic
    if do_disease and run_disease_task:
        return dfm_disease, preds_disease
    return pd.DataFrame(), []


if __name__ == "__main__":
    run_short_experiment(
        retrain=RETRAIN,
        run_traffic_task=RUN_TRAFFIC_TASK,
        run_disease_task=RUN_DISEASE_TASK,
        show_plot=False,
    )