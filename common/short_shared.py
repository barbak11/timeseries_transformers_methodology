from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def setup_dirs(models_dir: Path, preds_dir: Path, metrics_dir: Path) -> None:
    for d in (models_dir, preds_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)


def task_artifact_paths(artifacts_dir: Path, task_key: str) -> dict[str, Path]:
    base = artifacts_dir / str(task_key)
    return {
        "base_dir": base,
        "models_dir": base / "models",
        "preds_dir": base / "predictions",
        "metrics_dir": base / "metrics",
    }


def setup_task_dirs(paths: dict) -> None:
    for d in (paths["models_dir"], paths["preds_dir"], paths["metrics_dir"]):
        d.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(yt, yp)))


def wape(y_true, y_pred) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(yt))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(yt - yp)) / denom * 100)


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


def truncate_disease_train_random_weeks(train_df, short_ids, min_weeks=50, max_weeks=150, seed=42):
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


def rest_open_days_on_first_test_day(df, test_start):
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


def select_short_rest_ids(open_days_df, min_open_days: int, max_open_days: int):
    mask = (open_days_df["open_days"] > min_open_days) & (
        open_days_df["open_days"] < max_open_days
    )
    ids = open_days_df.loc[mask, "rest_id"].astype(int).unique().tolist()
    return sorted(ids)


def load_classic_split(df, test_start):
    df = df.sort_values(["rest_id", "date_dt"]).reset_index(drop=True)
    df["series_id"] = df["rest_id"].astype(str)
    train_df = df[df["date_dt"] < test_start].copy()
    test_df = df[df["date_dt"] >= test_start].copy()
    return train_df, test_df


def drop_weather_from_classic(df, weather_cols):
    cols = [c for c in weather_cols if c in df.columns]
    return df.drop(columns=cols, errors="ignore")
