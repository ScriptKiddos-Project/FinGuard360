# scripts/train_model.py
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as SK_VERSION
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# Modelos
import lightgbm as lgb
import category_encoders as ce

# === Optuna (tuning opcional) ===
try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

# === SHAP (opcional) ===
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False


# ------------------------------
# Utilitários
# ------------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ks_stat(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    order = np.argsort(y_prob)
    y = y_true[order]
    cum_pos = np.cumsum(y == 1) / max(1, np.sum(y == 1))
    cum_neg = np.cumsum(y == 0) / max(1, np.sum(y == 0))
    return float(np.max(np.abs(cum_pos - cum_neg)))


def ks_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ks_vals = tpr - fpr
    return thr, ks_vals


@dataclass
class TrainConfig:
    data_dir: Path
    artifacts_dir: Path
    seed: int = 42
    n_folds: int = 5
    quick: bool = False
    tune_trials: int = 0
    target_col: str = "TARGET"
    id_col: str = "SK_ID_CURR"
    ohe_max_unique: int = 15
    te_min_unique: int = 30


# ------------------------------
# 1) Dados e colunas
# ------------------------------
def load_train(cfg: TrainConfig) -> pd.DataFrame:
    path = cfg.data_dir / "application_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Não encontrei {path}. Coloque os dados em {cfg.data_dir}.")
    df = pd.read_csv(path)
    return df


def split_columns(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[List[str], List[str], List[str]]:
    ignore = {cfg.target_col, cfg.id_col}
    cols = [c for c in df.columns if c not in ignore]

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    nunq = df[cat_cols].nunique(dropna=True)
    cat_low = [c for c in cat_cols if nunq[c] <= cfg.ohe_max_unique]
    cat_high = [c for c in cat_cols if nunq[c] >= cfg.te_min_unique]
    mid = [c for c in cat_cols if c not in cat_low and c not in cat_high]
    cat_low += mid  # mid vai para OHE também

    return num_cols, cat_low, cat_high


# ------------------------------
# 2) Preprocessador
# ------------------------------
def _make_ohe() -> OneHotEncoder:
    """Cria OHE compatível com scikit-learn <1.2 e >=1.2."""
    if SK_VERSION >= "1.2":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(num_cols: List[str], cat_low: List[str], cat_high: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(with_centering=False)),
        ]
    )
    ohe_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", _make_ohe()),
        ]
    )
    te_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("te", ce.TargetEncoder(handle_missing="value", handle_unknown="value", smoothing=1.0)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat_low", ohe_pipe, cat_low),
            ("cat_high", te_pipe, cat_high),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # saída densa
    )
    return pre


def get_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_low: List[str], cat_high: List[str]) -> List[str]:
    names: List[str] = []
    names += list(num_cols)
    try:
        ohe = pre.named_transformers_["cat_low"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(cat_low).tolist()
    except Exception:
        ohe_names = [f"{c}_ohe" for c in cat_low]
    names += ohe_names
    names += list(cat_high)  # TE: 1 coluna por variável
    return names


# ------------------------------
# 3) Treino (CV + calibrador + final)
# ------------------------------
def train_with_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: TrainConfig,
    num_cols: List[str],
    cat_low: List[str],
    cat_high: List[str],
) -> Tuple[ColumnTransformer, lgb.LGBMClassifier, IsotonicRegression, Dict]:
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    oof_raw = np.zeros(len(y), dtype=float)
    fold_reports = []

    base_params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000 if not cfg.quick else 400,
        learning_rate=0.03 if not cfg.quick else 0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=-1,
        num_leaves=64,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=-1,
    )

    if cfg.tune_trials > 0 and HAS_OPTUNA:
        def objective(trial: "optuna.Trial"):
            params = base_params.copy()
            params.update(
                dict(
                    num_leaves=trial.suggest_int("num_leaves", 31, 255),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
                )
            )
            aucs = []
            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y[tr_idx], y[va_idx]

                pre = make_preprocessor(num_cols, cat_low, cat_high)
                X_tr_t = pre.fit_transform(X_tr, y_tr)
                X_va_t = pre.transform(X_va)

                mdl = lgb.LGBMClassifier(**params)
                mdl.fit(
                    X_tr_t, y_tr,
                    eval_set=[(X_va_t, y_va)],
                    callbacks=[lgb.early_stopping(200, verbose=False)],
                )
                proba = mdl.predict_proba(X_va_t)[:, 1]
                aucs.append(roc_auc_score(y_va, proba))
            return float(np.mean(aucs))

        study = optuna.create_study(direction="maximize")
        n_trials = cfg.tune_trials if not cfg.quick else max(1, cfg.tune_trials // 2)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = base_params.copy()
        best_params.update(study.best_params)
    else:
        best_params = base_params

    # CV para OOF e métricas
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pre = make_preprocessor(num_cols, cat_low, cat_high)
        X_tr_t = pre.fit_transform(X_tr, y_tr)
        X_va_t = pre.transform(X_va)

        mdl = lgb.LGBMClassifier(**best_params)
        mdl.fit(
            X_tr_t, y_tr,
            eval_set=[(X_va_t, y_va)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        proba = mdl.predict_proba(X_va_t)[:, 1]
        oof_raw[va_idx] = proba

        auc = roc_auc_score(y_va, proba)
        ks = ks_stat(y_va, proba)
        fold_reports.append({"fold": fold, "AUC": auc, "KS": ks})

    # Calibrador em OOF
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_raw, y)

    # Modelo final no full
    pre_full = make_preprocessor(num_cols, cat_low, cat_high)
    X_full_t = pre_full.fit_transform(X, y)
    mdl_full = lgb.LGBMClassifier(**best_params)
    mdl_full.fit(X_full_t, y)

    report = {
        "folds": fold_reports,
        "oof_auc": roc_auc_score(y, oof_raw),
        "oof_ks": ks_stat(y, oof_raw),
        "params": best_params,
    }
    return pre_full, mdl_full, calibrator, report


# ------------------------------
# 4) Gráficos: reliability + SHAP
# ------------------------------
def plot_reliability(y_true: np.ndarray, y_prob_raw: np.ndarray, calibrator: IsotonicRegression, out_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    y_prob_cal = calibrator.predict(y_prob_raw)
    df = pd.DataFrame({"y": y_true, "raw": y_prob_raw, "cal": y_prob_cal})
    df["bin"] = pd.qcut(df["raw"], q=10, duplicates="drop")
    grp = df.groupby("bin")
    x_raw = grp["raw"].mean().values
    y_obs = grp["y"].mean().values
    x_cal = grp["cal"].mean().values

    bs_raw = brier_score_loss(df["y"], df["raw"])
    bs_cal = brier_score_loss(df["y"], df["cal"])

    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfeito")
    plt.plot(x_raw, y_obs, "o-", label=f"Raw (Brier={bs_raw:.3f})")
    plt.plot(x_cal, y_obs, "o-", label=f"Calibrated (Brier={bs_cal:.3f})")
    plt.xlabel("Probabilidade prevista (média no bin)")
    plt.ylabel("Fração observada (default rate)")
    plt.title("Reliability curve (quantile bins)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(df["raw"], bins=30, alpha=0.6, label="Raw")
    plt.hist(df["cal"], bins=30, alpha=0.6, label="Calibrated")
    plt.xlabel("Probability of default")
    plt.ylabel("Count")
    plt.title("PD distribution — raw vs calibrated")
    plt.legend()

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_shap_top(model: lgb.LGBMClassifier, X_t: np.ndarray, feature_names: List[str], out_bar: Path, out_bee: Path, max_display: int = 20):
    if not HAS_SHAP:
        return
    try:
        import matplotlib.pyplot as plt
        explainer = shap.TreeExplainer(model)
        n = min(10000, X_t.shape[0])
        idx = np.random.choice(X_t.shape[0], size=n, replace=False)
        Xs = X_t[idx]
        sv = explainer(Xs, check_additivity=False)

        fig = plt.figure(figsize=(6, 8))
        shap.plots.bar(sv, max_display=max_display, show=False)
        plt.tight_layout()
        out_bar.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_bar, dpi=160)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 10))
        shap.plots.beeswarm(sv, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(out_bee, dpi=160)
        plt.close(fig)
    except Exception:
        pass


# ------------------------------
# 5) Bands e threshold
# ------------------------------
def make_bands_and_threshold(y_true: np.ndarray, y_prob_cal: np.ndarray) -> Tuple[Dict, Dict]:
    thr, ks_vals = ks_curve(y_true, y_prob_cal)
    best_idx = int(np.argmax(ks_vals))
    best_thr = float(thr[best_idx])
    best_ks = float(ks_vals[best_idx])
    auc = float(roc_auc_score(y_true, y_prob_cal))

    y_hat = (y_prob_cal >= best_thr).astype(int)
    TP = int(((y_hat == 1) & (y_true == 1)).sum())
    FP = int(((y_hat == 1) & (y_true == 0)).sum())
    TN = int(((y_hat == 0) & (y_true == 0)).sum())
    FN = int(((y_hat == 0) & (y_true == 1)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    approval_rate = float((y_hat == 0).mean())
    base_rate = float(np.mean(y_true))

    threshold = {
        "type": "ks_max",
        "value": best_thr,
        "KS": best_ks,
        "metrics_at_threshold": {
            "threshold": best_thr,
            "KS": best_ks,
            "AUC": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
            "approval_rate": approval_rate,
            "base_rate": base_rate,
        },
    }

    qs = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]
    qv = np.quantile(y_prob_cal, qs).tolist()
    edges = [-math.inf] + qv + [math.inf]
    names = ["Very Low", "Low", "Medium", "Elevated", "High", "Very High"]
    schema = [{"name": names[i], "lower": float(edges[i]), "upper": float(edges[i + 1])} for i in range(len(names))]
    bands = {"schema": schema, "quantiles": {str(q): float(v) for q, v in zip(qs, qv)}}
    return threshold, bands


# ------------------------------
# 6) Persistência
# ------------------------------
def save_artifacts(
    cfg: TrainConfig,
    pre: ColumnTransformer,
    model: lgb.LGBMClassifier,
    calibrator: IsotonicRegression,
    feature_names: List[str],
    raw_schema: List[str],
    threshold: Dict,
    bands: Dict,
    oof_raw: np.ndarray,
    y: np.ndarray,
):
    art = cfg.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)

    joblib.dump(pre, art / "preprocessor.joblib")
    joblib.dump(model, art / "lgbm_model.joblib")
    joblib.dump(calibrator, art / "isotonic.joblib")

    pd.DataFrame({"column": feature_names}).to_csv(art / "train_columns.csv", index=False)

    meta = {
        "threshold": threshold,
        "bands": bands,
        "base_rate": float(np.mean(y)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "schema": {
            "raw_schema": raw_schema,
            "feature_names": feature_names,
        },
    }
    with open(art / "thresholds_bands.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    plot_reliability(y, oof_raw, calibrator, art / "reliability_curve.png")


# ------------------------------
# 7) Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Treino do pipeline de Credit Risk (LightGBM + Isotonic).")
    parser.add_argument("--data-dir", type=str, default=str(Path("data/raw").resolve()), help="Diretório dos CSVs brutos.")
    parser.add_argument("--artifacts-dir", type=str, default=str(Path("artifacts").resolve()), help="Saída dos artefatos.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Modo rápido (menos árvores/sem tuning).")
    parser.add_argument("--tune-trials", type=int, default=0, help="Trials do Optuna; 0 desliga.")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=Path(args.data_dir),
        artifacts_dir=Path(args.artifacts_dir),
        seed=args.seed,
        n_folds=args.folds,
        quick=bool(args.quick),
        tune_trials=int(args.tune_trials),
    )

    print("Config:", cfg, "| scikit-learn:", SK_VERSION)
    set_seed(cfg.seed)

    df = load_train(cfg)
    y = df[cfg.target_col].astype(int).values
    num_cols, cat_low, cat_high = split_columns(df, cfg)

    raw_schema = num_cols + cat_low + cat_high
    X = df[raw_schema].copy()

    print(f"N={len(df)} | num={len(num_cols)} | cat_low={len(cat_low)} | cat_high={len(cat_high)}")

    pre_full, mdl_full, calibrator, report = train_with_cv(X, y, cfg, num_cols, cat_low, cat_high)
    print("OOF AUC:", f"{report['oof_auc']:.4f}", "| OOF KS:", f"{report['oof_ks']:.4f}")

    feature_names = get_feature_names(pre_full, num_cols, cat_low, cat_high)

    # Reconstituir OOF bruto para plot/calib estável
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    oof_raw = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pre = make_preprocessor(num_cols, cat_low, cat_high)
        X_tr_t = pre.fit_transform(X_tr, y_tr)
        X_va_t = pre.transform(X_va)

        mdl = lgb.LGBMClassifier(**report["params"])
        mdl.fit(
            X_tr_t, y_tr,
            eval_set=[(X_va_t, y_va)],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof_raw[va_idx] = mdl.predict_proba(X_va_t)[:, 1]

    oof_cal = calibrator.predict(oof_raw)
    threshold, bands = make_bands_and_threshold(y, oof_cal)

    save_artifacts(
        cfg=cfg,
        pre=pre_full,
        model=mdl_full,
        calibrator=calibrator,
        feature_names=feature_names,
        raw_schema=raw_schema,
        threshold=threshold,
        bands=bands,
        oof_raw=oof_raw,
        y=y,
    )

    if HAS_SHAP:
        try:
            X_full_t = pre_full.transform(X)
            plot_shap_top(
                model=mdl_full,
                X_t=X_full_t,
                feature_names=feature_names,
                out_bar=cfg.artifacts_dir / "shap_bar_top20.png",
                out_bee=cfg.artifacts_dir / "shap_beeswarm_top.png",
            )
            explainer = shap.TreeExplainer(mdl_full)
            n = min(5000, X_full_t.shape[0])
            idx = np.random.choice(X_full_t.shape[0], size=n, replace=False)
            sv = explainer(X_full_t[idx], check_additivity=False).values
            mean_abs = np.mean(np.abs(sv), axis=0)
            imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
            imp.to_csv(cfg.artifacts_dir / "shap_global_importance.csv", index=False)
        except Exception as e:
            print("Aviso: falha ao gerar SHAP:", e)

    print("\n=== Treino concluído ===")
    print("Artefatos salvos em:", cfg.artifacts_dir)
    print("Threshold por KS:", threshold["value"])
    print("OOF AUC:", f"{report['oof_auc']:.4f}", "| OOF KS:", f"{report['oof_ks']:.4f}")
    print("Params:", report["params"])


if __name__ == "__main__":
    main()
