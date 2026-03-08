# scripts/preprocess.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # raiz do repo
ART = ROOT / "artifacts"

REQUIRED = [
    "preprocessor.joblib",
    "lgbm_model.joblib",
    "isotonic.joblib",
    "train_columns.csv",          # nomes das features transformadas (ideal)
    "thresholds_bands.json",      # thresholds + bands + schema raw
]


def _assert_artifacts(art_dir: Path = ART) -> None:
    missing = [f for f in REQUIRED if not (art_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {art_dir}: {missing}. "
            "Treine/extraia os artefatos (notebook ou train_model.py) "
            "ou copie a pasta artifacts correta."
        )


def _load_feature_names(art_dir: Path) -> List[str]:
    """
    Lê train_columns.csv contendo os NOMES DAS FEATURES TRANSFORMADAS
    (saída do preprocessor) para alinhar no predict_proba.
    Aceita CSV com uma coluna (lista) ou multi-colunas (usa cabeçalho).
    """
    df = pd.read_csv(art_dir / "train_columns.csv")
    if df.shape[1] == 1:
        col = df.columns[0]
        return df[col].astype(str).tolist()
    return df.columns.astype(str).tolist()


def _load_raw_schema(bands_meta: Dict[str, Any]) -> List[str]:
    """
    Recupera o schema bruto (lista de colunas de entrada antes do preprocessor)
    a partir de thresholds_bands.json (chave 'schema.raw_schema').
    """
    schema = bands_meta.get("schema", {})
    raw_schema = schema.get("raw_schema")
    if not raw_schema or not isinstance(raw_schema, list):
        raise KeyError(
            "Não encontrei 'schema.raw_schema' em thresholds_bands.json. "
            "Inclua a lista de colunas brutas usadas no treino em "
            "bands_meta['schema']['raw_schema']."
        )
    return [str(c) for c in raw_schema]


def load_artifacts(art_dir: Path = ART) -> Dict[str, Any]:
    """
    Carrega artefatos para scoring:
      - preprocessor, model, calibrator
      - feature_names (transformadas)
      - raw_schema (brutas de entrada)
      - bands_meta (thresholds e bandas)
    """
    _assert_artifacts(art_dir)

    preprocessor = joblib.load(art_dir / "preprocessor.joblib")
    model = joblib.load(art_dir / "lgbm_model.joblib")
    calibrator = joblib.load(art_dir / "isotonic.joblib")

    feature_names = _load_feature_names(art_dir)

    with open(art_dir / "thresholds_bands.json", "r", encoding="utf-8") as f:
        bands_meta = json.load(f)

    raw_schema = _load_raw_schema(bands_meta)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "calibrator": calibrator,
        "feature_names": feature_names,  # transformadas (idealmente == X_t.shape[1])
        "raw_schema": raw_schema,        # brutas
        "bands_meta": bands_meta,
    }


def assign_band(pd_value: float, bands_schema: List[Dict[str, Any]]) -> str:
    """Retorna o nome da banda conforme limites (lower/upper)."""
    for b in bands_schema:
        lo = b.get("lower", float("-inf"))
        hi = b.get("upper", float("inf"))
        if lo <= pd_value < hi:
            return b["name"]
    return bands_schema[-1]["name"]


def _calibrate(calibrator: Any, pd_raw: np.ndarray) -> np.ndarray:
    """Aplica calibração isotônica; aceita transform() ou predict()."""
    if hasattr(calibrator, "transform"):
        return calibrator.transform(pd_raw)
    if hasattr(calibrator, "predict"):
        return calibrator.predict(pd_raw)
    raise RuntimeError("Calibrador sem métodos transform/predict.")


def _best_feature_names(art: Dict[str, Any], X_t, df: pd.DataFrame) -> List[str] | None:
    """
    Tenta nomes de features que combinem com X_t.shape[1].
    Ordem:
      1) art['feature_names'] se len == n_feats
      2) preprocessor.get_feature_names_out(df.columns) se existir e bater tamanho
      3) None (sem nomes; predição com matriz)
    """
    n_feats = X_t.shape[1]

    # 1) Lista carregada do artifacts (train_columns.csv)
    feat_names = art.get("feature_names")
    if isinstance(feat_names, list) and len(feat_names) == n_feats:
        return [str(c) for c in feat_names]

    # 2) Tentar do preprocessor
    prep = art.get("preprocessor")
    if hasattr(prep, "get_feature_names_out"):
        try:
            candidate = prep.get_feature_names_out(df.columns)
            if len(candidate) == n_feats:
                return [str(c) for c in candidate]
        except Exception:
            pass

    # 3) Sem nomes confiáveis
    return None


def score_df(df_raw: pd.DataFrame, art: Dict[str, Any]) -> pd.DataFrame:
    """
    Recebe DataFrame BRUTO e retorna:
      - pd_raw, pd_calibrated, band
      (Mantém SK_ID_CURR se vier no df_raw.)
    """
    # 1) Garante ordem/presença do schema bruto
    df = df_raw.copy()
    for c in art["raw_schema"]:
        if c not in df.columns:
            df[c] = np.nan
    df = df[art["raw_schema"]]

    # 2) Transforma
    X_t = art["preprocessor"].transform(df)

    # 3) Se possível, criar DataFrame com nomes; senão, usar matriz
    if hasattr(X_t, "toarray"):
        X_dense = X_t.toarray()
    else:
        X_dense = X_t

    names = _best_feature_names(art, X_t, df)
    if names is not None:
        X_infer = pd.DataFrame(X_dense, columns=names)
        pd_raw = art["model"].predict_proba(X_infer)[:, 1]
    else:
        pd_raw = art["model"].predict_proba(X_dense)[:, 1]

    # 4) Calibração
    pd_cal = _calibrate(art["calibrator"], pd_raw)

    # 5) Bandas
    bands_schema = art["bands_meta"]["bands"]["schema"]
    bands = [assign_band(v, bands_schema) for v in pd_cal]

    # 6) Saída (mantém SK_ID_CURR se existir)
    out = pd.DataFrame({
        "pd_raw": pd_raw,
        "pd_calibrated": pd_cal,
        "band": bands,
    })
    if "SK_ID_CURR" in df_raw.columns:
        out.insert(0, "SK_ID_CURR", df_raw["SK_ID_CURR"].values)

    return out


def score_records(records: List[Dict[str, Any]], art: Dict[str, Any]) -> pd.DataFrame:
    """Helper para receber lista de dicionários (JSON) e pontuar."""
    return score_df(pd.DataFrame.from_records(records), art)
