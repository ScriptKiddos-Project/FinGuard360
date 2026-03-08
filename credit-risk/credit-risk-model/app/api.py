# app/api.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Iterable
from pathlib import Path
import sys, io
import pandas as pd
import numpy as np

# ==========================================================
# Ajuste de PATHs para o Pylance e para execução da API
# - Rodar SEMPRE a partir da raiz do projeto:
#     python -m uvicorn app.api:app --reload --port 8000
# - Garanta que exista: scripts/__init__.py (vazio)
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]   # raiz do repo (contém app/ e scripts/)
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    sp = str(p)
    if sp not in sys.path:
        sys.path.append(sp)

# Agora podemos importar do pacote scripts
from scripts.preprocess import load_artifacts, assign_band  # sempre existe
try:
    from scripts.preprocess import score_with_decision      # pode não existir em versões antigas
except Exception:
    score_with_decision = None

# ----------------------------------------------------------
# Carrega artefatos (preprocessor, model, calibrator, schema)
# ----------------------------------------------------------
ART = Path(__file__).resolve().parents[1] / "artifacts"
# {'preprocessor','model','calibrator','feature_names','raw_schema','bands_meta'}
ART_DATA = load_artifacts(ART)
PREP = ART_DATA["preprocessor"]
MODEL = ART_DATA["model"]
CAL  = ART_DATA["calibrator"]
SCHEMA = ART_DATA["raw_schema"]                 # << usa schema BRUTO
FEATURE_NAMES = ART_DATA.get("feature_names")   # nomes transformados (pode não bater)
BANDS_META = ART_DATA["bands_meta"]
THRESH = float(BANDS_META["threshold"]["value"])
BANDS_SCHEMA = BANDS_META["bands"]["schema"]

def _calibrate(prob: np.ndarray) -> np.ndarray:
    """Usa o calibrador isotônico (transform/predict conforme objeto salvo)."""
    if hasattr(CAL, "transform"):
        return CAL.transform(prob)
    if hasattr(CAL, "predict"):
        return CAL.predict(prob)
    raise RuntimeError("Calibrador não possui métodos transform/predict.")

def _predict_proba_with_optional_names(Xt, df_cols: List[str]) -> np.ndarray:
    """
    Tenta prever usando DataFrame com nomes das features (se o tamanho bater).
    Se não bater, prevê direto com a matriz (evita erro de shape).
    """
    # Converte para denso se for esparso apenas quando formos criar DataFrame
    n_feats = Xt.shape[1]
    names = None

    # 1) tentar feature_names dos artifacts
    if isinstance(FEATURE_NAMES, list) and len(FEATURE_NAMES) == n_feats:
        names = [str(c) for c in FEATURE_NAMES]
    else:
        # 2) tentar do preprocessor
        if hasattr(PREP, "get_feature_names_out"):
            try:
                candidate = PREP.get_feature_names_out(df_cols)
                if len(candidate) == n_feats:
                    names = [str(c) for c in candidate]
            except Exception:
                names = None

    if names is not None:
        X_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        X_df = pd.DataFrame(X_dense, columns=names)
        return MODEL.predict_proba(X_df)[:, 1]

    # fallback: prever com a matriz diretamente (sem nomes)
    return MODEL.predict_proba(Xt)[:, 1]

def _score_core(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback caso 'score_with_decision' não exista no preprocess.
    Garante colunas, transforma, prediz, calibra, aplica bands e decisão.
    """
    df = df_raw.copy()
    for c in SCHEMA:
        if c not in df.columns:
            df[c] = np.nan
    df = df[SCHEMA]

    Xt = PREP.transform(df)
    pd_raw = _predict_proba_with_optional_names(Xt, df.columns.tolist())
    pd_cal = _calibrate(pd_raw)

    decision = np.where(pd_cal >= THRESH, "Decline/Review", "Approve")
    band = [assign_band(v, BANDS_SCHEMA) for v in pd_cal]

    out = pd.DataFrame(
        {
            "pd_raw": pd_raw,
            "pd_calibrated": pd_cal,
            "band": band,
            "decision": decision,
        }
    )
    if "SK_ID_CURR" in df_raw.columns:
        out.insert(0, "SK_ID_CURR", df_raw["SK_ID_CURR"].values)
    return out

def score_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Roteia para a função central do preprocess (se existir) ou usa o fallback local."""
    if callable(score_with_decision):
        return score_with_decision(df_raw, ART_DATA)
    return _score_core(df_raw)

# ------------------------------
# Sanitização de entradas
# ------------------------------
ALLOWED_KEYS = set(SCHEMA) | {"SK_ID_CURR"}  # SK_ID_CURR é opcional

def _sanitize_scalar(v: Any) -> Any:
    """Garante escalar simples; strings vazias viram NaN; listas/dicts são descartados."""
    if isinstance(v, (list, dict, tuple, set)):
        return np.nan
    if v == "" or v is None:
        return np.nan
    return v

def sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Mantém apenas colunas conhecidas, normaliza valores e retorna dict limpo."""
    clean = {}
    for k, v in row.items():
        if k in ALLOWED_KEYS:
            clean[k] = _sanitize_scalar(v)
    return clean

def ensure_any_feature(rows: Iterable[Dict[str, Any]]) -> None:
    """
    Valida que existe pelo menos UMA coluna de features conhecida (além de SK_ID_CURR).
    Evita requests vazias/irrelevantes.
    """
    for r in rows:
        keys = set(r.keys()) - {"SK_ID_CURR"}
        if len(keys & set(SCHEMA)) > 0:
            return
    raise HTTPException(status_code=422, detail="Nenhuma feature conhecida foi enviada.")

# ------------------------------
# Pydantic Models (com exemplos no Swagger)
# ------------------------------
class Record(BaseModel):
    data: Dict[str, Any] = Field(
        ...,
        description="Campos brutos da aplicação a serem pontuados. Colunas desconhecidas são ignoradas."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "data": {
                        "SK_ID_CURR": 100001,
                        "NAME_CONTRACT_TYPE": "Cash loans",
                        "CODE_GENDER": "M",
                        "FLAG_OWN_CAR": "Y",
                        "AMT_CREDIT": 450000,
                        "AMT_INCOME_TOTAL": 180000,
                        "DAYS_EMPLOYED": -1200,
                        "DAYS_BIRTH": -15000
                    }
                }
            ]
        }
    )

class Batch(BaseModel):
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de registros (cada item é um dicionário com as features)."
    )
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "rows": [
                        {
                            "SK_ID_CURR": 100001,
                            "NAME_CONTRACT_TYPE": "Cash loans",
                            "AMT_CREDIT": 450000
                        },
                        {
                            "SK_ID_CURR": 100002,
                            "NAME_CONTRACT_TYPE": "Revolving loans",
                            "AMT_CREDIT": 180000
                        }
                    ]
                }
            ]
        }
    )

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Serviço de pontuação de risco de crédito (Home Credit) com "
        "pré-processamento, modelo LGBM e calibração isotônica. "
        "Use /docs para testar."
    ),
    version="1.0.0"
)

# ------------------------------
# Rotas
# ------------------------------
@app.get("/health", tags=["Health"], summary="Status do serviço")
def health():
    """Retorna status e threshold atual usado na decisão."""
    return {"status": "ok", "threshold": THRESH}

@app.post("/score_one", tags=["Scoring"], summary="Pontuar um registro (JSON)")
def score_one(item: Record):
    """
    Recebe um dicionário com os campos do cliente, sanitiza, pontua e retorna:
    - `pd_raw`: probabilidade bruta do modelo
    - `pd_calibrated`: probabilidade calibrada (PD)
    - `band`: faixa de risco
    - `decision`: decisão baseada no threshold (Approve / Decline/Review)
    """
    row = sanitize_row(item.data)
    ensure_any_feature([row])
    df = pd.DataFrame([row])
    res = score_df(df)
    return res.to_dict(orient="records")[0]

@app.post("/score_batch", tags=["Scoring"], summary="Pontuar vários registros (JSON)")
def score_batch(batch: Batch):
    """
    Recebe uma lista de dicionários com campos dos clientes, sanitiza e pontua todos.
    Retorna um objeto com `n` e `rows`.
    """
    rows = [sanitize_row(r) for r in batch.rows]
    ensure_any_feature(rows)
    df = pd.DataFrame(rows)
    res = score_df(df)
    return {"n": len(res), "rows": res.to_dict(orient="records")}

@app.post(
    "/score-file",
    tags=["Scoring"],
    summary="Upload CSV e download de CSV pontuado",
    description=(
        "Recebe um arquivo CSV com colunas (parcialmente) compatíveis com o schema de treino "
        "e devolve um CSV com colunas `pd_raw`, `pd_calibrated`, `band`, `decision`."
    )
)
async def score_file(file: UploadFile = File(..., description="CSV de aplicações para pontuar")):
    # Lê o CSV para DataFrame (tenta detectar separador automaticamente)
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Arquivo vazio.")
        # Tentativa 1: detecção automática de separador (engine='python', sep=None)
        try:
            df = pd.read_csv(io.BytesIO(content), sep=None, engine="python", low_memory=False)
        except Exception:
            # Tentativa 2: separador padrão vírgula
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao ler CSV: {e}")

    # Sanitiza colunas desconhecidas/valores compostos
    keep_cols = list((set(df.columns) & set(SCHEMA)) | {"SK_ID_CURR"})
    if not keep_cols:
        raise HTTPException(status_code=422, detail="CSV não contém nenhuma coluna conhecida.")
    df = df[keep_cols].copy()
    for c in df.columns:
        df[c] = df[c].map(_sanitize_scalar)

    # Garante que exista pelo menos uma feature conhecida em alguma linha
    ensure_any_feature(df.to_dict(orient="records"))

    # Scoring
    scored = score_df(df)

    # Retorna como CSV por streaming
    buf = io.StringIO()
    scored.to_csv(buf, index=False)
    buf.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="scored_applications.csv"'}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Credit Risk Scoring API. Acesse /docs para o Swagger UI.",
        "docs": "/docs",
        "health": "/health"
    }
