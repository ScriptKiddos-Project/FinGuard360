# scripts/generate_raw_schema.py
from __future__ import annotations

from pathlib import Path
import json
import joblib

ART = Path(__file__).resolve().parents[1] / "artifacts"
TB_PATH = ART / "thresholds_bands.json"
PREP_PATH = ART / "preprocessor.joblib"

def _safe_float(x):
    # Garante valores serializáveis em JSON (sem Infinity)
    if x is None:
        return None
    try:
        f = float(x)
    except Exception:
        return x
    # Limita a um intervalo “grande”, mas finito
    if f == float("inf"):
        return 1e9
    if f == float("-inf"):
        return -1e9
    return f

def main():
    if not PREP_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {PREP_PATH}")
    if not TB_PATH.exists():
        raise FileNotFoundError(f"Não encontrei {TB_PATH}")

    # 1) extrai raw_schema do preprocessor
    preprocessor = joblib.load(PREP_PATH)
    if not hasattr(preprocessor, "feature_names_in_"):
        raise AttributeError(
            "O preprocessor não possui atributo feature_names_in_. "
            "Verifique a versão do scikit-learn ou considere extrair do dataset de treino."
        )
    raw_schema = [str(c) for c in preprocessor.feature_names_in_.tolist()]

    # 2) carrega thresholds_bands.json
    with open(TB_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 3) injeta o schema
    meta.setdefault("schema", {})
    meta["schema"]["raw_schema"] = raw_schema

    # 4) saneia bandas (lower/upper) para JSON puro (sem Infinity)
    bands = meta.get("bands", {})
    schema_bands = bands.get("schema", [])
    fixed_schema = []
    for band in schema_bands:
        fixed_schema.append({
            "name": band.get("name", "Unknown"),
            "lower": _safe_float(band.get("lower", -1e9)),
            "upper": _safe_float(band.get("upper", 1e9)),
        })
    if fixed_schema:
        meta["bands"]["schema"] = fixed_schema

    # 5) salva de volta
    with open(TB_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✔ Atualizado {TB_PATH} com schema.raw_schema ({len(raw_schema)} colunas) e bandas saneadas.")

if __name__ == "__main__":
    main()
