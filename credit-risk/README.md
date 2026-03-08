# Project Progress & Next Steps Checklist

A concise, professional overview of what we have accomplished so far and the planned activities ahead.

---

## ✔️ Completed

### 1. Environment & Setup

- [x] Created Conda environment **credit-risk** (Python 3.10).
- [x] Registered Jupyter kernel **Python (credit-risk)**.

### 2. Data Ingestion & Dictionary

- [x] Loaded `application_train.csv` and `application_test.csv`.
- [x] Loaded and standardized **HomeCredit_columns_description.csv**.

### 3. Data Understanding

- [x] Inspected shapes and previewed head of both train (48 744 × 121) and test (307 511 × 122) sets.
- [x] Computed data types and missing‐value rates for all features.
- [x] Identified top missing features in both datasets (real-estate attributes ~ 68–70% missing).

### 4. Data Dictionary Consolidation

- [x] Merged train summary with feature dictionary (219 documented variables).
- [x] Highlighted undocumented high-missing features.

### 5. Missing‐Value Handling

- [x] Created binary “\_missing_flag” indicators for high-missing numeric variables.
- [x] Imputed numeric features with medians; categorical features with “Unknown.”
- [x] Verified **zero** missing values remain.
- [x] Documented imputation approach and pandas warnings for future refactor.

---

## ⏳ To Do

### 6. Feature Scaling & Encoding

- [ ] Apply **RobustScaler** to all numeric features to mitigate outliers.
- [ ] One-hot encode low-cardinality categoricals.
- [ ] Target-encode high-cardinality categoricals.

### 7. Exploratory Data Analysis (EDA)

- [ ] Univariate distributions and summary statistics.
- [ ] Bivariate analysis vs. **TARGET** (boxplots, violin plots).
- [ ] Correlation heatmap and multicollinearity checks.
- [ ] Outlier detection and treatment plan.

### 8. Feature Engineering

- [ ] Aggregate auxiliary tables (`bureau.csv`, `previous_application.csv`, etc.) at client level.
- [ ] Create behavioral indicators (e.g. payment ratios, utilization rates).
- [ ] Temporal features (recency, application frequency).
- [ ] Interaction terms and polynomial features (if justified).

### 9. Model Development

- [ ] Split data into stratified K-folds (or time-based splits).
- [ ] Train baseline **Logistic Regression** with pipeline.
- [ ] Train tree-based models (Random Forest, XGBoost, LightGBM, CatBoost).
- [ ] Hyperparameter tuning with **Optuna**.
- [ ] Evaluate metrics: AUC-ROC, KS, Precision@k, Recall@k.

### 10. Model Explainability

- [ ] Generate **SHAP** global summary and dependence plots.
- [ ] Create local explanations (force plots) for sample clients.
- [ ] Document key drivers for stakeholder communication.

### 11. Validation & Monitoring

- [ ] Perform stability analysis (PSI, performance drift).
- [ ] Define retraining triggers and schedule.
- [ ] Implement model monitoring (Evidently.ai or WhyLabs).

### 12. Deployment & Reporting

- [ ] Build scoring **API** with FastAPI + Docker.
- [ ] Develop interactive **Streamlit** dashboard (or Power BI).
- [ ] Prepare final **README**, **presentation deck**, and **project narrative**.

---

> **Next immediate step:**  
> Proceed with Feature Scaling & Encoding (Section 6), then return here to tick off progress.

---

perfeito — aqui vai um “mapa do projeto” do JEITO MAIS CLARO possível: o que fizemos, por que fizemos, o que deu, e o que vamos atacar agora.

---

# 1) Problema de negócio (o que queremos)

Prever **probabilidade de inadimplência (TARGET=1)** para:

- **Tomada de decisão** (aprovação/limite/preço) e **priorização** (collections e monitoramento).
- Entrega **interpretável** (explicações SHAP, curva KS, thresholds) e **audível** (pipeline reprodutível).

KPIs principais:

- **AUC-ROC** e **KS** (discriminação global).
- **Precision\@20%** e **Recall\@20%** (qualidade do top-20% mais arriscado – útil para priorizar ações).
- **Brier score** e **reliability curve** (calibração das probabilidades como PD).

---

# 2) Dados & EDA (o que vimos)

- Bases principal: `application_train.csv` (com TARGET) e `application_test.csv`.
- **Missing alto** e concentrado nas variáveis “imobiliárias” (`COMMONAREA_*`, `LIVINGAPARTMENTS_*`, `YEARS_BUILD_*` etc., >60%).
- Dicionário `HomeCredit_columns_description.csv` usado para documentar variáveis.
- Checagem de tipos + padrão de ausências no train vs. test para evitar surpresas no scoring.

**Decisão:** tratar tudo **dentro da pipeline** (sem “pré-tratar” fora), para evitar vazamento e garantir reprodutibilidade.

---

# 3) Pré-processamento (profissional e sem vazamento)

Usamos um **`ColumnTransformer`** com três fluxos:

1. **Numéricas** → `SimpleImputer(median)` + `RobustScaler` (robusto a outliers).
2. **Categóricas de baixa cardinalidade** → `OneHotEncoder(handle_unknown='ignore')`.
3. **Categóricas de alta cardinalidade** → `TargetEncoder` (aprende médias condicionais **apenas no treino do fold**).

Tudo embrulhado em **`Pipeline`**, aplicado **dentro do CV**.
Isso garante que:

- encoders e imputers **não “olham” o validation fold**;
- scoring em produção replica exatamente o que aconteceu no treino.

Também **persistimos os nomes das features transformadas** (`artifacts/feature_names_logreg.csv`, 170 colunas) — útil para SHAP e auditoria.

---

# 4) Baseline — Logistic Regression (L2)

Rodamos 5-fold **estratificado** com `class_weight="balanced"`.

**OOF (fora-da-amostra):**

- **AUC ≈ 0.746**
- **KS ≈ 0.365**
- **P\@20 ≈ 0.200**
- **R\@20 ≈ 0.500**

Comentários:

- É um baseline **sólido** e interpretável.
- Apareceu `ConvergenceWarning` (lbfgs) → esperado em espaço de features alto (OHE + TE). Mitigamos com `max_iter↑` e opção de `solver="saga"` com L2.

---

# 5) Elastic Net (saga) — experimento e lição

Testamos **logística com elastic net** (`penalty='elasticnet'`, `solver='saga'`) para encolher coeficientes ruidosos.

**Resultado:**

- **Mais lenta** (horas por fold em alguns casos) e **métricas piores** (\~AUC 0.732, KS \~0.349, P\@20 \~0.194).
  **Decisão:** **descartar** para esse dataset (alto custo/baixo benefício). Mantemos **L2** como baseline linear.

---

# 6) Calibração de probabilidade (diagnóstico)

A **reliability curve** ficou **abaixo da diagonal** ⇒ o modelo **superestima PD** (over-prediction).
Isso é comum com `class_weight="balanced"` e dados desbalanceados.

**Plano:** **Isotonic Regression por fold** (fit→calibrate→validate sem vazamento) para melhorar **Brier** e alinhar PDs sem mexer muito na AUC/KS.

---

# 7) LightGBM + Optuna (o avanço)

Subimos para **árvores de gradiente** (LightGBM) com **busca de hiperparâmetros** via Optuna.

Pontos técnicos importantes que resolvemos:

- **LightGBM 4.x** não aceita `early_stopping_rounds` no `.fit()` estilo sklearn → usamos **callbacks** (`lgb.early_stopping`).
- `eval_set` dentro de `Pipeline` **não passa pelo preprocessor** → então **fitamos e transformamos** (`preprocessor.fit/transform`) **fora** e passamos **matriz numérica** para o LGBM (sem objetos).

**OOF por fold (seus números):**

- Fold1: AUC 0.757 | KS 0.385 | P\@20 0.211 | R\@20 0.523
- Fold2: AUC 0.768 | KS 0.396 | P\@20 0.217 | R\@20 0.538
- Fold3: AUC 0.759 | KS 0.388 | P\@20 0.211 | R\@20 0.523
- Fold4: AUC 0.766 | KS 0.404 | P\@20 0.214 | R\@20 0.530
- Fold5: AUC 0.757 | KS 0.383 | P\@20 0.209 | R\@20 0.518

**Média aproximada:**

- **AUC ≈ 0.761** (↑ vs 0.746 baseline)
- **KS ≈ 0.391** (↑ vs 0.365)
- **P\@20 ≈ 0.212** (↑ vs 0.200)
- **R\@20 ≈ 0.526** (↑ vs 0.500)

💡 **Leitura de negócio:** no **top-20%** mais arriscado, **capturamos mais inadimplentes e com maior precisão**, mantendo discriminação global melhor. Ou seja, **priorização melhor** com o mesmo esforço operacional.

---

# 8) Onde estamos agora & o que vamos resolver já

## O problema imediato

1. **Calibrar as probabilidades do LightGBM** (como PD confiável).

   - Objetivo: **reduzir Brier** e alinhar a **reliability curve** à diagonal.
   - Estratégia: **Isotonic** por fold (fit→calibrate→validate).

2. **Explicabilidade** com **SHAP**:

   - **Global** (quais variáveis puxam o risco para cima/baixo em média).
   - **Local** (explicar um cliente específico — ótimo para storytelling).

3. **Thresholding**:

   - Escolher um **cutoff** que maximize KS/Youden **ou** que maximize **lucro** dado custos de FP/FN (se você tiver esses custos).
   - Alternativa: **score bins** (Very High/High/Medium/Low) com cortes coerentes e estáveis.

## Por que isso?

- **AUC/KS** dizem “quão bem” separamos — **não** dizem se **probabilidade** está calibrada.
- Para **reportar PD** (e usar em pricing/limite), **calibração** é crítica.
- **SHAP** cria confiança: mostra **drivers de risco** de forma clara e defensável.
- **Threshold/binning** traduz o modelo para **decisão operacional** (quem aprovar, quem revisar, quem cobrar primeiro).

---

# 9) Próximos passos (em ordem)

**Agora**

1. Rodar **calibração isotônica** do LGBM (tenho a célula pronta — você já rodou a parte logística; a do LGBM está no meu último bloco).
2. Gerar **reliability curve (raw vs calibrated)** e **Brier** (antes/depois).
3. Gerar **SHAP** (summary bar + summary dot + exemplo local).

**Em seguida**
4\) **Escolher cutoff** (max KS / custo-benefício) e/ou **definir score bands**.
5\) **Persistir artefatos** (`preprocessor.joblib`, `lgbm_model.txt` ou `.joblib`, `isotonic.pkl`, `feature_names.csv`, `best_params.json`) + função `score(df)`.

**Depois (para fechar nível sênior)**
6\) **Feature engineering** com dados auxiliares (`bureau`, `previous_application`, `installments`, `credit_card_balance`, `POS_CASH`), agregações por cliente (p. ex.: atraso médio, max atraso, razão valor parcelado/limite, número de contratos ativos etc.).

- Isso costuma dar **salto grande de AUC/KS** nesse case.

7. **Drift & Monitoring** com Evidently (opcional) + **tracking** com MLflow (opcional).
8. **Entrega**: Streamlit simples (perfil de risco + SHAP local) e README 🔥.

---

# 10) Resumo em uma linha

**Já elevamos a performance saindo da regressão para LightGBM (↑AUC/KS/P\@20/R\@20); agora vamos calibrar as probabilidades para virar PD confiável e explicar o modelo com SHAP, fechando com thresholds/bins para operação.**

Se quiser, eu já **colo agora**:

- a **célula de calibração isotônica do LGBM** (pronta pra rodar),
- a **célula de SHAP** (global/local),
- e a **célula de persistência + função `score()`** para produção.
#   c r e d i t - r i s k  
 #   c r e d i t - r i s k  
 