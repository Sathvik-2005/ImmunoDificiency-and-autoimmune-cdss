"""
Immune Disease Risk Detection - ML Model
=========================================
• 70 / 30 train-test split (stratified)
• 3 models: Risk Detection, Disease Category, Risk Stratification
• Risk levels: No Risk | Low Risk | Moderate Risk | Severe Risk
• All test metrics saved to outputs/test_results.json for the GUI dashboard
"""

import pandas as pd
import numpy as np
import joblib, json, logging
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR    = BASE_DIR / "logs"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Feature columns ───────────────────────────────────────────
LAB_FEATURES = [
    'RBC_Count','Hemoglobin','Hematocrit','MCV','MCH','MCHC','RDW',
    'Reticulocyte_Count','WBC_Count','Neutrophils','Lymphocytes',
    'Monocytes','Eosinophils','Basophils','PLT_Count','MPV',
    'ANA','ESR','C3','C4','CRP','MBL_Level','Esbach'
]
SYMPTOM_FEATURES = [
    'Low-grade fever','Fatigue or chronic tiredness','Dizziness',
    'Weight loss','Rashes and skin lesions','Stiffness in the joints',
    'Brittle hair or hair loss','Dry eyes and/or mouth',
    "General 'unwell' feeling",'Joint pain'
]
DEMOGRAPHIC_FEATURES = ['Age','Gender_numeric','Sickness_Duration_Months']
ANTIBODY_FEATURES = [
    'Anti-dsDNA','Anti-Sm','Rheumatoid factor','ACPA','Anti-TPO',
    'Anti-Tg','Anti-SMA','Anti_dsDNA','Anti_enterocyte_antibodies',
    'anti_LKM1','Anti_RNP','ASCA','Anti_Ro_SSA','Anti_CBir1',
    'Anti_BP230','Anti_tTG','DGP','Anti_BP180','ASMA','Anti_IF',
    'IgG_IgE_receptor','Anti_SRP','Anti_desmoglein_3','Anti_La_SSB',
    'Anti_Jo1','ANCA','anti_centromere','Anti_desmoglein_1','EMA',
    'Anti_type_VII_collagen','C1_inhibitor','Anti_TIF1',
    'Anti_epidermal_basement_membrane_IgA','Anti_OmpC','pANCA',
    'Anti_tissue_transglutaminase','anti_Scl_70','Anti_Mi2',
    'Anti_parietal_cell','Progesterone_antibodies','Anti_Sm'
]
# Immunodeficiency-specific clinical flags (from registry dataset)
IMMUNODEF_FEATURES = ['Had_Transplant','Ig_Therapy_Ever','Family_HX_PI']

ALL_FEATURES = DEMOGRAPHIC_FEATURES + LAB_FEATURES + SYMPTOM_FEATURES + IMMUNODEF_FEATURES

# ── 4-Level Risk Labels ───────────────────────────────────────
RISK_LEVELS = ['No Risk', 'Low Risk', 'Moderate Risk', 'Severe Risk']

# ══════════════════════════════════════════════════════════════
# SEVERITY SCORING ENGINE
# ══════════════════════════════════════════════════════════════
def compute_severity_score(row: pd.Series) -> float:
    """
    Compute a continuous severity score (0.0 – 1.0) per patient
    based on real data distributions observed in the dataset:

      Antibody_Score    : diseased mean=3.9, normal mean=1.9
      Symptom_Burden    : diseased mean=4.9, normal mean=0.5
      ESR               : diseased mean=29,  both groups similar
      CRP               : diseased mean=5.4, both groups similar
      Sickness_Duration : 1–119 months, longer = more chronic

    Weights chosen to reflect clinical importance:
      Antibody positivity (40%) > Symptom load (35%) >
      Inflammation markers (15%) > Duration (10%)
    """
    # ── Antibody score: normalise to 0–1 (max observed = 11)
    ab = float(row.get('Antibody_Score', 0) or 0)
    ab_norm = min(ab / 11.0, 1.0)

    # ── Symptom burden: normalise to 0–1 (max = 10 symptoms)
    sym = float(row.get('Symptom_Burden', 0) or 0)
    sym_norm = min(sym / 10.0, 1.0)

    # ── Inflammation index: ESR 0-49, CRP 0-10
    esr = float(row.get('ESR', 0) or 0)
    crp = float(row.get('CRP', 0) or 0)
    inflam_norm = min((esr / 49.0) * 0.5 + (crp / 10.0) * 0.5, 1.0)

    # ── Sickness duration: 0–119 months
    dur = float(row.get('Sickness_Duration_Months', 0) or 0)
    dur_norm = min(dur / 119.0, 1.0)

    # ── ANA positive adds a baseline bump
    ana_bump = 0.05 if float(row.get('ANA', 0) or 0) > 0 else 0.0

    score = (
        ab_norm  * 0.40 +
        sym_norm * 0.35 +
        inflam_norm * 0.15 +
        dur_norm * 0.10 +
        ana_bump
    )
    return round(min(score, 1.0), 4)


def assign_risk_level(row: pd.Series) -> str:
    """
    Map each patient to one of four risk levels.

    Thresholds derived from the data:
      - Normal patients score ~0.05–0.25  → No Risk
      - Mild diseased (low antibodies, few symptoms) → Low Risk
      - Moderate diseased (several antibodies, several symptoms) → Moderate Risk
      - Severe (high antibody load + high symptom burden + chronic) → Severe Risk

    Normal patients always get 'No Risk' regardless of score,
    because their diagnosis is confirmed.
    """
    if row.get('Diagnosis', '') == 'Normal':
        return 'No Risk'

    score = row.get('severity_score', compute_severity_score(row))

    if score < 0.25:
        return 'Low Risk'
    elif score < 0.50:
        return 'Moderate Risk'
    else:
        return 'Severe Risk'


# ══════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
def load_and_preprocess(filepath: str) -> pd.DataFrame:
    log.info(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    log.info(f"  {len(df)} rows, {len(df.columns)} cols")

    # Gender → numeric
    if 'Gender' in df.columns:
        df['Gender_numeric'] = (
            df['Gender'].map({'Male':1,'Female':0,'male':1,'female':0}).fillna(0)
        )

    # Disease category
    df['Disease_Category'] = df['Diagnosis'].apply(_classify_disease)

    # Binary risk label (used by model 1)
    df['Risk_Label'] = (df['Diagnosis'] != 'Normal').astype(int)

    # Engineered features
    sym_cols = [c for c in SYMPTOM_FEATURES if c in df.columns]
    df['Symptom_Burden'] = df[sym_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    ab_cols = [c for c in ANTIBODY_FEATURES if c in df.columns]
    df['Antibody_Score'] = df[ab_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    if 'ESR' in df.columns and 'CRP' in df.columns:
        df['Inflammation_Index'] = df['ESR'].fillna(0)*0.5 + df['CRP'].fillna(0)*0.5

    if 'C3' in df.columns and 'C4' in df.columns:
        df['Complement_Activity'] = df['C3'].fillna(0) + df['C4'].fillna(0)

    # ── Severity score & 4-level risk ────────────────────────
    df['severity_score'] = df.apply(compute_severity_score, axis=1)
    df['Risk_Level']     = df.apply(assign_risk_level, axis=1)

    dist = df['Risk_Level'].value_counts().to_dict()
    log.info(f"  Risk level distribution: {dist}")
    return df


def _classify_disease(diagnosis: str) -> str:
    if pd.isna(diagnosis) or str(diagnosis).strip() == 'Normal':
        return 'Normal'
    d = str(diagnosis).lower()
    immuno_kw = [
        # Registry-derived exact keywords
        'agammaglobulinemia','immunodeficiency','scid','digeorge',
        'wiskott','ataxia telangiectasia','ataxia-telangiectasia',
        'granulomatous','complement deficien','mbl deficiency',
        'natural killer','leukocyte adhesion','chediak',
        # New dataset terms
        'cvid','common variable immune','hypogammaglobulinemia',
        'hyper igm','hyper ige','iga deficiency','igg subclass',
        'combined immune deficiency','immune dysregulation',
        'transient hypogammaglobulinemia','mucocutaneous candidiasis',
        'hlh','xlp','pigmentary disorder','myelodysplasia','gata2',
        'ectodermal dysplasia','nemo','viral infection','interferonopathy',
        'aicardi','mycobacteria','msmd','omenn','nk cell',
        'charge syndrome','dyskeratosis','tlr pathway',
        'neutropenia','neutrophil','phagocyte','pi connect',
        'autoimmune lymphoproliferative','alps','autoinflammatory',
        'predominantly antibody','cellular and humoral',
        'intrinsic and innate','syndromic features',
        'specific antibody deficiency'
    ]
    return 'Immunodeficiency' if any(k in d for k in immuno_kw) else 'Autoimmune'


def get_feature_cols(df: pd.DataFrame) -> list:
    extended = ALL_FEATURES + ['Symptom_Burden','Antibody_Score',
                                'Inflammation_Index','Complement_Activity',
                                'severity_score']
    return [c for c in extended if c in df.columns]


def _json_safe(obj):
    if isinstance(obj, dict):        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


# ══════════════════════════════════════════════════════════════
# MODEL 1 — BINARY RISK DETECTION  (70 / 30)
# ══════════════════════════════════════════════════════════════
def train_risk_model(df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("MODEL 1: BINARY RISK DETECTION  (70 / 30 split)")

    feat = get_feature_cols(df)
    X = df[feat].apply(pd.to_numeric, errors='coerce')
    y = df['Risk_Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    log.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, class_weight='balanced', n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred, target_names=['Normal','At Risk'], output_dict=True
    )
    cv = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

    rf = pipeline.named_steps['model']
    fi = pd.DataFrame({'feature': feat, 'importance': rf.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    fi.to_csv(OUTPUT_DIR / 'risk_model_feature_importance.csv', index=False)

    metrics = {
        'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
        'split': '70/30',
        'accuracy':    round(accuracy_score(y_test, y_pred), 4),
        'precision':   round(precision_score(y_test, y_pred), 4),
        'recall':      round(recall_score(y_test, y_pred), 4),
        'f1':          round(f1_score(y_test, y_pred), 4),
        'roc_auc':     round(roc_auc_score(y_test, y_proba), 4),
        'cv_auc_mean': round(float(cv.mean()), 4),
        'cv_auc_std':  round(float(cv.std()), 4),
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_curve': {'fpr': fpr[::5].tolist(), 'tpr': tpr[::5].tolist()},
        'feature_importance': fi.head(15).to_dict('records'),
        'label_names': ['Normal','At Risk'],
    }

    # ── Save misclassified test rows ──────────────────────────
    test_df = df.iloc[X_test.index].copy()
    test_df['predicted_label'] = y_pred
    test_df['predicted_proba'] = y_proba
    test_df['actual_label']    = y_test.values
    test_df['correct']         = (y_pred == y_test.values)

    wrong = test_df[~test_df['correct']].copy()
    wrong['model']             = 'Risk Detection'
    wrong['actual_text']       = wrong['actual_label'].map({0:'Normal', 1:'At Risk'})
    wrong['predicted_text']    = wrong['predicted_label'].map({0:'Normal', 1:'At Risk'})
    wrong['confidence']        = wrong['predicted_proba'].apply(
        lambda p: round(max(p, 1-p), 4))
    wrong['error_type']        = wrong.apply(lambda r:
        'False Negative (missed disease)' if r['actual_label']==1
        else 'False Positive (healthy flagged)', axis=1)

    out_cols = ['Patient_ID','Age','Gender','Diagnosis','Disease_Category',
                'actual_text','predicted_text','confidence','error_type',
                'Symptom_Burden','Antibody_Score','ESR','CRP','ANA',
                'WBC_Count','Lymphocytes','severity_score']
    out_cols = [c for c in out_cols if c in wrong.columns]
    wrong[out_cols].to_csv(OUTPUT_DIR / 'misclassified_risk.csv', index=False)

    fn = int(((y_test==1) & (y_pred==0)).sum())   # missed diseases
    fp = int(((y_test==0) & (y_pred==1)).sum())   # healthy flagged
    log.info(f"  Misclassified: {len(wrong)} / {len(X_test)} "
             f"(FN={fn} missed diseases, FP={fp} false alarms)")
    metrics['misclassified_count'] = len(wrong)
    metrics['false_negatives']     = fn
    metrics['false_positives']     = fp

    joblib.dump(pipeline, MODEL_DIR / 'risk_model.pkl')
    joblib.dump(feat,     MODEL_DIR / 'risk_model_features.pkl')
    log.info(f"  AUC={metrics['roc_auc']}  Acc={metrics['accuracy']}")
    return metrics


# ══════════════════════════════════════════════════════════════
# MODEL 2 — DISEASE CATEGORY  (70 / 30)
# ══════════════════════════════════════════════════════════════
def train_category_model(df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("MODEL 2: DISEASE CATEGORY CLASSIFICATION  (70 / 30 split)")

    feat = get_feature_cols(df)
    X    = df[feat].apply(pd.to_numeric, errors='coerce')
    le   = LabelEncoder()
    y    = le.fit_transform(df['Disease_Category'])
    names= le.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    log.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=5, random_state=42, subsample=0.8
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=names, output_dict=True)
    test_dist = {names[i]: int((y_test == i).sum()) for i in range(len(names))}

    metrics = {
        'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
        'split': '70/30',
        'accuracy':    round(accuracy_score(y_test, y_pred), 4),
        'f1_weighted': round(f1_score(y_test, y_pred, average='weighted'), 4),
        'confusion_matrix': cm,
        'classification_report': report,
        'label_names': names,
        'test_distribution': test_dist,
    }

    # ── Save misclassified test rows ──────────────────────────
    test_df = df.iloc[X_test.index].copy()
    test_df['predicted_idx']   = y_pred
    test_df['actual_idx']      = y_test
    test_df['predicted_cat']   = le.inverse_transform(y_pred)
    test_df['actual_cat']      = le.inverse_transform(y_test)
    test_df['correct']         = (y_pred == y_test)

    wrong = test_df[~test_df['correct']].copy()
    wrong['model']             = 'Disease Category'
    wrong['error_type']        = wrong.apply(
        lambda r: f"Predicted '{r['predicted_cat']}' but was '{r['actual_cat']}'",
        axis=1)
    probas = pipeline.predict_proba(X.iloc[X_test.index])
    wrong['confidence']        = [
        round(float(probas[i, y_pred[i]]), 4)
        for i in range(len(y_pred))
        if not (y_pred == y_test)[i]
    ][:len(wrong)]

    out_cols = ['Patient_ID','Age','Gender','Diagnosis',
                'actual_cat','predicted_cat','confidence','error_type',
                'Symptom_Burden','Antibody_Score','ESR','CRP','ANA',
                'WBC_Count','Lymphocytes','severity_score']
    out_cols = [c for c in out_cols if c in wrong.columns]
    wrong[out_cols].to_csv(OUTPUT_DIR / 'misclassified_category.csv', index=False)
    log.info(f"  Misclassified: {len(wrong)} / {len(X_test)}")
    metrics['misclassified_count'] = len(wrong)

    joblib.dump(pipeline, MODEL_DIR / 'category_model.pkl')
    joblib.dump(le,       MODEL_DIR / 'category_label_encoder.pkl')
    joblib.dump(feat,     MODEL_DIR / 'category_model_features.pkl')
    log.info(f"  Acc={metrics['accuracy']}  F1={metrics['f1_weighted']}")
    return metrics


# ══════════════════════════════════════════════════════════════
# MODEL 3 — 4-LEVEL RISK STRATIFICATION  (70 / 30)
# Levels: No Risk | Low Risk | Moderate Risk | Severe Risk
# ══════════════════════════════════════════════════════════════
def train_stratification_model(df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("MODEL 3: 4-LEVEL RISK STRATIFICATION  (70 / 30 split)")
    log.info("  Levels: No Risk | Low Risk | Moderate Risk | Severe Risk")

    # Features most relevant to severity
    strat_cols = [
        'severity_score',       # master score
        'Antibody_Score',       # antibody load
        'Symptom_Burden',       # symptom count
        'Inflammation_Index',   # ESR + CRP composite
        'Age',
        'WBC_Count',
        'Lymphocytes',
        'Neutrophils',
        'ESR', 'CRP', 'ANA',
        'Sickness_Duration_Months'
    ]
    strat_cols = [c for c in strat_cols if c in df.columns]

    X = df[strat_cols].apply(pd.to_numeric, errors='coerce')
    y_raw = df['Risk_Level']

    # Encode the 4 levels with a fixed order
    le_strat = LabelEncoder()
    le_strat.classes_ = np.array(RISK_LEVELS)  # fix order: No/Low/Moderate/Severe
    y = y_raw.map({lvl: i for i, lvl in enumerate(RISK_LEVELS)})

    log.info(f"  Distribution:\n{y_raw.value_counts().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    log.info(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')
    cm     = confusion_matrix(y_test, y_pred, labels=[0,1,2,3]).tolist()
    report = classification_report(
        y_test, y_pred, labels=[0,1,2,3],
        target_names=RISK_LEVELS, output_dict=True
    )

    # Risk distribution in full dataset
    risk_dist = y_raw.value_counts().to_dict()
    # Distribution on test set
    test_risk_dist = {
        RISK_LEVELS[i]: int((y_test == i).sum())
        for i in range(4)
        if (y_test == i).sum() > 0
    }

    metrics = {
        'train_size': int(len(X_train)), 'test_size': int(len(X_test)),
        'split': '70/30',
        'accuracy':    round(float(acc), 4),
        'f1_weighted': round(float(f1), 4),
        'confusion_matrix': cm,
        'classification_report': report,
        'label_names': RISK_LEVELS,
        'risk_distribution': {k: int(v) for k, v in risk_dist.items()},
        'test_risk_distribution': test_risk_dist,
    }

    log.info(f"  Acc={acc:.3f}  F1={f1:.3f}")

    # ── Save misclassified test rows ──────────────────────────
    test_df = df.iloc[X_test.index].copy()
    test_df['predicted_idx']    = y_pred
    test_df['actual_idx']       = y_test.values
    test_df['predicted_level']  = [RISK_LEVELS[i] if i < len(RISK_LEVELS) else 'Unknown'
                                    for i in y_pred]
    test_df['actual_level']     = [RISK_LEVELS[i] if i < len(RISK_LEVELS) else 'Unknown'
                                    for i in y_test.values]
    test_df['correct']          = (y_pred == y_test.values)

    wrong = test_df[~test_df['correct']].copy()
    wrong['model']              = 'Risk Stratification'
    wrong['error_direction']    = wrong.apply(
        lambda r: 'Under-estimated risk' if r['predicted_idx'] < r['actual_idx']
                  else 'Over-estimated risk', axis=1)
    wrong['error_type']         = wrong.apply(
        lambda r: f"Predicted '{r['predicted_level']}' but was '{r['actual_level']}'  →  {r['error_direction']}",
        axis=1)
    wrong['levels_off']         = abs(wrong['predicted_idx'] - wrong['actual_idx'])

    out_cols = ['Patient_ID','Age','Gender','Diagnosis','Disease_Category',
                'actual_level','predicted_level','error_type','levels_off','error_direction',
                'Symptom_Burden','Antibody_Score','severity_score',
                'ESR','CRP','ANA','WBC_Count','Lymphocytes']
    out_cols = [c for c in out_cols if c in wrong.columns]
    wrong[out_cols].sort_values('levels_off', ascending=False).to_csv(
        OUTPUT_DIR / 'misclassified_risk_level.csv', index=False)

    under = int((wrong['error_direction'] == 'Under-estimated risk').sum())
    over  = int((wrong['error_direction'] == 'Over-estimated risk').sum())
    log.info(f"  Misclassified: {len(wrong)} / {len(X_test)} "
             f"(Under={under}, Over={over})")
    metrics['misclassified_count']  = len(wrong)
    metrics['under_estimated']      = under
    metrics['over_estimated']       = over

    joblib.dump(pipeline,   MODEL_DIR / 'strat_model.pkl')
    joblib.dump(strat_cols, MODEL_DIR / 'strat_features.pkl')
    return metrics


# ══════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════
def train_all(data_path: str = None, progress_callback=None) -> dict:
    """
    Train all 3 models on 70/30 split.
    Saves full metrics → outputs/test_results.json
    """
    # ── Discover and load datasets ────────────────────────────
    if data_path and Path(data_path).exists():
        df_raw = pd.read_csv(data_path)
        log.info(f"Using custom dataset: {data_path}  ({len(df_raw)} rows)")
    else:
        # Auto-discover: prefer the combined dataset, otherwise merge individual ones
        # Priority: full combined → combined_with_normal → individual files
        combined_all = DATA_DIR / 'combined_all.csv'
        combined_old = DATA_DIR / 'combined_with_normal_2000.csv'
        ai            = DATA_DIR / 'Comprehensive_Autoimmune_Disorder_Dataset.csv'
        im            = DATA_DIR / 'immunodeficiency_dataset.csv'

        if combined_all.exists():
            df_raw = pd.read_csv(combined_all)
            log.info(f"Using combined_all dataset: {len(df_raw)} rows")
        elif combined_old.exists():
            dfs = [pd.read_csv(combined_old)]
            if im.exists(): dfs.append(pd.read_csv(im))
            df_raw = pd.concat(dfs, ignore_index=True)
            log.info(f"Merged datasets: {len(df_raw)} rows")
        else:
            dfs = []
            if ai.exists(): dfs.append(pd.read_csv(ai))
            if im.exists(): dfs.append(pd.read_csv(im))
            if not dfs:
                raise FileNotFoundError("No dataset found in data/ directory.")
            df_raw = pd.concat(dfs, ignore_index=True)
            log.info(f"Merged {len(dfs)} datasets: {len(df_raw)} rows")

    tmp = DATA_DIR / '_merged_temp.csv'
    df_raw.to_csv(tmp, index=False)

    total = 4
    if progress_callback: progress_callback(1, total, "Preprocessing & computing severity scores…")
    df = load_and_preprocess(str(tmp))

    if progress_callback: progress_callback(2, total, "Training binary risk model (70/30)…")
    risk_m = train_risk_model(df)

    if progress_callback: progress_callback(3, total, "Training disease category model (70/30)…")
    cat_m = train_category_model(df)

    if progress_callback: progress_callback(4, total, "Training 4-level stratification model (70/30)…")
    strat_m = train_stratification_model(df)

    # ── Dataset statistics for dashboard ─────────────────────
    age_hist, age_bins = np.histogram(df['Age'].dropna(), bins=10)
    gender_dist = df['Gender'].value_counts().to_dict() if 'Gender' in df.columns else {}

    results = {
        'trained_at':       datetime.now().isoformat(),
        'total_samples':    int(len(df)),
        'train_test_split': '70 / 30',
        'dataset': {
            'total_records':    int(len(df)),
            'diagnosis_count':  int(df['Diagnosis'].nunique()),
            'category_dist':    {k: int(v) for k, v in
                                  df['Disease_Category'].value_counts().items()},
            'risk_level_dist':  {k: int(v) for k, v in
                                  df['Risk_Level'].value_counts().items()},
            'diagnosis_top30':  {k: int(v) for k, v in
                                  df['Diagnosis'].value_counts().head(30).items()},
            'gender_dist':      {k: int(v) for k, v in gender_dist.items()},
            'age_histogram': {
                'counts': age_hist.tolist(),
                'bins':   [round(float(b),1) for b in age_bins.tolist()],
            },
            'avg_age':            round(float(df['Age'].mean()), 1),
            'avg_symptom_burden': round(float(df['Symptom_Burden'].mean()), 2),
            'avg_severity_score': round(float(df['severity_score'].mean()), 4),
        },
        'risk_model':     risk_m,
        'category_model': cat_m,
        'strat_model':    strat_m,
    }

    out = OUTPUT_DIR / 'test_results.json'
    with open(out, 'w') as f:
        json.dump(_json_safe(results), f, indent=2)
    log.info(f"Saved test results → {out}")

    meta = {
        'trained_at': results['trained_at'],
        'total_samples': results['total_samples'],
        'train_test_split': '70/30',
        'risk_model':    {k: risk_m[k] for k in
                          ('accuracy','precision','recall','f1','roc_auc','cv_auc_mean')},
        'category_model':{k: cat_m[k] for k in ('accuracy','f1_weighted')},
        'strat_model':   {k: strat_m[k] for k in ('accuracy','f1_weighted','risk_distribution')},
        'feature_columns': get_feature_cols(df),
    }
    with open(MODEL_DIR / 'model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════
# SINGLE-PATIENT PREDICTION
# ══════════════════════════════════════════════════════════════
def predict_patient(patient_data: dict) -> dict:
    """
    Run all 3 models on a single patient.
    Returns risk score, disease category, AND 4-level risk label.
    """
    try:
        risk_model  = joblib.load(MODEL_DIR / 'risk_model.pkl')
        risk_feat   = joblib.load(MODEL_DIR / 'risk_model_features.pkl')
        cat_model   = joblib.load(MODEL_DIR / 'category_model.pkl')
        le          = joblib.load(MODEL_DIR / 'category_label_encoder.pkl')
        cat_feat    = joblib.load(MODEL_DIR / 'category_model_features.pkl')
        strat_model = joblib.load(MODEL_DIR / 'strat_model.pkl')
        strat_feat  = joblib.load(MODEL_DIR / 'strat_features.pkl')
    except FileNotFoundError:
        return {'error': 'Models not trained yet. Please run training first.'}

    def row(features):
        return pd.DataFrame([{f: patient_data.get(f, np.nan) for f in features}])

    # ── Compute engineered features first ────────────────────
    ab_cols  = [c for c in ANTIBODY_FEATURES if c in patient_data]
    sym_cols = [c for c in SYMPTOM_FEATURES  if c in patient_data]

    patient_data['Antibody_Score']    = sum(float(patient_data.get(c, 0) or 0) for c in ab_cols)
    patient_data['Symptom_Burden']    = sum(float(patient_data.get(c, 0) or 0) for c in sym_cols)
    patient_data['Inflammation_Index'] = (
        float(patient_data.get('ESR', 0) or 0) * 0.5 +
        float(patient_data.get('CRP', 0) or 0) * 0.5
    )
    patient_data['Complement_Activity'] = (
        float(patient_data.get('C3', 0) or 0) +
        float(patient_data.get('C4', 0) or 0)
    )
    patient_data['severity_score'] = compute_severity_score(patient_data)

    # ── Model 1: Binary risk ─────────────────────────────────
    risk_proba = risk_model.predict_proba(row(risk_feat))[0][1]

    # ── Model 2: Disease category ────────────────────────────
    cat_idx    = cat_model.predict(row(cat_feat))[0]
    cat_name   = le.inverse_transform([cat_idx])[0]
    cat_probas = cat_model.predict_proba(row(cat_feat))[0]

    # ── Model 3: 4-level risk stratification ─────────────────
    X_strat    = row(strat_feat).apply(pd.to_numeric, errors='coerce').fillna(0)
    level_idx  = strat_model.predict(X_strat)[0]
    risk_level = RISK_LEVELS[int(level_idx)] if int(level_idx) < len(RISK_LEVELS) else 'Unknown'
    level_probas = strat_model.predict_proba(X_strat)[0]

    return {
        'risk_score':    round(float(risk_proba), 4),
        'risk_label':    'At Risk' if risk_proba >= 0.5 else 'Normal',
        'disease_category': cat_name,
        'category_probabilities': {
            le.classes_[i]: round(float(p), 4) for i, p in enumerate(cat_probas)
        },
        'risk_level':    risk_level,                          # ← 4-level output
        'severity_score': round(patient_data['severity_score'], 4),
        'level_probabilities': {
            RISK_LEVELS[i]: round(float(p), 4)
            for i, p in enumerate(level_probas)
            if i < len(RISK_LEVELS)
        },
    }


if __name__ == '__main__':
    import sys
    train_all(sys.argv[1] if len(sys.argv) > 1 else None)
