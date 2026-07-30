"""
train.py

Treinamento e comparacao dos modelos de previsao de inadimplencia:
    - Baseline: Regressao Logistica
    - Random Forest
    - LightGBM

Cada modelo e encapsulado em um Pipeline do imbalanced-learn que aplica, na
ordem: pre-processamento (imputacao/winsorizacao, ajustado somente em
X_train) -> SMOTE (balanceamento de classes, aplicado somente em X_train,
nunca no teste) -> classificador. Isso garante que nenhuma informacao do
conjunto de teste vaza para o treino.
"""

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features import build_preprocessing_pipeline

TARGET_COL = "SeriousDlqin2yrs"
RANDOM_STATE = 42


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Split estratificado treino/teste (estratificado pela variavel alvo,
    para preservar a proporcao de inadimplentes em ambos os conjuntos)."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def get_candidate_models() -> dict:
    """Modelos candidatos a comparar. LightGBM entra como representante de
    gradient boosting (requirements.txt tambem inclui xgboost, caso queira
    adicionar como uma quarta comparacao)."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
        ),
    }


def build_model_pipeline(classifier) -> ImbPipeline:
    """Monta o pipeline completo: pre-processamento -> SMOTE -> classificador.

    Usa o Pipeline do imbalanced-learn (nao o Pipeline padrao do
    scikit-learn) porque o SMOTE precisa participar do fit_resample durante
    o .fit(), etapa que o Pipeline padrao nao suporta (ele so encadeia
    transform/predict, nao resample).
    """
    return ImbPipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )


def train_and_compare(X_train, y_train, X_test, y_test) -> dict:
    """Treina todos os modelos candidatos e retorna, para cada um, o
    pipeline ajustado, as probabilidades previstas no teste e as metricas.
    """
    from src.evaluate import calculate_metrics  # import local evita ciclo

    results = {}
    for name, classifier in get_candidate_models().items():
        pipeline = build_model_pipeline(classifier)
        pipeline.fit(X_train, y_train)

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = calculate_metrics(y_test, y_pred_proba)

        results[name] = {
            "pipeline": pipeline,
            "y_pred_proba": y_pred_proba,
            "metrics": metrics,
        }
        print(f"{name}: {metrics}")

    return results


def save_model(pipeline, name: str, output_dir: str = "models") -> Path:
    """Serializa o pipeline treinado (pre-processamento + SMOTE + modelo)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{name}.joblib"
    joblib.dump(pipeline, file_path)
    return file_path


if __name__ == "__main__":
    df = pd.read_csv("data/processed/credit_features.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    results = train_and_compare(X_train, y_train, X_test, y_test)

    best_name = max(results, key=lambda k: results[k]["metrics"]["auc_roc"])
    print(f"\nMelhor modelo por AUC-ROC: {best_name}")
    save_model(results[best_name]["pipeline"], best_name)
