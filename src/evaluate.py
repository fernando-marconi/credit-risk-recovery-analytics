"""
evaluate.py

Avaliacao do modelo treinado e explicabilidade com SHAP.

Metricas de avaliacao (padrao de mercado para modelos de credito):
    - AUC-ROC: capacidade geral do modelo de ordenar bons e maus pagadores.
    - KS (Kolmogorov-Smirnov): maior distancia entre as distribuicoes
      cumulativas de score dos bons e dos maus pagadores. E a metrica mais
      usada no mercado de credito/cobranca para avaliar poder discriminatorio
      (mais falada em comites de credito do que o proprio AUC-ROC).
    - Recall / Precision na classe positiva (inadimplentes), no threshold
      escolhido.
"""

import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_ks(y_true, y_pred_proba) -> float:
    """Estatistica KS: maior distancia entre as distribuicoes de score dos
    bons (y_true == 0) e maus pagadores (y_true == 1)."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    score_bons = y_pred_proba[y_true == 0]
    score_maus = y_pred_proba[y_true == 1]
    return float(ks_2samp(score_bons, score_maus).statistic)


def calculate_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """Calcula as principais metricas de avaliacao do modelo de credito."""
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)

    return {
        "auc_roc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "ks": round(calculate_ks(y_true, y_pred_proba), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
    }


def get_confusion_matrix(y_true, y_pred_proba, threshold: float = 0.5) -> np.ndarray:
    """Matriz de confusao no threshold escolhido."""
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def get_roc_curve(y_true, y_pred_proba):
    """Retorna (fpr, tpr, thresholds) para plotar a curva ROC."""
    return roc_curve(y_true, y_pred_proba)


def find_ks_optimal_threshold(y_true, y_pred_proba) -> float:
    """Retorna o ponto de corte (threshold) que maximiza a distancia KS
    (TPR - FPR) — o criterio mais usado na pratica de credito/cobranca para
    transformar a probabilidade prevista em uma decisao de negocio, em vez
    de usar o 0.5 padrao (que e arbitrario e nao reflete apetite de risco)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks_por_threshold = tpr - fpr
    best_idx = int(np.argmax(ks_por_threshold))
    return float(thresholds[best_idx])


def metrics_at_threshold(y_true, y_pred_proba, threshold: float) -> dict:
    """Recall/precisao/F1 em um ponto de corte especifico — usado para
    comparar o threshold padrao (0.5) com um threshold escolhido por
    criterio de negocio (ex.: o threshold otimo de KS)."""
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def get_precision_recall_curve(y_true, y_pred_proba):
    """Retorna (precision, recall, thresholds) para plotar a curva
    Precisao-Recall — visualiza o trade-off completo entre as duas metricas
    conforme o threshold varia."""
    return precision_recall_curve(y_true, y_pred_proba)


def explain_model(pipeline, X_sample: pd.DataFrame, sample_size: int = 500):
    """Gera explicacoes SHAP (global) para o modelo treinado dentro de um
    pipeline (preprocessing -> smote -> classifier).

    O SHAP precisa dos dados ja pre-processados (na mesma escala vista pelo
    classificador) e dos nomes das colunas pos-transformacao. Por isso
    extraimos o step de preprocessing do pipeline (ja ajustado no treino) e
    aplicamos manualmente antes de chamar o explainer - o step de SMOTE nao
    entra aqui, pois so existe para balancear o treino, nunca para gerar
    explicacoes.
    """
    preprocessing = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    X_sample = X_sample.sample(min(sample_size, len(X_sample)), random_state=42)
    X_transformed = preprocessing.transform(X_sample)
    feature_names = preprocessing.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X_sample.index)

    explainer = shap.Explainer(classifier, X_transformed_df)
    shap_values = explainer(X_transformed_df)

    # Dependendo da versao do SHAP/do modelo, classificadores binarios podem
    # retornar um array com uma dimensao extra por classe (n, features, 2).
    # Quando isso acontece, ficamos apenas com a classe positiva (inadimplente).
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_transformed_df
