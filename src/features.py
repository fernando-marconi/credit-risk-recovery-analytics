"""
features.py

Engenharia de atributos para o modelo de risco de credito.

Este modulo e dividido em duas partes, por design:

1. Atributos derivados "seguros" (add_derived_features): transformacoes que
   nao dependem de estatisticas calculadas sobre os dados (ex.: somas,
   flags, razoes simples). Podem ser aplicadas antes do split treino/teste
   sem risco de vazamento de dados (data leakage).

2. Pipeline de pre-processamento "sensivel" (build_preprocessing_pipeline):
   imputacao de valores ausentes e tratamento de outliers, que DEVEM ser
   ajustados (fit) apenas nos dados de treino e depois aplicados (transform)
   no teste, para nao vazar informacao do conjunto de teste para o treino.
   Por isso esta etapa e implementada como um sklearn Pipeline/ColumnTransformer,
   usado dentro de 03_modelagem.ipynb, e nao aplicado diretamente aqui.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
OUTLIER_COLUMNS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
# MonthlyIncome/NumberOfDependents: ausentes originais do dataset.
# PAST_DUE_COLUMNS entram aqui tambem porque clean_data() converteu os
# codigos de erro (96/98) em NaN - sem imputa-los, modelos que nao aceitam
# NaN (Regressao Logistica, Random Forest) quebrariam no .fit().
IMPUTE_COLUMNS = ["MonthlyIncome", "NumberOfDependents"] + PAST_DUE_COLUMNS


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos derivados que nao dependem de estatisticas do dataset
    (seguros para aplicar antes do split treino/teste)."""
    df = df.copy()

    # Flags de valor ausente (o padrao de ausencia pode ser informativo)
    df["MonthlyIncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependentsMissing"] = df["NumberOfDependents"].isna().astype(int)

    # Total de vezes em atraso (soma das 3 faixas). NaN (ex-codigos de erro)
    # tratado como 0 apenas para esta soma; a coluna original permanece
    # intacta para a etapa de imputacao no pipeline de modelagem.
    df["TotalTimesPastDue"] = df[PAST_DUE_COLUMNS].fillna(0).sum(axis=1)
    df["EverPastDue"] = (df["TotalTimesPastDue"] > 0).astype(int)

    # Flag de utilizacao acima de 100% (valor teoricamente invalido, mas
    # presente no dataset - pode ser um sinal de risco por si so)
    df["UtilizationOver1"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)

    df["HasDependents"] = (df["NumberOfDependents"].fillna(0) > 0).astype(int)
    df["HasRealEstateLoan"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(int)

    return df


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita (capa) valores extremos com base em percentis aprendidos no fit.

    Implementado como transformer do scikit-learn para que os limites sejam
    calculados apenas no conjunto de treino (fit) e depois aplicados de forma
    consistente ao conjunto de teste (transform), evitando vazamento de dados.
    """

    def __init__(self, lower_quantile: float = 0.0, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Monta o ColumnTransformer de pre-processamento sensivel a vazamento:
    imputacao (mediana) para colunas com ausentes e winsorizacao (cap por
    percentil) para colunas com outliers extremos. Deve ser ajustado (fit)
    apenas em X_train, dentro de um Pipeline junto com o modelo.
    """
    return ColumnTransformer(
        transformers=[
            ("impute_income_dependents", SimpleImputer(strategy="median"), IMPUTE_COLUMNS),
            ("winsorize_outliers", Winsorizer(upper_quantile=0.99), OUTLIER_COLUMNS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ponto de entrada usado pelo notebook: aplica apenas os atributos
    derivados seguros. A imputacao/winsorizacao acontece no pipeline de
    modelagem (ver build_preprocessing_pipeline), nao aqui.
    """
    return add_derived_features(df)
