"""
data_ingestion.py

Carrega o dataset "Give Me Some Credit" (Kaggle) a partir de data/raw/ e
realiza a primeira etapa de limpeza.

Fonte oficial dos dados (requer conta gratuita no Kaggle):
    https://www.kaggle.com/c/GiveMeSomeCredit/data
Salve "cs-training.csv" em data/raw/ antes de rodar este modulo.

Colunas do dataset:
    SeriousDlqin2yrs                       -> variavel alvo (1 = ficou 90+ dias
                                               em atraso ou pior nos ultimos 2 anos)
    RevolvingUtilizationOfUnsecuredLines   -> uso do limite rotativo (esperado 0-1,
                                               mas ha outliers extremos no dataset)
    age                                     -> idade (ha 1 registro conhecido com age=0)
    NumberOfTime30-59DaysPastDueNotWorse    -> nº de vezes com atraso de 30-59 dias
    DebtRatio                               -> razao divida/renda
    MonthlyIncome                           -> renda mensal (~20% de valores ausentes)
    NumberOfOpenCreditLinesAndLoans         -> nº de linhas de credito abertas
    NumberOfTimes90DaysLate                 -> nº de vezes com atraso de 90+ dias
    NumberRealEstateLoansOrLines            -> nº de emprestimos/linhas imobiliarias
    NumberOfTime60-89DaysPastDueNotWorse    -> nº de vezes com atraso de 60-89 dias
    NumberOfDependents                      -> nº de dependentes (~2,5% de valores ausentes)

Problemas de qualidade de dados conhecidos neste dataset (documentados na
comunidade do Kaggle) e tratados em clean_data():
    - Os tres campos de "numero de vezes em atraso" possuem um pequeno grupo
      de registros com valores 96 ou 98, que sao codigos de erro do sistema
      de origem, nao contagens reais -> tratados como valores ausentes.
    - Existe 1 registro com age igual a 0 -> removido.
"""

from pathlib import Path

import pandas as pd

RAW_COLUMNS_TO_DROP = ["Unnamed: 0"]
ERROR_CODES_PAST_DUE = [96, 98]
PAST_DUE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


def load_raw_data(path: str = "data/raw/cs-training.csv") -> pd.DataFrame:
    """Carrega o dataset bruto de credito a partir de um arquivo CSV local.

    Aceita tanto um caminho local (recomendado, apos baixar do Kaggle) quanto
    uma URL http(s) apontando para um CSV no mesmo formato.
    """
    path_str = str(path)
    if path_str.startswith("http://") or path_str.startswith("https://"):
        df = pd.read_csv(path_str)
    else:
        file_path = Path(path_str)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado em '{file_path}'. Baixe 'cs-training.csv' em "
                "https://www.kaggle.com/c/GiveMeSomeCredit/data e salve em data/raw/."
            )
        df = pd.read_csv(file_path)

    existing_cols_to_drop = [c for c in RAW_COLUMNS_TO_DROP if c in df.columns]
    if existing_cols_to_drop:
        df = df.drop(columns=existing_cols_to_drop)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza basica conhecida para o dataset Give Me Some Credit.

    - Converte codigos de erro (96, 98) nos campos de atraso para NaN.
    - Remove o registro com age == 0 (erro de cadastro conhecido no dataset).
    - Nao imputa MonthlyIncome/NumberOfDependents aqui: a estrategia de
      imputacao e decidida na etapa de engenharia de atributos (features.py),
      para manter esta funcao focada em correcao de erros, nao em modelagem.
    """
    df = df.copy()

    for col in PAST_DUE_COLUMNS:
        if col in df.columns:
            df.loc[df[col].isin(ERROR_CODES_PAST_DUE), col] = pd.NA

    if "age" in df.columns:
        df = df[df["age"] > 0]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df_raw = load_raw_data("data/raw/cs-training.csv")
    df_clean = clean_data(df_raw)
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path / "credit_clean.csv", index=False)
    print(f"Dados tratados salvos em {output_path / 'credit_clean.csv'} ({len(df_clean)} linhas).")
