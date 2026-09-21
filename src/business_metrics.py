"""
business_metrics.py

Traduz o historico de atraso dos clientes em indicadores usados por times de
credito e recuperacao (collections): Roll Rate, Aging de Carteira e Taxa de
Cura.

Limitacao importante dos dados (documentada de forma transparente, como
qualquer analise de credito deveria fazer): o dataset "Give Me Some Credit"
e uma fotografia unica de cada cliente (uma linha por pessoa), sem
historico mes a mes da mesma pessoa ao longo do tempo. Por isso, Roll Rate
e Aging de Carteira - que classicamente comparam o mesmo cliente em dois
periodos consecutivos - sao aproximados aqui a partir das tres colunas de
contagem de atraso (30-59, 60-89, 90+ dias), que registram quantas vezes
cada cliente ja esteve em cada faixa ao longo do seu historico. Cada
funcao abaixo documenta exatamente qual aproximacao esta sendo usada.
"""

import pandas as pd

from src.features import PAST_DUE_COLUMNS

BUCKET_30, BUCKET_60, BUCKET_90 = PAST_DUE_COLUMNS
TARGET_COL = "SeriousDlqin2yrs"


def calculate_roll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula o Roll Rate aproximado entre faixas de atraso.

    Como o dataset nao acompanha o mesmo cliente em dois meses consecutivos,
    o Roll Rate aqui e uma proxy cumulativa: entre os clientes que ja
    estiveram alguma vez na faixa de origem, qual percentual tambem ja
    esteve na faixa seguinte (mais grave)? Isso mede a tendencia de
    deterioracao ao longo do historico do cliente - nao uma transicao
    mes a mes real, que exigiria dados em painel (mesmo cliente, varios
    meses), que este dataset nao possui.

    Retorna um DataFrame com uma linha por transicao (de -> para).
    """
    reached_30 = df[BUCKET_30].fillna(0) >= 1
    reached_60 = df[BUCKET_60].fillna(0) >= 1
    reached_90 = df[BUCKET_90].fillna(0) >= 1

    transitions = [
        {
            "de": "30-59 dias",
            "para": "60-89 dias",
            "clientes_na_faixa_origem": int(reached_30.sum()),
            "clientes_que_progrediram": int((reached_30 & reached_60).sum()),
        },
        {
            "de": "60-89 dias",
            "para": "90+ dias",
            "clientes_na_faixa_origem": int(reached_60.sum()),
            "clientes_que_progrediram": int((reached_60 & reached_90).sum()),
        },
    ]

    result = pd.DataFrame(transitions)
    result["roll_rate"] = (
        result["clientes_que_progrediram"] / result["clientes_na_faixa_origem"]
    ).round(4)
    return result


def calculate_aging(df: pd.DataFrame) -> pd.DataFrame:
    """Distribui a carteira pela faixa de atraso mais grave ja atingida por
    cada cliente (aging de carteira), a partir das tres colunas de contagem.

    Cada cliente entra em uma unica categoria - a pior que ele ja atingiu -
    da mais leve para a mais grave: "Nunca atrasou", "30-59 dias",
    "60-89 dias", "90+ dias".
    """
    reached_30 = df[BUCKET_30].fillna(0) >= 1
    reached_60 = df[BUCKET_60].fillna(0) >= 1
    reached_90 = df[BUCKET_90].fillna(0) >= 1

    def worst_bucket(is_30, is_60, is_90):
        if is_90:
            return "90+ dias"
        if is_60:
            return "60-89 dias"
        if is_30:
            return "30-59 dias"
        return "Nunca atrasou"

    faixas = [
        worst_bucket(a, b, c) for a, b, c in zip(reached_30, reached_60, reached_90)
    ]

    aging = pd.Series(faixas, name="faixa_aging").value_counts().reset_index()
    aging.columns = ["faixa_aging", "quantidade_clientes"]
    aging["percentual"] = (aging["quantidade_clientes"] / len(df) * 100).round(2)

    ordem = ["Nunca atrasou", "30-59 dias", "60-89 dias", "90+ dias"]
    aging["faixa_aging"] = pd.Categorical(
        aging["faixa_aging"], categories=ordem, ordered=True
    )
    return aging.sort_values("faixa_aging").reset_index(drop=True)


def calculate_cure_rate(df: pd.DataFrame) -> float:
    """Taxa de Cura: entre os clientes que ja tiveram algum atraso no
    historico (30, 60 ou 90+ dias em algum momento), qual percentual NAO
    se tornou inadimplente grave nos 2 anos seguintes
    (SeriousDlqin2yrs == 0)? Esses clientes regularizaram a situacao por
    conta propria, sem chegar ao pior desfecho que o modelo tenta prever.
    """
    ever_past_due = (
        (df[BUCKET_30].fillna(0) >= 1)
        | (df[BUCKET_60].fillna(0) >= 1)
        | (df[BUCKET_90].fillna(0) >= 1)
    )

    total_com_historico_atraso = int(ever_past_due.sum())
    if total_com_historico_atraso == 0:
        return 0.0

    curados = int((ever_past_due & (df[TARGET_COL] == 0)).sum())
    return round(curados / total_com_historico_atraso, 4)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/credit_features.csv")

    print("Roll Rate:")
    print(calculate_roll_rate(df))

    print("\nAging de Carteira:")
    print(calculate_aging(df))

    taxa_cura = calculate_cure_rate(df)
    print(f"\nTaxa de Cura: {taxa_cura:.2%}")
