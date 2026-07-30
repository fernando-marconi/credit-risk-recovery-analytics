"""
business_metrics.py

Traduz as previsoes do modelo em indicadores usados por times de credito e
recuperacao (collections):

    - Roll Rate: probabilidade de um cliente migrar de uma faixa de atraso
      para a proxima (ex.: de 30 para 60 dias).
    - Aging de Carteira: distribuicao da carteira por tempo de atraso
      (0-29, 30-59, 60-89, 90+ dias).
    - Taxa de Cura: percentual de clientes inadimplentes que regularizam a
      situacao sem necessidade de acao de cobranca.

Estes indicadores sao o que diferencia este projeto de um exercicio puro de
classificacao: aqui o objetivo e apoiar decisao de negocio, nao so prever.

TODO:
    - Implementar calculate_roll_rate(df) -> pd.DataFrame
    - Implementar calculate_aging(df) -> pd.DataFrame
    - Implementar calculate_cure_rate(df) -> float
    - Exportar os indicadores para data/processed/ (consumo pelo Power BI)
"""


def calculate_roll_rate(df):
    raise NotImplementedError("TODO: implementar calculo de roll rate.")


def calculate_aging(df):
    raise NotImplementedError("TODO: implementar calculo de aging de carteira.")


def calculate_cure_rate(df):
    raise NotImplementedError("TODO: implementar calculo de taxa de cura.")
