"""
genai_explainer.py

Camada opcional de IA generativa: traduz os indicadores de negocio
(Roll Rate, Aging, Taxa de Cura, distribuicao de score) em um resumo
executivo em linguagem natural, para publico nao-tecnico.

Reaproveita a mesma integracao usada no projeto EcoStream Insight
(Llama 3.1 via Groq).

Requer a variavel de ambiente GROQ_API_KEY (ver .env.example).

TODO:
    - Implementar build_prompt(indicadores: dict) -> str
    - Implementar generate_executive_summary(indicadores: dict) -> str
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def build_prompt(indicadores: dict) -> str:
    """Monta o prompt com os indicadores de negocio para o modelo de linguagem."""
    raise NotImplementedError("TODO: implementar construcao do prompt.")


def generate_executive_summary(indicadores: dict) -> str:
    """Gera um resumo executivo em linguagem natural a partir dos indicadores."""
    raise NotImplementedError("TODO: implementar chamada ao Groq/Llama 3.1.")


if __name__ == "__main__":
    # Exemplo de uso (a implementar):
    # indicadores = {"roll_rate_30_60": 0.18, "taxa_cura": 0.42, ...}
    # print(generate_executive_summary(indicadores))
    pass
