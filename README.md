# Análise de Risco de Crédito e Recuperação (Credit Risk & Recovery Analytics)

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit--learn](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/Explicabilidade-SHAP-8A2BE2.svg)](https://shap.readthedocs.io/)
[![Power BI](https://img.shields.io/badge/Dashboard-Power%20BI-F2C811.svg)](https://powerbi.microsoft.com/)
[![IA Generativa](https://img.shields.io/badge/IA-Llama%203.1-0668E1.svg)](https://groq.com/)

Projeto de ponta a ponta que simula a rotina de um analista de dados em um time de **crédito e recuperação (collections)**: previsão de inadimplência, cálculo dos indicadores de negócio usados no setor bancário e apresentação dos resultados em um dashboard executivo, com uma camada opcional de IA generativa para tradução dos números em linguagem de negócio.

## Contexto e Motivação

Instituições financeiras precisam decidir, todos os dias, quais clientes priorizar nas ações de cobrança e como calibrar o apetite de risco na concessão de crédito. Esse projeto reproduz esse cenário: a partir de dados históricos de clientes, o objetivo é estimar a probabilidade de inadimplência e transformar essa previsão em indicadores acionáveis para o negócio — não apenas em uma métrica de modelo isolada.

## Problema de Negócio

- Prever a probabilidade de um cliente entrar em atraso/inadimplência nos próximos meses.
- Segmentar a carteira por faixas de risco para orientar priorização de cobrança.
- Calcular indicadores de recuperação usados no mercado: **Roll Rate**, **Aging de Carteira** e **Taxa de Cura**.
- Explicar as decisões do modelo de forma compreensível para áreas de negócio (explicabilidade).

## Fonte de Dados

- Base pública de crédito ao consumidor ([Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), Kaggle) como base principal de modelagem.
- Séries e indicadores públicos do Banco Central do Brasil ([API SGS](https://dadosabertos.bcb.gov.br/)) como contexto complementar de mercado (ex.: inadimplência agregada, Selic).

## Arquitetura do Pipeline

O projeto é estruturado em cinco etapas:

### 1. Ingestão e Tratamento
Carga dos dados brutos, tratamento de valores ausentes/outliers e engenharia de atributos (ex.: utilização de limite, histórico de atrasos, renda vs. dívida).

### 2. Modelagem Preditiva
Comparação entre um modelo baseline (Regressão Logística) e modelos de maior capacidade (Random Forest, XGBoost/LightGBM), com tratamento de desbalanceamento de classes (SMOTE), reaproveitando a abordagem já validada no projeto de Detecção de Fraudes.

### 3. Explicabilidade
Uso de SHAP para identificar quais variáveis mais influenciam cada previsão — essencial em crédito, onde decisões automatizadas precisam ser justificáveis.

### 4. Indicadores de Negócio (Recuperação de Crédito)
Cálculo dos indicadores usados por times de crédito/cobrança a partir das previsões do modelo:
- **Roll Rate**: probabilidade de um cliente migrar de uma faixa de atraso para a próxima (ex.: de 30 para 60 dias).
- **Aging de Carteira**: distribuição da carteira por tempo de atraso.
- **Taxa de Cura**: percentual de clientes inadimplentes que regularizam a situação sem ação de cobrança.

### 5. Dashboard e Camada de IA Generativa
- Painel em **Power BI** com a visão consolidada da carteira, score de risco por segmento e evolução dos indicadores.
- Camada opcional de IA generativa (Llama 3.1 via Groq, mesma integração usada no projeto EcoStream Insight) que traduz os indicadores do dashboard em um resumo executivo em linguagem natural.

## Especificações Técnicas

### Requisitos de Software
- Python 3.9 ou superior
- Bibliotecas: Pandas, Numpy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM, SHAP, Matplotlib, Seaborn, Jupyter, python-dotenv, Groq
- Power BI Desktop (gratuito) para o dashboard

### Estrutura de Diretórios
- `/data/raw`: dados brutos (não versionados no Git — ver `.gitignore`)
- `/data/processed`: dados tratados, prontos para modelagem e para o Power BI
- `/notebooks`: exploração, engenharia de atributos, modelagem e explicabilidade
- `/src`: código-fonte modular (ingestão, features, treino, avaliação, métricas de negócio, camada de IA generativa)
- `/models`: modelos treinados serializados
- `/powerbi`: arquivo `.pbix` do dashboard e instruções de uso
- `/images`: capturas de tela do dashboard e gráficos para documentação
- `/docs`: material de divulgação do projeto (ex.: post para LinkedIn)

## Segurança e Variáveis de Ambiente

Assim como no projeto EcoStream Insight, nenhuma chave de API é versionada no repositório. Copie `.env.example` para `.env` e preencha localmente:
- `GROQ_API_KEY`: necessária apenas se a camada opcional de IA generativa for utilizada.

## Como Executar

1. Criar e ativar um ambiente virtual, e instalar as dependências:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Baixar a base "Give Me Some Credit" do Kaggle e salvar em `data/raw/`.
3. Executar os notebooks na ordem: `01_eda` → `02_feature_engineering` → `03_modelagem` → `04_explicabilidade_shap`.
4. Abrir `powerbi/dashboard.pbix` no Power BI Desktop, apontando para os dados processados em `data/processed/`.
5. (Opcional) Configurar `.env` com `GROQ_API_KEY` e executar `src/genai_explainer.py` para gerar o resumo executivo em linguagem natural.

## Resultados

Comparação dos três modelos no conjunto de teste (threshold padrão de 0,5):

| Modelo | AUC-ROC | KS | Recall | Precisão |
|---|---|---|---|---|
| **LightGBM** | **0,8627** | **0,5795** | 0,2125 | 0,5266 |
| Regressão Logística | 0,8613 | 0,5761 | 0,7541 | 0,2248 |
| Random Forest | 0,8602 | 0,5662 | 0,3761 | 0,4351 |

O LightGBM foi selecionado como modelo final por ter o melhor AUC-ROC e KS — as métricas corretas para comparar modelos, pois independem do ponto de corte escolhido. Um KS acima de 0,55 é considerado forte no mercado de crédito (a referência de "bom" geralmente começa em 0,4), e o AUC-ROC obtido está em linha com as melhores soluções da competição original no Kaggle.

Também foi adicionada uma análise do ponto de corte (threshold): em vez do 0,5 padrão, o notebook de modelagem calcula o threshold que maximiza o KS — a referência mais usada em comitês de crédito para transformar probabilidade em decisão — e recalcula recall/precisão nesse ponto, evidenciando que a escolha do corte é uma decisão de negócio (quantos inadimplentes aceito deixar passar em troca de aprovar mais clientes bons), não apenas uma saída do modelo.

**Resultado da análise de threshold (LightGBM):**

| Threshold | Recall | Precisão | F1 |
|---|---|---|---|
| Padrão (0,5) | 0,2125 | 0,5266 | 0,3028 |
| Ótimo (KS) — **0,0759** | **0,7875** | 0,2133 | **0,3357** |

O threshold ótimo ficou bem abaixo de 0,5 — comportamento esperado quando o SMOTE é usado no treino: ele balanceia as classes em 50/50 durante o `.fit()`, o que descalibra a probabilidade bruta prevista pelo modelo em relação à distribuição real do teste (~6-7% de inadimplentes). Por isso a fronteira de decisão "natural" aprendida no mundo balanceado do treino corresponde a um valor bem menor quando aplicada aos dados reais. No threshold ótimo, o modelo passa a capturar quase 8 em cada 10 futuros inadimplentes (recall de 78,75%), ao custo de precisão mais baixa (1 em cada 5 clientes sinalizados é de fato de risco) — um trade-off explícito de negócio, não uma limitação do modelo.

> Pendente: principais variáveis explicativas (SHAP) e capturas de tela do dashboard Power BI.

## Roadmap

- [x] EDA e engenharia de atributos
- [x] Modelagem e comparação de algoritmos (com análise de threshold)
- [ ] Explicabilidade com SHAP
- [ ] Cálculo dos indicadores de negócio (Roll Rate, Aging, Taxa de Cura)
- [ ] Dashboard em Power BI
- [ ] Camada de IA generativa para resumo executivo
- [ ] Publicação do post no LinkedIn

## Autor

**Fernando Marconi Veloso Ribeiro**
Cientista de Dados | Pós-graduado em Ciência de Dados e Big Data (PUC Minas)
[LinkedIn](https://www.linkedin.com/in/fernandomarconi) · [GitHub](https://github.com/fernando-marconi)
