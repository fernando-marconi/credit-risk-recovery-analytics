# Dashboard Power BI

Este diretorio guarda o arquivo `dashboard.pbix` do projeto (a construir no Power BI Desktop, gratuito).

## Como montar o dashboard

1. Exporte os dados processados (incluindo os indicadores calculados em `src/business_metrics.py`) para `data/processed/`.
2. Abra o Power BI Desktop e conecte-o aos arquivos CSV/Parquet dessa pasta.
3. Paginas sugeridas para o dashboard:
   - **Visao Geral da Carteira**: volume de clientes, valor total, distribuicao por faixa de risco (score do modelo).
   - **Indicadores de Recuperacao**: Roll Rate por faixa de atraso, Aging de Carteira, Taxa de Cura ao longo do tempo.
   - **Explicabilidade**: principais variaveis que influenciam o score (a partir da saida do SHAP).
4. Salve o arquivo como `dashboard.pbix` nesta pasta.
5. Para compartilhar o resultado sem precisar de licenca Power BI Pro:
   - Use "Arquivo > Publicar na Web" (gera um link publico, ideal para o README e o LinkedIn), ou
   - Grave um GIF/video curto navegando pelo dashboard e salve em `/images` para usar na documentacao.

> O arquivo `.pbix` nao e um formato de texto, entao nao ha como gerar um "esqueleto" dele automaticamente — esta pasta guarda apenas as instrucoes ate que o dashboard seja construido.
