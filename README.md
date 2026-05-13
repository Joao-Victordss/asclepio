# classificador-mastite-iot

Projeto Python para treinar e avaliar modelos de classificação de mastite bovina a partir de dados de sensores do úbere, leitura térmica da região mamária e meses pós-parto.

A metodologia foi inspirada no artigo “MasPA: A Machine Learning Application to Predict Risk of Mastitis in Cattle from AMS Sensor Data” (AgriEngineering, 2021, DOI: https://doi.org/10.3390/agriengineering3030037), adaptada para este repositório.

## Estrutura

```text
classificador-mastite-iot/
├─ dados/
│  ├─ bruto/mastite_iot_bruto.csv
│  └─ processado/
│     ├─ mastite_iot_tratado.csv
│     └─ mastite_iot_balanceado.csv
├─ modelos/
│  ├─ random_forest_mastite.pkl.gz
│  └─ random_forest_mastite_relatorio.json
├─ src/
│  ├─ dados/preparar_base.py
│  └─ modelos/treinar_random_forest.py
├─ requirements.txt
└─ README.md
```

## Ambiente

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Preparação dos dados

A base bruta esperada fica em:

```text
dados/bruto/mastite_iot_bruto.csv
```

A base MasPA/Mendeley usa `class1=1` para mastite e `class1=0` para saudável. O script de preparação converte para a convenção usada no treinamento:

- `classe=0`: mastite
- `classe=1`: saudável

O script também preserva `Cow_ID`, usado para separar treino/teste por animal e evitar que registros da mesma vaca apareçam nos dois conjuntos.

```bash
python src/dados/preparar_base.py
```

Saídas:

- `dados/processado/mastite_iot_tratado.csv`
- `dados/processado/mastite_iot_balanceado.csv`

## Treinamento

Treino rápido priorizando recall de mastite, com features derivadas e comparação entre Random Forest, Extra Trees e Balanced Random Forest:

```bash
python src/modelos/treinar_random_forest.py --feature-set engineered --search-mode quick --refit-metric recall_mastite --models all --jobs -1
```

Busca mais ampla:

```bash
python src/modelos/treinar_random_forest.py --feature-set engineered --search-mode standard --refit-metric recall_mastite --models all --jobs -1
```

O treino exige `Cow_ID` por padrão e usa `StratifiedGroupKFold` para avaliação por animal. Para bases sem identificador de animal, use explicitamente:

```bash
python src/modelos/treinar_random_forest.py --allow-row-split
```

## Saídas do modelo

- `modelos/random_forest_mastite.pkl.gz`: modelo treinado comprimido.
- `modelos/random_forest_mastite_relatorio.json`: métricas, hiperparâmetros, comparação de modelos, análise de limiares e importância das features.

Último treino registrado:

- Holdout por animal (`Cow_ID`)
- Melhor modelo: `random_forest`
- Features: `engineered` (`33` colunas)
- Acurácia: `98,56%`
- Sensibilidade mastite: `97,49%`
- Especificidade saudável: `99,34%`
- Validação cruzada por animal: `99,50% ± 0,44%`
- Recall mastite na validação cruzada: `99,39% ± 0,94%`

No holdout, a matriz de confusão foi:

```text
[[544, 14],
 [  5, 757]]
```

Com limiar conservador de revisão em `0.25`, os falsos negativos de mastite caem de `14` para `3`, com `22` falsos alertas em saudáveis.

## Limitações

- O modelo é uma ferramenta de apoio à triagem, não diagnóstico definitivo.
- A avaliação correta deve ser feita por animal (`Cow_ID`), não por linha.
- Ainda não há explicabilidade por instância (SHAP/LIME).
- Próxima etapa planejada: calibrar limiares de decisão e adicionar explicabilidade por instância.

## Referências

- GHAFOOR, Naeem Abdul; SITKOWSKA, Beata. MasPA: a machine learning application to predict risk of mastitis in cattle from AMS sensor data. *AgriEngineering*, Basel, v. 3, n. 3, p. 575-583, 2021. DOI: 10.3390/agriengineering3030037.
- Repositório original relacionado: https://github.com/naeemmrz/MasPA.py
