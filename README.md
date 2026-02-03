# classificador-mastite-iot

Projeto em Python para classificar risco de mastite em vacas leiteiras usando dados de sensores de úbere e temperatura. A metodologia de preparação de dados e escolha do modelo é inspirada no artigo “MasPA: A Machine Learning Application to Predict Risk of Mastitis in Cattle from AMS Sensor Data” (AgriEngineering, 2021, DOI: https://doi.org/10.3390/agriengineering3030037), adaptada para este repositório.

## Estrutura do projeto
```
classificador-mastite-iot/
├─ dados/
│  ├─ bruto/
│  │  └─ mastite_iot_bruto.csv        # substituir pelo CSV bruto original
│  └─ processado/                     # gerado pelos scripts
├─ modelos/                           # modelos treinados (.pkl)
├─ src/
│  ├─ dados/preparar_base.py          # prepara e balanceia os dados
│  └─ modelos/treinar_random_forest.py# treina e avalia o modelo
├─ exemplo_entrada.csv                # exemplo de entrada para inferência
├─ app_streamlit.py                   # app web para inferência
├─ requirements.txt
└─ README.md
```

## Preparação do ambiente local
1) Criar e ativar o ambiente virtual (Windows PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
2) Instalar as dependências:
```bash
pip install -r requirements.txt
```

## Fluxo de dados e treino
1) Coloque o CSV bruto em `dados/bruto/mastite_iot_bruto.csv` (mesmos nomes de features do artigo ou ajuste o mapa de renomeação em `src/dados/preparar_base.py`).
   - Os dados brutos e processados **não são versionados**; mantenha-os fora do repositório remoto.
2) Gerar bases tratada e balanceada:
```bash
python src/dados/preparar_base.py
```
Saídas: `dados/processado/mastite_iot_tratado.csv` e `dados/processado/mastite_iot_balanceado.csv`.
3) Treinar o modelo Random Forest:
```bash
python src/modelos/treinar_random_forest.py
```
Saída: `modelos/random_forest_mastite.pkl.gz` (gzip, menor para deploy).

## Uso da aplicação (inferência)
1) Garantir que o modelo `modelos/random_forest_mastite.pkl.gz` está no repositório (ou `random_forest_mastite.pkl` como fallback).  
2) Rodar o app:
```bash
streamlit run app_streamlit.py
```
3) Enviar um CSV no formato descrito abaixo e visualizar o resultado na tabela.

### Modelo de arquivo para upload
- Inclua uma coluna `ID` e as features numéricas: `Months_after_giving_birth, IUFL, EUFL, IUFR, EUFR, IURL, EURL, IURR, EURR, Temperature`.
- Use ponto como separador decimal e vírgula como separador de coluna (CSV padrão).
- Há um botão no app para baixar `exemplo_entrada.csv`, que já está no repositório.

## Formato esperado dos dados
- O CSV deve conter uma coluna `ID` (identificador do animal) e as features numéricas:
  `Months_after_giving_birth, IUFL, EUFL, IUFR, EUFR, IURL, EURL, IURR, EURR, Temperature`
- Exemplos:
  - Exemplo de entrada: `exemplo_entrada.csv`
  - Classe prevista: `Mastite` ou `Saudável`
  - Probabilidades: `prob_mastite` (0 = mastite), `prob_saudavel` (1 = saudável)
- O app valida tipos numéricos e recusa linhas com valores vazios/inválidos.

## Deploy no Streamlit Community Cloud
1) Suba o repositório no GitHub incluindo `modelos/random_forest_mastite.pkl.gz`.
2) Em https://share.streamlit.io crie um novo app apontando para este repo e o arquivo `app_streamlit.py`.
3) A configuração de tema e limites já está em `.streamlit/config.toml` (upload até 200 MB; app aplica limite lógico de 5 MB).

## Limitações conhecidas
- O modelo foi treinado com Random Forest simples; não há comparação com modelos mais leves (ex.: LightGBM).  
- Não há explicabilidade por instância (SHAP/LIME).  
- Dados de treino não acompanham o repositório por privacidade; reproduza o treino localmente se precisar re-treinar.  
- O app não substitui diagnóstico veterinário.

## Referências
- GHAFOOR, Naeem Abdul; SITKOWSKA, Beata. MasPA: a machine learning application to predict risk of mastitis in cattle from AMS sensor data. *AgriEngineering*, Basel, v. 3, n. 3, p. 575-583, 2021. DOI: 10.3390/agriengineering3030037. Acesso em: 10 dez. 2025.
- Repositório original relacionado: https://github.com/naeemmrz/MasPA.py
