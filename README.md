# classificador-mastite-iot

Projeto em Python para apoiar a triagem de mastite em vacas leiteiras usando dados de sensores de úbere e temperatura. A metodologia de preparação de dados e escolha do modelo é inspirada no artigo “MasPA: A Machine Learning Application to Predict Risk of Mastitis in Cattle from AMS Sensor Data” (AgriEngineering, 2021, DOI: https://doi.org/10.3390/agriengineering3030037), adaptada para este repositório.

## Estrutura do projeto
```
classificador-mastite-iot/
├─ main.py                            # API FastAPI + servidor do frontend
├─ static/                            # HTML, CSS e JavaScript da interface
├─ images/                            # logos e ícones da interface
├─ modelos/                           # modelo treinado (.pkl.gz ou .pkl)
├─ src/
│  ├─ dados/preparar_base.py          # preparação da base (apoio acadêmico)
│  └─ modelos/treinar_random_forest.py# treino do Random Forest
├─ requirements.txt
├─ exemplo_entrada.csv                # exemplo de entrada para inferência
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

## Execução da aplicação
1) Garanta que o modelo `modelos/random_forest_mastite.pkl.gz` exista no projeto.
   - O arquivo `modelos/random_forest_mastite.pkl` continua aceito como fallback.
2) Inicie a aplicação:
```bash
uvicorn main:app --reload
```
3) Abra `http://127.0.0.1:8000`.
4) Use a interface web para:
   - enviar um CSV;
   - preencher dados manualmente;
   - baixar o CSV de exemplo;
   - exportar o resultado da classificação em CSV ou XLSX;
   - escolher o delimitador do CSV na exportação;
   - receber uma triagem conservadora com nível de risco e recomendação de revisão.

### Modelo de arquivo para upload
- Inclua uma coluna `ID` e as features numéricas: `Months_after_giving_birth, IUFL, EUFL, IUFR, EUFR, IURL, EURL, IURR, EURR, Temperature`.
- Use ponto como separador decimal e vírgula como separador de coluna (CSV padrão).
- Há um botão na interface para baixar `exemplo_entrada.csv`, que já está no repositório.

## Formato esperado dos dados
- O CSV deve conter uma coluna `ID` (identificador do animal) e as features numéricas:
  `Months_after_giving_birth, IUFL, EUFL, IUFR, EUFR, IURL, EURL, IURR, EURR, Temperature`
- Exemplos:
  - Exemplo de entrada: `exemplo_entrada.csv`
  - Sinal do modelo: `Mastite` ou `Saudável`
  - Triagem principal: `Alta suspeita`, `Monitorar com cautela` ou `Baixo risco`
  - Probabilidades: `prob_mastite` (0 = mastite), `prob_saudavel` (1 = saudável)
- O app valida tipos numéricos e recusa linhas com valores vazios/inválidos.

## Endpoints principais
- `GET /` serve a interface web.
- `GET /exemplo` baixa o CSV de exemplo.
- `POST /predict` recebe um CSV via `multipart/form-data`.
- `POST /predict/manual` recebe dados manuais em JSON.
- `POST /export/results` exporta os resultados em CSV ou XLSX.

## Treinamento do modelo
Os scripts de preparação e treino continuam no repositório como apoio técnico do TCC, mas não fazem parte do fluxo principal da aplicação web.

Para atualizar o modelo com foco em não deixar casos de mastite passarem:
```bash
python src/dados/preparar_base.py
python src/modelos/treinar_random_forest.py --feature-set full --search-mode quick --refit-metric recall_mastite --jobs -1
```

Se quiser uma busca mais completa:
```bash
python src/modelos/treinar_random_forest.py --feature-set full --search-mode standard --refit-metric recall_mastite --jobs -1
```

O script salva:
- `modelos/random_forest_mastite.pkl.gz`
- `modelos/random_forest_mastite_relatorio.json`

## Limitações conhecidas
- O modelo foi organizado como ferramenta de triagem conservadora; o resultado deve ser usado junto com avaliação clínica e rotina operacional da fazenda.  
- Não há explicabilidade por instância (SHAP/LIME).  
- Dados de treino não acompanham o repositório por privacidade.  
- O app não substitui diagnóstico veterinário.

## Referências
- GHAFOOR, Naeem Abdul; SITKOWSKA, Beata. MasPA: a machine learning application to predict risk of mastitis in cattle from AMS sensor data. *AgriEngineering*, Basel, v. 3, n. 3, p. 575-583, 2021. DOI: 10.3390/agriengineering3030037. Acesso em: 10 dez. 2025.
- Repositório original relacionado: https://github.com/naeemmrz/MasPA.py
