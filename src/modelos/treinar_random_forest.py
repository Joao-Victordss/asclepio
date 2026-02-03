from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# Caminhos principais do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BASE_BALANCEADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_balanceado.csv"
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl.gz"


def carregar_dados(caminho: Path):
    """
    Lê a base balanceada e separa em X (features) e y (classe).

    Espera que:
    - a coluna de rótulo se chame 'classe'
    - as demais colunas sejam numéricas
    """
    df = pd.read_csv(caminho)

    if "classe" not in df.columns:
        raise ValueError("A coluna 'classe' não foi encontrada na base balanceada.")

    X = df.drop(columns=["classe"])
    y = df["classe"]

    return X, y


def treinar_e_avaliar(X, y):
    """
    Faz o split 80/20 estratificado e treina o Random Forest.

    Calcula e imprime:
    - acurácia
    - matriz de confusão
    - relatório de classificação
    - sensibilidade (classe 0 = mastite)
    - especificidade (classe 1 = saudável)
    """
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    modelo = RandomForestClassifier(
        criterion="entropy",
        n_estimators=100,
        random_state=1000,
    )

    modelo.fit(X_treino, y_treino)

    # Predição no conjunto de teste
    y_pred = modelo.predict(X_teste)

    # Métricas principais
    acuracia = accuracy_score(y_teste, y_pred)
    cm = confusion_matrix(y_teste, y_pred, labels=[0, 1])

    # Desempacotar matriz de confusão pensando em 0 = mastite, 1 = saudável
    # cm =
    # [[TP_mastite, FN_mastite],
    #  [FP_mastite, TN_saudavel]]
    TP_mastite = cm[0, 0]
    FN_mastite = cm[0, 1]
    FP_mastite = cm[1, 0]
    TN_saudavel = cm[1, 1]

    sens_mastite = TP_mastite / (TP_mastite + FN_mastite)
    espec_saudavel = TN_saudavel / (TN_saudavel + FP_mastite)

    print("=== MÉTRICAS DO MODELO RANDOM FOREST ===")
    print(f"Acurácia no conjunto de teste: {acuracia:.4f}")
    print("\nMatriz de confusão (linhas = verdadeiro, colunas = previsto):")
    print(cm)
    print("\nRelatório de classificação (sklearn):")
    print(classification_report(y_teste, y_pred, digits=4))

    print(f"\nSensibilidade (mastite = classe 0): {sens_mastite:.4f}")
    print(f"Especificidade (saudáveis = classe 1): {espec_saudavel:.4f}")

    return modelo


def salvar_modelo(modelo, caminho: Path):
    """
    Salva o modelo comprimido para reduzir tamanho do artefato de deploy.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho, compress=("gzip", 3))
    print(f"\nModelo salvo (gzip) em: {caminho}")


def main():
    X, y = carregar_dados(CAMINHO_BASE_BALANCEADA)
    modelo = treinar_e_avaliar(X, y)
    salvar_modelo(modelo, CAMINHO_MODELO)


if __name__ == "__main__":
    main()
