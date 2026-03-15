import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split


# Caminhos principais do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BASE_TRATADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_tratado.csv"
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl.gz"
CAMINHO_RELATORIO = RAIZ_PROJETO / "modelos" / "random_forest_mastite_relatorio.json"
COLUNA_ROTULO = "classe"

FEATURES_FULL = [
    "Months_after_giving_birth",
    "IUFL",
    "EUFL",
    "IUFR",
    "EUFR",
    "IURL",
    "EURL",
    "IURR",
    "EURR",
    "Temperature",
]
FEATURES_ARTIGO = [
    "IUFL",
    "EUFL",
    "IUFR",
    "EUFR",
    "IURL",
    "EURL",
    "IURR",
    "EURR",
    "Temperature",
]
FEATURE_SETS = {
    "full": FEATURES_FULL,
    "article": FEATURES_ARTIGO,
}
SEARCH_MODES = ("quick", "standard")
REFIT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_mastite",
    "recall_mastite",
    "specificity_saudavel",
)


def carregar_dados(caminho: Path, feature_set: str):
    """
    Lê a base tratada e separa em X (features) e y (classe).

    Espera que:
    - a coluna de rótulo se chame 'classe';
    - as features estejam no conjunto escolhido.
    """
    df = pd.read_csv(caminho)
    colunas_features = FEATURE_SETS[feature_set]

    if COLUNA_ROTULO not in df.columns:
        raise ValueError("A coluna 'classe' não foi encontrada na base tratada.")

    faltantes = [coluna for coluna in colunas_features if coluna not in df.columns]
    if faltantes:
        raise ValueError(
            f"As seguintes colunas de features não foram encontradas na base tratada: {faltantes}"
        )

    X = df.loc[:, colunas_features]
    y = df[COLUNA_ROTULO]

    return X, y


def criar_modelo(**kwargs) -> RandomForestClassifier:
    parametros = {
        "random_state": 1000,
        "n_jobs": 1,
    }
    parametros.update(kwargs)
    return RandomForestClassifier(
        **parametros,
    )


def criar_pipeline(**rf_kwargs) -> Pipeline:
    return Pipeline(
        steps=[
            ("oversampler", RandomOverSampler(random_state=42)),
            ("random_forest", criar_modelo(**rf_kwargs)),
        ]
    )


def obter_scorers() -> dict:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_mastite": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_mastite": make_scorer(recall_score, pos_label=0),
        "specificity_saudavel": make_scorer(recall_score, pos_label=1),
    }


def montar_grid(search_mode: str) -> list[dict]:
    if search_mode == "quick":
        return [
            {
                "random_forest__criterion": ["gini", "entropy"],
                "random_forest__n_estimators": [200, 400],
                "random_forest__max_features": ["sqrt", "log2"],
                "random_forest__max_depth": [None, 24],
                "random_forest__min_samples_split": [2, 5],
                "random_forest__min_samples_leaf": [1, 2],
            }
        ]

    return [
        {
            "random_forest__criterion": ["gini", "entropy"],
            "random_forest__n_estimators": [100, 200, 400],
            "random_forest__max_features": ["sqrt", "log2", None],
            "random_forest__max_depth": [None, 12, 24],
            "random_forest__min_samples_split": [2, 5],
            "random_forest__min_samples_leaf": [1, 2],
        }
    ]


def buscar_melhor_modelo(X_treino, y_treino, search_mode: str, refit_metric: str, n_jobs: int) -> GridSearchCV:
    scorers = obter_scorers()
    cv_busca = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    busca = GridSearchCV(
        estimator=criar_pipeline(),
        param_grid=montar_grid(search_mode),
        scoring=scorers,
        refit=refit_metric,
        cv=cv_busca,
        n_jobs=n_jobs,
        verbose=1,
    )
    busca.fit(X_treino, y_treino)
    return busca


def calcular_metricas_holdout(y_teste, y_pred) -> dict:
    acuracia = accuracy_score(y_teste, y_pred)
    cm = confusion_matrix(y_teste, y_pred, labels=[0, 1])

    tp_mastite = int(cm[0, 0])
    fn_mastite = int(cm[0, 1])
    fp_mastite = int(cm[1, 0])
    tn_saudavel = int(cm[1, 1])

    sens_mastite = tp_mastite / (tp_mastite + fn_mastite)
    espec_saudavel = tn_saudavel / (tn_saudavel + fp_mastite)

    return {
        "accuracy": float(acuracia),
        "sensitivity_mastite": float(sens_mastite),
        "specificity_saudavel": float(espec_saudavel),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_teste, y_pred, digits=4, output_dict=True),
    }


def avaliar_holdout(modelo, X_teste, y_teste) -> dict:
    y_pred = modelo.predict(X_teste)
    metricas = calcular_metricas_holdout(y_teste, y_pred)

    print("\n=== AVALIAÇÃO NO HOLDOUT (SEM VAZAMENTO) ===")
    print(f"Acurácia: {metricas['accuracy']:.4f}")
    print(f"Sensibilidade mastite: {metricas['sensitivity_mastite']:.4f}")
    print(f"Especificidade saudáveis: {metricas['specificity_saudavel']:.4f}")
    print("Matriz de confusão (linhas = verdadeiro, colunas = previsto):")
    print(metricas["confusion_matrix"])
    print("\nRelatório de classificação (sklearn):")
    print(classification_report(y_teste, y_pred, digits=4))

    return metricas


def avaliar_validacao_cruzada(X, y, melhores_parametros: dict, n_jobs: int) -> dict:
    pipeline = criar_pipeline(**melhores_parametros)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    resultados = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring=obter_scorers(),
        n_jobs=n_jobs,
    )

    metricas = {}
    print("\n=== VALIDAÇÃO CRUZADA (10 FOLDS, SEM VAZAMENTO) ===")
    for nome_metrica, valores in resultados.items():
        if not nome_metrica.startswith("test_"):
            continue
        nome_limpo = nome_metrica.removeprefix("test_")
        media = float(valores.mean())
        desvio = float(valores.std())
        metricas[nome_limpo] = {
            "mean": media,
            "std": desvio,
        }
        print(f"{nome_limpo}: {media:.4f} ± {desvio:.4f}")

    return metricas


def treinar_modelo_final(X, y, melhores_parametros: dict) -> RandomForestClassifier:
    oversampler = RandomOverSampler(random_state=42)
    X_balanceado, y_balanceado = oversampler.fit_resample(X, y)
    modelo = criar_modelo(**melhores_parametros)
    modelo.fit(X_balanceado, y_balanceado)
    return modelo


def salvar_modelo(modelo, caminho: Path):
    """
    Salva o modelo comprimido para reduzir tamanho do artefato de deploy.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho, compress=("gzip", 3))
    print(f"\nModelo salvo (gzip) em: {caminho}")


def salvar_relatorio(relatorio: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(relatorio, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Relatório salvo em: {caminho}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina e avalia o Random Forest para classificação de mastite."
    )
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="full",
        help="Conjunto de features usado no treino. 'full' inclui meses pós-parto; 'article' usa apenas as variáveis do artigo.",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default="quick",
        help="Tamanho da busca de hiperparâmetros. 'quick' é mais rápido; 'standard' é mais completo.",
    )
    parser.add_argument(
        "--refit-metric",
        choices=REFIT_METRICS,
        default="recall_mastite",
        help="Métrica usada para escolher o melhor modelo na grid search. O padrao prioriza nao deixar mastite passar.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Número de processos paralelos na busca e na validação cruzada. Use -1 para todos os núcleos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    X, y = carregar_dados(CAMINHO_BASE_TRATADA, args.feature_set)

    print("=== CONFIGURAÇÃO DO TREINO ===")
    print(f"Feature set: {args.feature_set}")
    print(f"Modo de busca: {args.search_mode}")
    print(f"Métrica de seleção: {args.refit_metric}")
    print(f"Jobs paralelos: {args.jobs}")
    print(f"Features usadas: {FEATURE_SETS[args.feature_set]}")

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    print(f"Treino: {X_treino.shape[0]} amostras | Teste holdout: {X_teste.shape[0]} amostras")

    busca = buscar_melhor_modelo(
        X_treino,
        y_treino,
        search_mode=args.search_mode,
        refit_metric=args.refit_metric,
        n_jobs=args.jobs,
    )
    melhores_parametros = {
        chave.removeprefix("random_forest__"): valor
        for chave, valor in busca.best_params_.items()
        if chave.startswith("random_forest__")
    }

    print("\n=== MELHORES HIPERPARÂMETROS ===")
    print(melhores_parametros)
    print(f"Melhor score ({args.refit_metric}) na busca: {busca.best_score_:.4f}")

    metricas_holdout = avaliar_holdout(busca.best_estimator_, X_teste, y_teste)
    metricas_cv = avaliar_validacao_cruzada(X, y, melhores_parametros, n_jobs=args.jobs)

    modelo_final = treinar_modelo_final(X, y, melhores_parametros)
    salvar_modelo(modelo_final, CAMINHO_MODELO)

    relatorio = {
        "feature_set": args.feature_set,
        "search_mode": args.search_mode,
        "refit_metric": args.refit_metric,
        "jobs": args.jobs,
        "features": FEATURE_SETS[args.feature_set],
        "best_params": melhores_parametros,
        "grid_search_best_score": float(busca.best_score_),
        "holdout_metrics": metricas_holdout,
        "cross_validation_metrics": metricas_cv,
        "scikit_learn_notes": {
            "max_features_auto_removed": True,
            "max_features_used_in_search": ["sqrt", "log2", None],
        },
    }
    salvar_relatorio(relatorio, CAMINHO_RELATORIO)


if __name__ == "__main__":
    main()
