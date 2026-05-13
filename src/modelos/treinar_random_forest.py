import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BASE_TRATADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_tratado.csv"
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl.gz"
CAMINHO_RELATORIO = RAIZ_PROJETO / "modelos" / "random_forest_mastite_relatorio.json"

COLUNA_ROTULO = "classe"
COLUNA_GRUPO = "Cow_ID"
RANDOM_STATE = 1000

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
SENSORES = ["IUFL", "EUFL", "IUFR", "EUFR", "IURL", "EURL", "IURR", "EURR"]
FEATURE_SETS = ("article", "full", "engineered")
SEARCH_MODES = ("quick", "standard")
REFIT_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision_mastite",
    "recall_mastite",
    "specificity_saudavel",
)
MODELOS = ("random_forest", "extra_trees", "balanced_random_forest")
LIMIARES_TRIAGEM = (0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10)


def carregar_dados(caminho: Path, feature_set: str):
    df = pd.read_csv(caminho)

    if COLUNA_ROTULO not in df.columns:
        raise ValueError("A coluna 'classe' não foi encontrada na base tratada.")

    grupos = df[COLUNA_GRUPO].astype(str) if COLUNA_GRUPO in df.columns else None
    X = montar_matriz_features(df, feature_set)
    y = df[COLUNA_ROTULO].astype(int)

    return X, y, grupos


def montar_matriz_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set == "article":
        colunas = FEATURES_ARTIGO
    else:
        colunas = FEATURES_FULL

    faltantes = [coluna for coluna in colunas if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"As seguintes features não foram encontradas na base: {faltantes}")

    X = df.loc[:, colunas].apply(pd.to_numeric, errors="raise").copy()
    if feature_set == "engineered":
        X = adicionar_features_derivadas(X)

    return X


def adicionar_features_derivadas(X: pd.DataFrame) -> pd.DataFrame:
    sensores = X[SENSORES]
    quartos = pd.DataFrame(
        {
            "quarter_fl_mean": X[["IUFL", "EUFL"]].mean(axis=1),
            "quarter_fr_mean": X[["IUFR", "EUFR"]].mean(axis=1),
            "quarter_rl_mean": X[["IURL", "EURL"]].mean(axis=1),
            "quarter_rr_mean": X[["IURR", "EURR"]].mean(axis=1),
        },
        index=X.index,
    )

    derivadas = pd.DataFrame(index=X.index)
    derivadas["sensor_mean"] = sensores.mean(axis=1)
    derivadas["sensor_std"] = sensores.std(axis=1)
    derivadas["sensor_min"] = sensores.min(axis=1)
    derivadas["sensor_max"] = sensores.max(axis=1)
    derivadas["sensor_range"] = derivadas["sensor_max"] - derivadas["sensor_min"]
    derivadas["front_mean"] = X[["IUFL", "EUFL", "IUFR", "EUFR"]].mean(axis=1)
    derivadas["rear_mean"] = X[["IURL", "EURL", "IURR", "EURR"]].mean(axis=1)
    derivadas["left_mean"] = X[["IUFL", "EUFL", "IURL", "EURL"]].mean(axis=1)
    derivadas["right_mean"] = X[["IUFR", "EUFR", "IURR", "EURR"]].mean(axis=1)
    derivadas["internal_mean"] = X[["IUFL", "IUFR", "IURL", "IURR"]].mean(axis=1)
    derivadas["external_mean"] = X[["EUFL", "EUFR", "EURL", "EURR"]].mean(axis=1)
    derivadas["front_rear_abs_diff"] = (derivadas["front_mean"] - derivadas["rear_mean"]).abs()
    derivadas["left_right_abs_diff"] = (derivadas["left_mean"] - derivadas["right_mean"]).abs()
    derivadas["internal_external_abs_diff"] = (
        derivadas["internal_mean"] - derivadas["external_mean"]
    ).abs()
    derivadas["temperature_sensor_mean_diff"] = X["Temperature"] - derivadas["sensor_mean"]
    derivadas["months_temperature_product"] = X["Months_after_giving_birth"] * X["Temperature"]
    derivadas = pd.concat([derivadas, quartos], axis=1)
    derivadas["quarter_mean_range"] = quartos.max(axis=1) - quartos.min(axis=1)
    derivadas["front_quarter_abs_diff"] = (
        quartos["quarter_fl_mean"] - quartos["quarter_fr_mean"]
    ).abs()
    derivadas["rear_quarter_abs_diff"] = (
        quartos["quarter_rl_mean"] - quartos["quarter_rr_mean"]
    ).abs()

    return pd.concat([X, derivadas], axis=1)


def criar_random_forest(**kwargs) -> RandomForestClassifier:
    parametros = {
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    parametros.update(kwargs)
    return RandomForestClassifier(**parametros)


def criar_extra_trees(**kwargs) -> ExtraTreesClassifier:
    parametros = {
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
    }
    parametros.update(kwargs)
    return ExtraTreesClassifier(**parametros)


def criar_balanced_random_forest(**kwargs) -> BalancedRandomForestClassifier:
    parametros = {
        "random_state": RANDOM_STATE,
        "n_jobs": 1,
        "replacement": True,
    }
    parametros.update(kwargs)
    return BalancedRandomForestClassifier(**parametros)


def criar_pipeline(nome_modelo: str, **kwargs):
    if nome_modelo == "random_forest":
        return Pipeline(
            steps=[
                ("oversampler", RandomOverSampler(random_state=42)),
                ("random_forest", criar_random_forest(**kwargs)),
            ]
        )

    if nome_modelo == "extra_trees":
        return Pipeline(
            steps=[
                ("oversampler", RandomOverSampler(random_state=42)),
                ("extra_trees", criar_extra_trees(**kwargs)),
            ]
        )

    if nome_modelo == "balanced_random_forest":
        return criar_balanced_random_forest(**kwargs)

    raise ValueError(f"Modelo não suportado: {nome_modelo}")


def obter_scorers() -> dict:
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_mastite": make_scorer(precision_score, pos_label=0, zero_division=0),
        "recall_mastite": make_scorer(recall_score, pos_label=0),
        "specificity_saudavel": make_scorer(recall_score, pos_label=1),
    }


def montar_grid(nome_modelo: str, search_mode: str) -> list[dict]:
    if nome_modelo == "random_forest":
        prefixo = "random_forest__"
        if search_mode == "quick":
            return [
                {
                    f"{prefixo}criterion": ["gini", "entropy"],
                    f"{prefixo}n_estimators": [200, 400],
                    f"{prefixo}max_features": ["sqrt", "log2"],
                    f"{prefixo}max_depth": [None, 24],
                    f"{prefixo}min_samples_leaf": [1, 2],
                }
            ]
        return [
            {
                f"{prefixo}criterion": ["gini", "entropy"],
                f"{prefixo}n_estimators": [200, 400, 600],
                f"{prefixo}max_features": ["sqrt", "log2", None],
                f"{prefixo}max_depth": [None, 12, 24],
                f"{prefixo}min_samples_split": [2, 5],
                f"{prefixo}min_samples_leaf": [1, 2],
            }
        ]

    if nome_modelo == "extra_trees":
        prefixo = "extra_trees__"
        if search_mode == "quick":
            return [
                {
                    f"{prefixo}criterion": ["gini", "entropy"],
                    f"{prefixo}n_estimators": [300, 500],
                    f"{prefixo}max_features": ["sqrt", "log2"],
                    f"{prefixo}max_depth": [None, 24],
                    f"{prefixo}min_samples_leaf": [1, 2],
                }
            ]
        return [
            {
                f"{prefixo}criterion": ["gini", "entropy"],
                f"{prefixo}n_estimators": [300, 500, 700],
                f"{prefixo}max_features": ["sqrt", "log2", None],
                f"{prefixo}max_depth": [None, 12, 24],
                f"{prefixo}min_samples_split": [2, 5],
                f"{prefixo}min_samples_leaf": [1, 2],
            }
        ]

    if nome_modelo == "balanced_random_forest":
        if search_mode == "quick":
            return [
                {
                    "criterion": ["gini", "entropy"],
                    "n_estimators": [300, 500],
                    "max_features": ["sqrt", "log2"],
                    "max_depth": [None, 24],
                    "min_samples_leaf": [1, 2],
                }
            ]
        return [
            {
                "criterion": ["gini", "entropy"],
                "n_estimators": [300, 500, 700],
                "max_features": ["sqrt", "log2", None],
                "max_depth": [None, 12, 24],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
        ]

    raise ValueError(f"Modelo não suportado: {nome_modelo}")


def criar_cv_busca(grupos_treino):
    if grupos_treino is not None:
        n_grupos = grupos_treino.nunique()
        if n_grupos < 5:
            raise ValueError("A validação por grupo exige pelo menos 5 animais distintos no treino.")
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    return StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def buscar_melhor_modelo(
    nome_modelo: str,
    X_treino,
    y_treino,
    grupos_treino,
    search_mode: str,
    refit_metric: str,
    n_jobs: int,
) -> GridSearchCV:
    scorers = obter_scorers()
    cv_busca = criar_cv_busca(grupos_treino)
    busca = GridSearchCV(
        estimator=criar_pipeline(nome_modelo),
        param_grid=montar_grid(nome_modelo, search_mode),
        scoring=scorers,
        refit=refit_metric,
        cv=cv_busca,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=False,
    )
    busca.fit(X_treino, y_treino, groups=grupos_treino)
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

    print("\n=== AVALIAÇÃO NO HOLDOUT ===")
    print(f"Acurácia: {metricas['accuracy']:.4f}")
    print(f"Sensibilidade mastite: {metricas['sensitivity_mastite']:.4f}")
    print(f"Especificidade saudáveis: {metricas['specificity_saudavel']:.4f}")
    print("Matriz de confusão (linhas = verdadeiro, colunas = previsto):")
    print(metricas["confusion_matrix"])
    print("\nRelatório de classificação (sklearn):")
    print(classification_report(y_teste, y_pred, digits=4))

    return metricas


def avaliar_limiares(modelo, X_teste, y_teste) -> list[dict]:
    probabilidades = modelo.predict_proba(X_teste)
    classes = list(modelo.classes_)
    if 0 not in classes:
        return []

    prob_mastite = probabilidades[:, classes.index(0)]
    y_real_mastite = y_teste.to_numpy() == 0
    resultados = []

    for limiar in LIMIARES_TRIAGEM:
        revisar = prob_mastite >= limiar
        tp = int((revisar & y_real_mastite).sum())
        fn = int((~revisar & y_real_mastite).sum())
        fp = int((revisar & ~y_real_mastite).sum())
        tn = int((~revisar & ~y_real_mastite).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        resultados.append(
            {
                "threshold": limiar,
                "tp_mastite": tp,
                "fn_mastite": fn,
                "fp_mastite": fp,
                "tn_saudavel": tn,
                "recall_mastite": float(recall),
                "specificity_saudavel": float(specificity),
                "precision_mastite": float(precision),
                "revisar": int(tp + fp),
            }
        )

    print("\n=== LIMIARES DE TRIAGEM NO HOLDOUT ===")
    for item in resultados:
        print(
            f"limiar={item['threshold']:.2f} "
            f"fn_mastite={item['fn_mastite']} "
            f"fp_mastite={item['fp_mastite']} "
            f"recall={item['recall_mastite']:.4f} "
            f"especificidade={item['specificity_saudavel']:.4f}"
        )

    return resultados


def avaliar_validacao_cruzada(modelo, X, y, grupos, n_jobs: int) -> dict:
    if grupos is not None:
        n_splits = min(10, grupos.nunique())
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        descricao_cv = f"VALIDAÇÃO CRUZADA POR ANIMAL ({n_splits} FOLDS)"
    else:
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        descricao_cv = "VALIDAÇÃO CRUZADA ESTRATIFICADA (10 FOLDS)"

    resultados = cross_validate(
        estimator=clone(modelo),
        X=X,
        y=y,
        groups=grupos,
        cv=cv,
        scoring=obter_scorers(),
        n_jobs=n_jobs,
    )

    metricas = {}
    print(f"\n=== {descricao_cv} ===")
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


def dividir_treino_teste(X, y, grupos):
    if grupos is None:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        return X_treino, X_teste, y_treino, y_teste, None, None, "estratificado por linha"

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    idx_treino, idx_teste = next(splitter.split(X, y, groups=grupos))
    return (
        X.iloc[idx_treino],
        X.iloc[idx_teste],
        y.iloc[idx_treino],
        y.iloc[idx_teste],
        grupos.iloc[idx_treino],
        grupos.iloc[idx_teste],
        "por animal (Cow_ID)",
    )


def limpar_parametros(parametros: dict) -> dict:
    limpos = {}
    for chave, valor in parametros.items():
        if "__" in chave:
            chave = chave.split("__", maxsplit=1)[1]
        limpos[chave] = valor
    return limpos


def resumir_busca(nome_modelo: str, busca: GridSearchCV, refit_metric: str) -> dict:
    indice = int(busca.best_index_)
    resumo = {
        "model": nome_modelo,
        "best_score": float(busca.best_score_),
        "best_params": limpar_parametros(busca.best_params_),
    }
    for metrica in REFIT_METRICS:
        chave = f"mean_test_{metrica}"
        if chave in busca.cv_results_:
            resumo[f"mean_{metrica}"] = float(busca.cv_results_[chave][indice])
    resumo["selection_metric"] = refit_metric
    return resumo


def selecionar_melhor_busca(buscas: list[tuple[str, GridSearchCV]]) -> tuple[str, GridSearchCV]:
    return max(buscas, key=lambda item: item[1].best_score_)


def treinar_modelo_final(modelo_base, X, y):
    modelo = clone(modelo_base)
    modelo.fit(X, y)
    return modelo


def obter_estimador_base(modelo):
    if isinstance(modelo, Pipeline):
        return modelo.steps[-1][1]
    return modelo


def obter_importancias(modelo, feature_names: list[str]) -> list[dict]:
    estimador = obter_estimador_base(modelo)
    importancias = getattr(estimador, "feature_importances_", None)
    if importancias is None:
        return []

    pares = sorted(zip(feature_names, importancias), key=lambda item: item[1], reverse=True)
    return [
        {
            "feature": feature,
            "importance": float(importancia),
        }
        for feature, importancia in pares[:25]
    ]


def salvar_modelo(modelo, caminho: Path):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, caminho, compress=("gzip", 3))
    print(f"\nModelo salvo (gzip) em: {caminho}")


def salvar_relatorio(relatorio: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(relatorio, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Relatório salvo em: {caminho}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina, compara e avalia modelos para classificação de mastite."
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SETS,
        default="engineered",
        help="Conjunto de features. 'engineered' adiciona métricas derivadas dos sensores.",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default="quick",
        help="Tamanho da busca de hiperparâmetros.",
    )
    parser.add_argument(
        "--refit-metric",
        choices=REFIT_METRICS,
        default="recall_mastite",
        help="Métrica usada para escolher o melhor modelo. O padrão prioriza não deixar mastite passar.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELOS + ("all",),
        default=["all"],
        help="Modelos a comparar. Use 'all' para testar todos.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Número de processos paralelos na busca e na validação cruzada.",
    )
    parser.add_argument(
        "--allow-row-split",
        action="store_true",
        help=(
            "Permite avaliação estratificada por linha quando Cow_ID não estiver disponível. "
            "Use apenas para bases sem identificador de animal."
        ),
    )
    return parser.parse_args()


def normalizar_modelos(modelos: list[str]) -> list[str]:
    if "all" in modelos:
        return list(MODELOS)
    return modelos


def main():
    args = parse_args()
    modelos = normalizar_modelos(args.models)
    X, y, grupos = carregar_dados(CAMINHO_BASE_TRATADA, args.feature_set)
    if grupos is None and not args.allow_row_split:
        raise ValueError(
            "A base tratada não contém Cow_ID. Rode src/dados/preparar_base.py novamente "
            "para preservar o identificador do animal, ou use --allow-row-split se a base "
            "realmente não tiver grupos."
        )

    print("=== CONFIGURAÇÃO DO TREINO ===")
    print(f"Feature set: {args.feature_set}")
    print(f"Modo de busca: {args.search_mode}")
    print(f"Métrica de seleção: {args.refit_metric}")
    print(f"Modelos avaliados: {', '.join(modelos)}")
    print(f"Jobs paralelos: {args.jobs}")
    print(f"Total de features: {X.shape[1]}")
    agrupamento = f"{COLUNA_GRUPO} ({grupos.nunique()} animais)" if grupos is not None else "indisponível"
    print(f"Agrupamento de avaliação: {agrupamento}")

    (
        X_treino,
        X_teste,
        y_treino,
        y_teste,
        grupos_treino,
        grupos_teste,
        tipo_split,
    ) = dividir_treino_teste(X, y, grupos)
    print(
        f"Split holdout: {tipo_split} | "
        f"Treino: {X_treino.shape[0]} amostras | Teste holdout: {X_teste.shape[0]} amostras"
    )
    if grupos_treino is not None and grupos_teste is not None:
        print(
            f"Animais no treino: {grupos_treino.nunique()} | "
            f"Animais no teste: {grupos_teste.nunique()}"
        )

    buscas = []
    resumos_modelos = []
    for nome_modelo in modelos:
        print(f"\n=== BUSCA: {nome_modelo} ===")
        busca = buscar_melhor_modelo(
            nome_modelo,
            X_treino,
            y_treino,
            grupos_treino,
            search_mode=args.search_mode,
            refit_metric=args.refit_metric,
            n_jobs=args.jobs,
        )
        buscas.append((nome_modelo, busca))
        resumo = resumir_busca(nome_modelo, busca, args.refit_metric)
        resumos_modelos.append(resumo)
        print(f"Melhor score ({args.refit_metric}) para {nome_modelo}: {busca.best_score_:.4f}")
        print(f"Melhores parametros: {resumo['best_params']}")

    melhor_nome, melhor_busca = selecionar_melhor_busca(buscas)
    melhor_estimador_holdout = melhor_busca.best_estimator_
    melhores_parametros = limpar_parametros(melhor_busca.best_params_)
    print(f"\n=== MELHOR MODELO: {melhor_nome} ===")
    print(f"Melhor score ({args.refit_metric}) na busca: {melhor_busca.best_score_:.4f}")
    print(melhores_parametros)

    metricas_holdout = avaliar_holdout(melhor_estimador_holdout, X_teste, y_teste)
    limiares_holdout = avaliar_limiares(melhor_estimador_holdout, X_teste, y_teste)
    metricas_cv = avaliar_validacao_cruzada(melhor_estimador_holdout, X, y, grupos, n_jobs=args.jobs)

    modelo_final = treinar_modelo_final(melhor_estimador_holdout, X, y)
    salvar_modelo(modelo_final, CAMINHO_MODELO)

    relatorio = {
        "feature_set": args.feature_set,
        "search_mode": args.search_mode,
        "refit_metric": args.refit_metric,
        "jobs": args.jobs,
        "features": list(X.columns),
        "feature_count": int(X.shape[1]),
        "models_evaluated": modelos,
        "model_comparison": sorted(
            resumos_modelos,
            key=lambda item: item["best_score"],
            reverse=True,
        ),
        "best_model": melhor_nome,
        "best_params": melhores_parametros,
        "grid_search_best_score": float(melhor_busca.best_score_),
        "group_column": COLUNA_GRUPO if grupos is not None else None,
        "evaluation_split": tipo_split,
        "holdout_metrics": metricas_holdout,
        "threshold_analysis": limiares_holdout,
        "cross_validation_metrics": metricas_cv,
        "top_feature_importances": obter_importancias(modelo_final, list(X.columns)),
    }
    salvar_relatorio(relatorio, CAMINHO_RELATORIO)


if __name__ == "__main__":
    main()
