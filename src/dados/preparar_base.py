from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Constantes de caminhos
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
CAMINHO_BASE_BRUTA = RAIZ_PROJETO / "dados" / "bruto" / "mastite_iot_bruto.csv"
CAMINHO_BASE_TRATADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_tratado.csv"
CAMINHO_BASE_BALANCEADA = RAIZ_PROJETO / "dados" / "processado" / "mastite_iot_balanceado.csv"


def carregar_base_bruta() -> pd.DataFrame:
    """Lê a base bruta e exibe o shape."""
    # A base fornecida está separada por ponto e vírgula (;). Ajuste se necessário.
    df_bruto = pd.read_csv(CAMINHO_BASE_BRUTA, sep=";")
    print(f"Base bruta carregada com shape: {df_bruto.shape}")
    return df_bruto


def limpar_e_padronizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Seleciona colunas relevantes, trata tipos e padroniza o rótulo.
    """
    # No CSV fornecido, o rótulo vem como "class1". Ajuste aqui se mudar.
    NOME_ROTULO_BRUTO = "class1"

    # Caso seja necessário renomear colunas do bruto para os nomes padrão das features
    mapa_renomear = {
        "Months after giving birth": "Months_after_giving_birth",
        # Inclua mapeamentos extras se os nomes do CSV mudarem.
        # "Mastitis": "classe",
    }

    if mapa_renomear:
        df = df.rename(columns=mapa_renomear)

    if NOME_ROTULO_BRUTO not in df.columns:
        raise ValueError(
            f"Coluna de rótulo '{NOME_ROTULO_BRUTO}' não encontrada. "
            "Ajuste NOME_ROTULO_BRUTO para o nome correto do rótulo no CSV bruto."
        )

    colunas_features = [
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

    faltantes = [col for col in colunas_features if col not in df.columns]
    if faltantes:
        raise ValueError(
            f"As seguintes colunas de features estão faltando na base bruta: {faltantes}"
        )

    df = df[colunas_features + [NOME_ROTULO_BRUTO]].copy()

    for coluna in colunas_features:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.dropna(subset=colunas_features + [NOME_ROTULO_BRUTO])

    if NOME_ROTULO_BRUTO != "classe":
        df = df.rename(columns={NOME_ROTULO_BRUTO: "classe"})

    # -------------------------------------------------------------------------
    # ATENCAO: verifique se os rotulos estao corretos antes de treinar.
    #
    # No dataset Mendeley (usado no artigo MasPA), class1=1 significa
    # MASTITE e class1=0 significa SAUDAVEL — o oposto do que este codigo
    # espera (0=mastite, 1=saudavel).
    #
    # Se o seu CSV bruto usar essa convencao, descomente a linha abaixo:
    df["classe"] = 1 - df["classe"]   # inverte: 1->0(mastite), 0->1(saudavel)
    # -------------------------------------------------------------------------
    df["classe"] = df["classe"].astype(int)

    # Validacao automatica: mastite deve ter temperatura media MAIOR
    temp_cls0 = df[df["classe"] == 0]["Temperature"].mean()
    temp_cls1 = df[df["classe"] == 1]["Temperature"].mean()
    print(f"Temp media classe 0: {temp_cls0:.2f} | Temp media classe 1: {temp_cls1:.2f}")
    if temp_cls0 < temp_cls1:
        print("AVISO: classe 0 tem temperatura MENOR que classe 1.")
        print("       Se classe 0 = mastite, isso e clinicamente INVERTIDO.")
        print("       Considere descomentar a linha de inversao acima.")
    else:
        print("OK: classe 0 tem temperatura maior (padrao clinico correto para mastite).")
    print("Distribuição de classes (após limpeza):")
    print(df["classe"].value_counts())

    return df


def balancear_base(df: pd.DataFrame) -> pd.DataFrame:
    """Balanceia a base com RandomOverSampler."""
    X = df.drop(columns=["classe"])
    y = df["classe"]

    sampler = RandomOverSampler(random_state=42)
    X_res, y_res = sampler.fit_resample(X, y)

    df_balanceado = pd.DataFrame(X_res, columns=X.columns)
    df_balanceado["classe"] = y_res

    print("Distribuição de classes após balanceamento:")
    print(df_balanceado["classe"].value_counts())

    return df_balanceado


def main() -> None:
    df_bruto = carregar_base_bruta()
    df_tratado = limpar_e_padronizar(df_bruto)

    CAMINHO_BASE_TRATADA.parent.mkdir(parents=True, exist_ok=True)
    df_tratado.to_csv(CAMINHO_BASE_TRATADA, index=False)
    print(f"Base tratada salva em: {CAMINHO_BASE_TRATADA}")

    df_balanceado = balancear_base(df_tratado)
    df_balanceado.to_csv(CAMINHO_BASE_BALANCEADA, index=False)
    print(f"Base balanceada salva em: {CAMINHO_BASE_BALANCEADA}")


if __name__ == "__main__":
    main()
