import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Caminhos ──────────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ / "modelos" / "random_forest_mastite.pkl.gz"
CAMINHO_MODELO_LEGACY = RAIZ / "modelos" / "random_forest_mastite.pkl"
CAMINHO_EXEMPLO = RAIZ / "exemplo_entrada.csv"
STATIC_DIR  = RAIZ / "static"
IMAGES_DIR  = RAIZ / "images"

COLUNAS_ESPERADAS = [
    "Months_after_giving_birth",
    "IUFL", "EUFL",
    "IUFR", "EUFR",
    "IURL", "EURL",
    "IURR", "EURR",
    "Temperature",
]

CLASSES = {
    0: "Mastite",
    1: "Saudável",
}

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mastite_api")

# ── Modelo (carregado uma vez na inicialização) ───────────────────────────────
def _obter_caminho_modelo() -> Path | None:
    if CAMINHO_MODELO.exists():
        return CAMINHO_MODELO
    if CAMINHO_MODELO_LEGACY.exists():
        return CAMINHO_MODELO_LEGACY
    return None


def _carregar_modelo():
    caminho = _obter_caminho_modelo()
    if caminho is None:
        logger.warning(
            "Modelo não encontrado em %s nem em %s",
            CAMINHO_MODELO,
            CAMINHO_MODELO_LEGACY,
        )
        return None

    logger.info("Carregando modelo de %s", caminho)
    modelo = joblib.load(caminho)
    logger.info("Modelo carregado com sucesso.")
    return modelo


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.modelo = _carregar_modelo()
    yield

# ── App FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Classificador de Mastite Bovina",
    description="API para classificação de risco de mastite via Machine Learning (Aprendizado de Máquina).",
    version="1.0.0",
    lifespan=lifespan,
)

# Servir arquivos estáticos (CSS, JS) e imagens
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── Rotas ─────────────────────────────────────────────────────────────────────
class RegistroManual(BaseModel):
    ID: str
    Months_after_giving_birth: float
    IUFL: float
    EUFL: float
    IUFR: float
    EUFR: float
    IURL: float
    EURL: float
    IURR: float
    EURR: float
    Temperature: float


class LoteManual(BaseModel):
    registros: list[RegistroManual]


@app.get("/", include_in_schema=False)
def raiz():
    """Serve a landing page."""
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html; charset=utf-8")


@app.get("/exemplo", summary="Baixar CSV de exemplo")
def baixar_exemplo():
    """Retorna o arquivo CSV de exemplo para o usuário testar o sistema."""
    if not CAMINHO_EXEMPLO.exists():
        raise HTTPException(status_code=404, detail="Arquivo de exemplo não encontrado.")
    return FileResponse(
        str(CAMINHO_EXEMPLO),
        filename="exemplo_entrada.csv",
        media_type="text/csv",
    )


@app.post("/predict", summary="Classificar animais via CSV")
async def predict(arquivo: UploadFile = File(..., description="Arquivo CSV com os dados dos animais")):
    """
    Recebe um CSV com os dados dos sensores e retorna a classificação de cada animal.

    Colunas obrigatórias: ID, Months_after_giving_birth, IUFL, EUFL, IUFR, EUFR,
    IURL, EURL, IURR, EURR, Temperature.
    """
    modelo = _obter_modelo()
    try:
        df = await _ler_csv_upload(arquivo)
        return JSONResponse(_executar_predicao(modelo, df))
    finally:
        await arquivo.close()


@app.post("/predict/manual", summary="Classificar animais via entrada manual")
async def predict_manual(payload: LoteManual):
    modelo = _obter_modelo()
    if not payload.registros:
        raise HTTPException(status_code=422, detail="Informe pelo menos um animal para classificar.")

    df = pd.DataFrame([registro.model_dump() for registro in payload.registros])
    return JSONResponse(_executar_predicao(modelo, df))


# ── Funções auxiliares ────────────────────────────────────────────────────────

def _preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas_ausentes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes no CSV: {', '.join(colunas_ausentes)}. "
            "Veja o arquivo de exemplo para o formato correto."
        )

    df_sel = df.loc[:, COLUNAS_ESPERADAS].apply(pd.to_numeric, errors="coerce")

    cols_invalidas = df_sel.columns[df_sel.isna().any()].tolist()
    if cols_invalidas:
        raise ValueError(
            f"Valores inválidos ou vazios nas colunas: {', '.join(cols_invalidas)}. Revise o CSV."
        )

    if df_sel.empty:
        raise ValueError("Nenhuma linha válida após a validação dos dados.")

    return df_sel


async def _ler_csv_upload(arquivo: UploadFile) -> pd.DataFrame:
    if not (arquivo.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo com extensão .csv")

    conteudo = await arquivo.read()
    if len(conteudo) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande ({len(conteudo) / 1024 / 1024:.1f} MB). Limite: 5 MB.",
        )

    try:
        df = pd.read_csv(io.BytesIO(conteudo))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Não foi possível ler o CSV: {exc}") from exc

    logger.info("CSV recebido com shape %s", df.shape)
    return df


def _obter_modelo():
    modelo = getattr(app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não disponível. Treine e salve o modelo antes de usar a API.",
        )
    return modelo


def _executar_predicao(modelo, df: pd.DataFrame) -> dict:
    if "ID" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="A coluna 'ID' é obrigatória para identificar cada animal.",
        )

    try:
        df_modelo = _preparar_dados(df)
        probabilidades = modelo.predict_proba(df_modelo)
        predicoes = modelo.predict(df_modelo)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro na inferência: %s", exc)
        raise HTTPException(status_code=500, detail=f"Erro ao executar a classificação: {exc}") from exc

    logger.info("Inferência concluída para %d animais.", len(predicoes))

    indices_classe = {classe: idx for idx, classe in enumerate(modelo.classes_)}
    if 0 not in indices_classe:
        raise HTTPException(status_code=500, detail="O modelo carregado não possui a classe de mastite.")

    resultados = []
    idx_mastite = indices_classe[0]
    idx_saudavel = indices_classe.get(1)

    for animal_id, pred, probs in zip(df["ID"].astype(str), predicoes, probabilidades):
        item = {
            "id": animal_id,
            "classe_prevista": CLASSES.get(int(pred), str(pred)),
            "prob_mastite": round(float(probs[idx_mastite]), 4),
        }
        if idx_saudavel is not None:
            item["prob_saudavel"] = round(float(probs[idx_saudavel]), 4)
        resultados.append(item)

    total = len(resultados)
    n_mastite = sum(1 for r in resultados if r["classe_prevista"] == CLASSES[0])

    return {
        "total": total,
        "mastite": n_mastite,
        "saudavel": total - n_mastite,
        "resultados": resultados,
    }


# ── Ponto de entrada (para rodar localmente) ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
