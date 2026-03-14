import csv
import io
import logging
import math
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from xml.sax.saxutils import escape as xml_escape

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

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

    @field_validator("ID")
    @classmethod
    def validar_id(cls, value: str) -> str:
        valor = value.strip()
        if not valor:
            raise ValueError("Informe um ID válido.")
        return valor

    @field_validator(
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
    )
    @classmethod
    def validar_campos_numericos(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Informe um número válido.")
        if value < 0:
            raise ValueError("O valor deve ser maior ou igual a zero.")
        return value


class LoteManual(BaseModel):
    registros: list[RegistroManual]


class ResultadoExportacao(BaseModel):
    id: str
    classe_prevista: str
    prob_mastite: float
    prob_saudavel: float | None = None


class ExportacaoPayload(BaseModel):
    formato: Literal["csv", "xlsx"]
    delimitador: str | None = None
    resultados: list[ResultadoExportacao]

    @field_validator("delimitador")
    @classmethod
    def validar_delimitador(cls, value: str | None) -> str | None:
        if value is None:
            return value

        delimitadores_validos = {",", ";", "\t", "|"}
        if value not in delimitadores_validos:
            raise ValueError("Escolha um delimitador CSV válido.")
        return value


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


@app.post("/export/results", summary="Exportar resultados da classificação")
async def exportar_resultados(payload: ExportacaoPayload):
    if not payload.resultados:
        raise HTTPException(status_code=422, detail="Não há resultados para exportar.")

    delimitador = payload.delimitador or ","
    linhas = _montar_linhas_exportacao(payload.resultados)

    if payload.formato == "csv":
        conteudo = _gerar_csv_exportacao(linhas, delimitador)
        extensao = "csv"
        media_type = "text/csv; charset=utf-8"
    else:
        conteudo = _gerar_xlsx_exportacao(linhas)
        extensao = "xlsx"
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    headers = {
        "Content-Disposition": f'attachment; filename="resultado_classificacao.{extensao}"'
    }
    return Response(content=conteudo, media_type=media_type, headers=headers)


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

    cols_negativas = df_sel.columns[(df_sel < 0).any()].tolist()
    if cols_negativas:
        raise ValueError(
            f"Valores negativos não são permitidos nas colunas: {', '.join(cols_negativas)}."
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


def _montar_linhas_exportacao(resultados: list[ResultadoExportacao]) -> list[dict]:
    linhas = []
    for item in resultados:
        linhas.append(
            {
                "ID do animal": item.id,
                "Classe prevista": item.classe_prevista,
                "Prob. Mastite": item.prob_mastite,
                "Prob. Saudável": item.prob_saudavel,
            }
        )
    return linhas


def _gerar_csv_exportacao(linhas: list[dict], delimitador: str) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(linhas[0].keys()),
        delimiter=delimitador,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(linhas)
    return buffer.getvalue().encode("utf-8-sig")


def _gerar_xlsx_exportacao(linhas: list[dict]) -> bytes:
    workbook_buffer = io.BytesIO()

    with zipfile.ZipFile(workbook_buffer, "w", compression=zipfile.ZIP_DEFLATED) as arquivo_zip:
        arquivo_zip.writestr("[Content_Types].xml", _xlsx_content_types())
        arquivo_zip.writestr("_rels/.rels", _xlsx_root_rels())
        arquivo_zip.writestr("xl/workbook.xml", _xlsx_workbook())
        arquivo_zip.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_rels())
        arquivo_zip.writestr("xl/worksheets/sheet1.xml", _xlsx_sheet(linhas))

    return workbook_buffer.getvalue()


def _xlsx_content_types() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""


def _xlsx_root_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""


def _xlsx_workbook() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Resultados" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def _xlsx_workbook_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""


def _xlsx_sheet(linhas: list[dict]) -> str:
    cabecalhos = list(linhas[0].keys())
    todas_linhas = [cabecalhos] + [list(linha.values()) for linha in linhas]

    xml_linhas = []
    for indice_linha, valores in enumerate(todas_linhas, start=1):
        celulas = []
        for indice_coluna, valor in enumerate(valores, start=1):
            referencia = f"{_numero_para_coluna_excel(indice_coluna)}{indice_linha}"
            if isinstance(valor, (int, float)) and not isinstance(valor, bool) and value_is_finite(valor):
                celulas.append(f'<c r="{referencia}"><v>{valor}</v></c>')
                continue

            texto = "" if valor is None else xml_escape(str(valor))
            celulas.append(
                f'<c r="{referencia}" t="inlineStr"><is><t>{texto}</t></is></c>'
            )

        xml_linhas.append(f'<row r="{indice_linha}">{"".join(celulas)}</row>')

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {''.join(xml_linhas)}
  </sheetData>
</worksheet>"""


def _numero_para_coluna_excel(numero: int) -> str:
    letras = []
    while numero > 0:
        numero, resto = divmod(numero - 1, 26)
        letras.append(chr(65 + resto))
    return "".join(reversed(letras))


def value_is_finite(valor: float) -> bool:
    return math.isfinite(float(valor))


# ── Ponto de entrada (para rodar localmente) ─────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
