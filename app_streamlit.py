import io
import logging
from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st

RAIZ_PROJETO = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl.gz"
CAMINHO_MODELO_LEGACY = RAIZ_PROJETO / "modelos" / "random_forest_mastite.pkl"
CAMINHO_EXEMPLO = RAIZ_PROJETO / "exemplo_entrada.csv"
CAMINHO_LOGO = RAIZ_PROJETO / "images" / "ulbra-logo.png"

COLUNAS_ESPERADAS = [
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
MAX_UPLOAD_MB = 5


def _inject_custom_css():
    """Injeta CSS customizado para a landing page."""
    st.markdown(
        """
        <style>
        /* ========== VARIAVEIS DE COR ========== */
        :root {
            --primary: #FF4B4B;
            --primary-dark: #E63946;
            --primary-light: #FFE5E5;
            --secondary-bg: #F0F2F6;
            --text: #262730;
            --text-muted: #6C757D;
            --white: #FFFFFF;
            --border: #E0E0E0;
            --shadow: rgba(0, 0, 0, 0.08);
        }

        /* ========== RESET E BASE ========== */
        .main .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            max-width: 1000px !important;
        }

        /* ========== HERO SECTION ========== */
        .hero-section {
            background: var(--primary);
            padding: 3.5rem 2rem;
            text-align: center;
            color: var(--white);
            border-radius: 12px;
            margin-bottom: 3rem;
            box-shadow: 0 4px 20px var(--shadow);
        }

        .hero-section h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            line-height: 1.2;
            color: var(--white);
        }

        .hero-section p {
            font-size: 1.15rem;
            opacity: 0.95;
            max-width: 700px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ========== SECOES ========== */
        .section {
            margin-bottom: 3.5rem;
        }

        .section-header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .section-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        .section-subtitle {
            color: var(--text-muted);
            font-size: 1.05rem;
        }

        /* ========== CARDS DE FUNCIONALIDADES ========== */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .feature-card {
            background: var(--white);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            box-shadow: 0 8px 24px var(--shadow);
            transform: translateY(-4px);
            border-color: var(--primary-light);
        }

        .feature-icon {
            width: 64px;
            height: 64px;
            background: var(--primary-light);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.25rem auto;
        }

        .feature-icon svg {
            width: 32px;
            height: 32px;
            color: var(--primary);
        }

        .feature-card h3 {
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .feature-card p {
            color: var(--text-muted);
            font-size: 0.95rem;
            line-height: 1.6;
            margin: 0;
        }

        /* ========== COMO FUNCIONA ========== */
        .steps-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 2rem;
        }

        .step {
            text-align: center;
        }

        .step-number {
            width: 64px;
            height: 64px;
            background: var(--primary);
            color: var(--white);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.5rem;
            margin: 0 auto 1rem auto;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }

        .step h4 {
            color: var(--text);
            margin: 0 0 0.5rem 0;
            font-size: 1.05rem;
            font-weight: 600;
        }

        .step p {
            color: var(--text-muted);
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        /* ========== INFO BOX ========== */
        .info-box {
            background: var(--secondary-bg);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .info-box h4 {
            color: var(--text);
            margin: 0 0 0.75rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .info-box p {
            color: var(--text);
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.7;
        }

        /* ========== AVISO ========== */
        .warning-box {
            background: #FFF3CD;
            border-left: 4px solid #FFC107;
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 2rem;
        }

        .warning-box p {
            color: #856404;
            margin: 0;
            font-size: 0.9rem;
            line-height: 1.6;
        }

        /* ========== FOOTER ========== */
        .site-footer {
            background: var(--secondary-bg);
            color: var(--text);
            padding: 2.5rem 2rem;
            border-radius: 12px;
            margin-top: 4rem;
        }

        .footer-content {
            max-width: 900px;
            margin: 0 auto;
        }

        .footer-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-bottom: 2rem;
        }

        .footer-section h4 {
            color: var(--text);
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .footer-section p {
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.6;
            margin: 0;
        }

        .footer-section ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .footer-section li {
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .footer-divider {
            border: none;
            border-top: 1px solid var(--border);
            margin: 0 0 1.5rem 0;
        }

        .footer-bottom {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }

        .footer-logo {
            height: 40px;
            width: auto;
        }

        .footer-bottom p {
            color: var(--text-muted);
            font-size: 0.85rem;
            margin: 0;
        }

        .footer-bottom .highlight {
            color: var(--text);
            font-weight: 600;
        }

        /* ========== REMOVER CURSOR DE LINK ========== */
        .hero-section h1,
        .hero-section p,
        .section-title,
        .section-subtitle,
        .feature-card h3,
        .feature-card p,
        .step h4,
        .step p,
        .info-box h4,
        .info-box p,
        .footer-section h4,
        .footer-section p,
        .footer-section li,
        .footer-bottom p {
            cursor: default !important;
            pointer-events: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_landing_page():
    """Renderiza a landing page com design HTML/CSS profissional."""
    _inject_custom_css()

    # Carregar logo em base64
    logo_base64 = ""
    if CAMINHO_LOGO.exists():
        with open(CAMINHO_LOGO, "rb") as img_file:
            logo_base64 = base64.b64encode(img_file.read()).decode()

    # Hero Section
    st.markdown(
        """
        <div class="hero-section">
            <h1>Classificador de Mastite Bovina</h1>
            <p>
                Sistema inteligente para detecção precoce de mastite em vacas leiteiras
                utilizando dados de sensores IoT e algoritmos de Machine Learning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sobre o Projeto
    st.markdown(
        """
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Sobre o Projeto</h2>
                <p class="section-subtitle">Tecnologia aplicada à pecuária leiteira</p>
            </div>
            <div class="info-box">
                <h4>O que é Mastite?</h4>
                <p>A mastite é uma inflamação da glândula mamária, sendo uma das doenças mais frequentes e onerosas na bovinocultura leiteira.
                Causa redução na produção de leite, descarte de leite contaminado, custos elevados com tratamento veterinário e,
                em casos severos, pode levar ao descarte do animal. A detecção precoce é fundamental para minimizar perdas econômicas
                e garantir o bem-estar animal.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Funcionalidades
    st.markdown(
        """
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Funcionalidades</h2>
                <p class="section-subtitle">Recursos disponíveis no sistema</p>
            </div>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                        </svg>
                    </div>
                    <h3>Coleta de Dados IoT</h3>
                    <p>Processa dados de 8 sensores de corrente elétrica posicionados no úbere, além da temperatura corporal.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                        </svg>
                    </div>
                    <h3>Modelo Preditivo</h3>
                    <p>Algoritmo Random Forest treinado com dados balanceados para classificação precisa.</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                    </div>
                    <h3>Análise de Risco</h3>
                    <p>Fornece probabilidades detalhadas para cada diagnóstico, auxiliando na tomada de decisão.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Como Funciona
    st.markdown(
        """
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Como Funciona</h2>
                <p class="section-subtitle">Processo de classificação em 4 etapas</p>
            </div>
            <div class="steps-container">
                <div class="step">
                    <div class="step-number">1</div>
                    <h4>Coleta</h4>
                    <p>Obtenha as leituras dos sensores IoT</p>
                </div>
                <div class="step">
                    <div class="step-number">2</div>
                    <h4>Upload</h4>
                    <p>Envie o arquivo CSV ou preencha manualmente</p>
                </div>
                <div class="step">
                    <div class="step-number">3</div>
                    <h4>Processamento</h4>
                    <p>O modelo analisa os dados enviados</p>
                </div>
                <div class="step">
                    <div class="step-number">4</div>
                    <h4>Resultado</h4>
                    <p>Receba o diagnóstico e probabilidades</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Aviso
    st.markdown(
        """
        <div class="warning-box">
            <p><strong>Aviso importante:</strong> Este sistema é uma ferramenta de apoio à decisão desenvolvida para fins acadêmicos.
            Os resultados não substituem a avaliação clínica de um médico veterinário qualificado.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Especificações
    with st.expander("Ver especificação das variáveis de entrada"):
        st.dataframe(_schema_dataframe(), use_container_width=True)

    # Botões de ação
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Iniciar Classificação", type="primary", use_container_width=True):
            st.session_state.started = True
            st.rerun()

        if CAMINHO_EXEMPLO.exists():
            with CAMINHO_EXEMPLO.open("rb") as f:
                st.download_button(
                    "Baixar arquivo de exemplo",
                    data=f,
                    file_name="exemplo_entrada.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown("---")

    # Footer
    footer_html = f"""
        <div class="site-footer">
            <div class="footer-content">
                <div class="footer-grid">
                    <div class="footer-section">
                        <h4>Sobre o Trabalho</h4>
                        <p>Trabalho de Conclusão de Curso desenvolvido como requisito para obtenção do título de graduação.
                        O projeto aplica técnicas de Machine Learning e Internet das Coisas (IoT) na detecção precoce de mastite bovina.</p>
                    </div>
                    <div class="footer-section">
                        <h4>Tecnologias</h4>
                        <ul>
                            <li>Python 3.x</li>
                            <li>Scikit-learn</li>
                            <li>Streamlit</li>
                            <li>Pandas</li>
                        </ul>
                    </div>
                    <div class="footer-section">
                        <h4>Referências</h4>
                        <ul>
                            <li>Artigo MasPA</li>
                            <li>AgriEngineering, 2021</li>
                            <li>DOI: 10.3390/agriengineering3030037</li>
                        </ul>
                    </div>
                </div>
                <hr class="footer-divider">
                <div class="footer-bottom">
                    {"<img src='data:image/png;base64," + logo_base64 + "' alt='ULBRA' class='footer-logo'>" if logo_base64 else ""}
                    <p><span class="highlight">Classificador de Mastite Bovina</span> - Trabalho de Conclusão de Curso - 2026/1</p>
                </div>
            </div>
        </div>
        """
    
    st.markdown(footer_html, unsafe_allow_html=True)


def _configurar_logger() -> tuple[logging.Logger, io.StringIO]:
    """Cria logger em memória para exibir no app."""
    buffer = io.StringIO()
    logger = logging.getLogger("mastite_app")
    logger.setLevel(logging.INFO)
    # Evitar handlers duplicados em reruns
    if not logger.handlers:
        handler = logging.StreamHandler(buffer)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        handler = logger.handlers[0]
        handler.stream = buffer
    return logger, buffer


@st.cache_resource(show_spinner="Carregando modelo...")
def carregar_modelo(_logger: logging.Logger):
    caminho = CAMINHO_MODELO if CAMINHO_MODELO.exists() else CAMINHO_MODELO_LEGACY
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {CAMINHO_MODELO} nem em {CAMINHO_MODELO_LEGACY}. "
            "Treine e salve o modelo antes do deploy."
        )
    _logger.info("Carregando modelo de %s", caminho)
    modelo = joblib.load(caminho)
    _logger.info("Modelo carregado com sucesso.")
    return modelo


def preparar_dados(df: pd.DataFrame) -> pd.DataFrame:
    colunas_ausentes = [c for c in COLUNAS_ESPERADAS if c not in df.columns]
    if colunas_ausentes:
        raise ValueError(
            f"Colunas ausentes no CSV enviado: {', '.join(colunas_ausentes)}. "
            "Use o layout de exemplo_entrada.csv."
        )

    df_sel = df[COLUNAS_ESPERADAS].copy()

    # Garantir tipos numéricos
    for col in COLUNAS_ESPERADAS:
        df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")
    if df_sel[COLUNAS_ESPERADAS].isna().any().any():
        cols_invalidas = df_sel.columns[df_sel.isna().any()].tolist()
        raise ValueError(
            f"Valores inválidos ou vazios em colunas: {', '.join(cols_invalidas)}. "
            "Revise o CSV."
        )

    if df_sel.empty:
        raise ValueError("Nenhuma linha válida após validação dos dados.")

    return df_sel


def _schema_dataframe() -> pd.DataFrame:
    """Retorna um pequeno dicionário de dados para exibir na UI."""
    return pd.DataFrame(
        [
            ("ID", "string", "Identificador único do animal"),
            ("Months_after_giving_birth", "float", "Meses pós-parto"),
            ("IUFL", "float", "Corrente interna úbere frontal esquerda"),
            ("EUFL", "float", "Corrente externa úbere frontal esquerda"),
            ("IUFR", "float", "Corrente interna úbere frontal direita"),
            ("EUFR", "float", "Corrente externa úbere frontal direita"),
            ("IURL", "float", "Corrente interna úbere traseira esquerda"),
            ("EURL", "float", "Corrente externa úbere traseira esquerda"),
            ("IURR", "float", "Corrente interna úbere traseira direita"),
            ("EURR", "float", "Corrente externa úbere traseira direita"),
            ("Temperature", "float", "Temperatura do animal (°C)"),
        ],
        columns=["Coluna", "Tipo", "Descrição"],
    )


def main() -> None:
    st.set_page_config(
        page_title="Classificador de Mastite Bovina",
        page_icon=None,
        layout="centered",
    )

    if "started" not in st.session_state:
        st.session_state.started = False

    def processar(df_original: pd.DataFrame, logger: logging.Logger):
        st.subheader("Dados recebidos")
        st.dataframe(df_original)

        if "ID" not in df_original.columns:
            st.error("A coluna 'ID' é obrigatória para identificar cada animal.")
            logger.error("Coluna ID ausente.")
            return

        df_sem_id = df_original.drop(columns=["ID"])

        try:
            df_modelo = preparar_dados(df_sem_id)
            logger.info("Dados validados. Linhas: %d", len(df_modelo))
        except ValueError as exc:
            st.error(str(exc))
            logger.error("Falha na validação dos dados: %s", exc)
            return

        try:
            modelo = carregar_modelo(logger)
        except FileNotFoundError as exc:
            st.error(str(exc))
            logger.error("Modelo não encontrado.")
            return
        except Exception as exc:
            st.error(f"Erro ao carregar modelo: {exc}")
            logger.exception("Erro ao carregar modelo: %s", exc)
            return

        try:
            probabilidades = modelo.predict_proba(df_modelo)
            predicoes = modelo.predict(df_modelo)
            logger.info("Inferência concluída.")
        except Exception as exc:
            st.error(f"Erro ao executar a inferência: {exc}")
            logger.exception("Erro na inferência: %s", exc)
            return

        classes_modelo = list(modelo.classes_)
        idx_mastite = classes_modelo.index(0)
        idx_saudavel = classes_modelo.index(1) if 1 in classes_modelo else None

        df_resultado = df_original.copy()
        df_resultado["classe_prevista"] = [
            "Mastite" if p == 0 else "Saudável" for p in predicoes
        ]
        df_resultado["prob_mastite"] = probabilidades[:, idx_mastite]
        if idx_saudavel is not None:
            df_resultado["prob_saudavel"] = probabilidades[:, idx_saudavel]

        st.subheader("Resultados")
        st.dataframe(df_resultado)

        total = len(df_resultado)
        mastite = (df_resultado["classe_prevista"] == "Mastite").sum()
        saudavel = (df_resultado["classe_prevista"] == "Saudável").sum()

        st.markdown(
            f"**Total de animais:** {total} | "
            f"**Com risco de mastite:** {mastite} | "
            f"**Saudáveis:** {saudavel}"
        )

    # ---------- UI principal ----------
    if not st.session_state.started:
        _render_landing_page()
        return

    # Sidebar com botão de voltar e configurações
    with st.sidebar:
        if st.button("Voltar ao início", use_container_width=True):
            st.session_state.started = False
            st.rerun()

        st.divider()
        st.subheader("Entrada de dados")
        st.caption(f"Tamanho máximo: {MAX_UPLOAD_MB} MB")
        arquivo = st.file_uploader(
            "Selecione um arquivo CSV", type=["csv"]
        )

        if CAMINHO_EXEMPLO.exists():
            with CAMINHO_EXEMPLO.open("rb") as f:
                st.download_button(
                    "Baixar arquivo de exemplo",
                    data=f,
                    file_name="exemplo_entrada.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # Título da página de classificação
    st.title("Classificação de Mastite")
    st.caption("Envie seus dados para análise")

    logger, buffer_logs = _configurar_logger()

    aba_upload, aba_manual = st.tabs(["Upload CSV", "Entrada manual"])

    with aba_upload:

        if arquivo is None:
            st.info(
                "Envie um CSV no formato de exemplo_entrada.csv para obter previsões."
            )
        elif arquivo.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(
                f"O arquivo enviado tem {arquivo.size / (1024*1024):.2f} MB. "
                f"O limite é {MAX_UPLOAD_MB} MB."
            )
        else:
            try:
                df_original = pd.read_csv(arquivo)
                logger.info("CSV carregado com shape %s", df_original.shape)
                processar(df_original, logger)
            except Exception as exc:
                st.error(f"Não foi possível ler o CSV enviado: {exc}")
                logger.exception("Erro ao ler CSV: %s", exc)

    with aba_manual:
        st.write("Preencha os dados abaixo ou edite as linhas para vários animais.")
        seed = pd.DataFrame(
            [
                {
                    "ID": "Cow-001",
                    "Months_after_giving_birth": 3.0,
                    "IUFL": 12.4,
                    "EUFL": 11.8,
                    "IUFR": 13.1,
                    "EUFR": 12.9,
                    "IURL": 10.7,
                    "EURL": 10.5,
                    "IURR": 11.3,
                    "EURR": 11.0,
                    "Temperature": 38.6,
                }
            ]
        )

        df_editor = st.data_editor(seed, num_rows="dynamic", key="manual_editor")

        if st.button("Diagnosticar entradas manuais", type="primary"):
            if df_editor.empty:
                st.warning("Adicione pelo menos uma linha para diagnosticar.")
            else:
                df_editor["ID"] = df_editor["ID"].astype(str)
                processar(df_editor, logger)

    with st.sidebar:
        st.divider()
        with st.expander("Logs do sistema"):
            st.code(buffer_logs.getvalue() or "Nenhum log registrado.")


if __name__ == "__main__":
    main()
