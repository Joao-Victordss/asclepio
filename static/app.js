/* ═══════════════════════════════════════════════════════
   NAV - scroll shadow + menu mobile
════════════════════════════════════════════════════════ */
const nav       = document.getElementById('nav');
const navBurger = document.getElementById('navBurger');
const navMobile = document.getElementById('navMobile');
const tabCsv     = document.getElementById('tabCsv');
const tabManual  = document.getElementById('tabManual');
const resultsEl  = document.getElementById('results');
const resultsBody = document.getElementById('resultsBody');
const resultsMeta = document.getElementById('resultsMeta');
const pageSizeOptions = document.getElementById('pageSizeOptions');
const resultsPagination = document.getElementById('resultsPagination');
const paginationInfo = document.getElementById('paginationInfo');
const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');

window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 10);
}, { passive: true });

navBurger.addEventListener('click', () => {
  const isOpen = navMobile.classList.toggle('open');
  navBurger.setAttribute('aria-expanded', String(isOpen));
});
navMobile.querySelectorAll('a').forEach(l => l.addEventListener('click', () => {
  navMobile.classList.remove('open');
  navBurger.setAttribute('aria-expanded', 'false');
}));

/* ═══════════════════════════════════════════════════════
   TABS
════════════════════════════════════════════════════════ */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('tab-active'));
    btn.classList.add('tab-active');

    const tab = btn.dataset.tab;
    setHidden(tabCsv, tab !== 'csv');
    setHidden(tabManual, tab !== 'manual');
    hideError();
    hideErrorManual();
    hideResults();
  });
});

/* ═══════════════════════════════════════════════════════
   UTILIDADES
════════════════════════════════════════════════════════ */
function formatBytes(bytes) {
  if (bytes < 1024)        return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function setHidden(element, hidden) {
  element.hidden = hidden;
}

function normalizeErrorMessage(detail) {
  if (Array.isArray(detail)) {
    return detail.map(item => item.msg || 'Erro de validação.').join(' ');
  }

  if (detail && typeof detail === 'object') {
    return detail.msg || 'Erro inesperado ao processar a requisição.';
  }

  return detail || 'Erro inesperado ao processar a requisição.';
}

/** Escapa HTML para prevenir XSS */
function esc(str) {
  const d = document.createElement('div');
  d.appendChild(document.createTextNode(String(str)));
  return d.innerHTML;
}

/* ═══════════════════════════════════════════════════════
   ABA CSV - upload, drag & drop
════════════════════════════════════════════════════════ */
const uploadArea        = document.getElementById('uploadArea');
const fileInput         = document.getElementById('fileInput');
const uploadBtn         = document.getElementById('uploadBtn');
const fileSelected      = document.getElementById('fileSelected');
const fileNameEl        = document.getElementById('fileName');
const fileSizeEl        = document.getElementById('fileSize');
const fileRemove        = document.getElementById('fileRemove');
const analyzeBtn        = document.getElementById('analyzeBtn');
const analyzeBtnText    = document.getElementById('analyzeBtnText');
const analyzeBtnSpinner = document.getElementById('analyzeBtnSpinner');
const resultError       = document.getElementById('resultError');
const errorMsg          = document.getElementById('errorMsg');
const downloadExemplo   = document.getElementById('downloadExemplo');

let selectedFile = null;
let currentPage = 1;
let pageSize = 10;

function showFileSelected(file) {
  selectedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatBytes(file.size);
  setHidden(uploadArea, true);
  setHidden(fileSelected, false);
  hideError();
  hideResults();
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  setHidden(uploadArea, false);
  setHidden(fileSelected, true);
  hideError();
  hideResults();
}

function showError(msg) {
  errorMsg.textContent        = msg;
  setHidden(resultError, false);
}

function hideError() {
  setHidden(resultError, true);
  errorMsg.textContent = '';
}

function hideResults() {
  setHidden(resultsEl, true);
  setHidden(resultsPagination, true);
}

function setCsvLoading(on) {
  analyzeBtn.disabled             = on;
  analyzeBtnText.style.display    = on ? 'none' : '';
  analyzeBtnSpinner.style.display = on ? 'flex' : 'none';
}

// Botão "Selecionar arquivo" abre o input
uploadBtn.addEventListener('click', () => fileInput.click());

// Clique na área de upload; ignora se for o link de exemplo ou o próprio botão
uploadArea.addEventListener('click', (e) => {
  if (e.target.closest('#downloadExemplo') || e.target.closest('#uploadBtn')) return;
  fileInput.click();
});

// Impede que o link de download propague para o handler da área
downloadExemplo.addEventListener('click', (e) => e.stopPropagation());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) showFileSelected(fileInput.files[0]);
});

// Drag & drop
uploadArea.addEventListener('dragover',  (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', ()  => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) { showError('Apenas arquivos .csv são aceitos.'); return; }
  showFileSelected(file);
});

fileRemove.addEventListener('click', resetUpload);

// Analisar CSV
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  if (selectedFile.size > 5 * 1024 * 1024) {
    showError(`Arquivo muito grande (${formatBytes(selectedFile.size)}). Limite: 5 MB.`);
    return;
  }
  hideError();
  setCsvLoading(true);
  const form = new FormData();
  form.append('arquivo', selectedFile);
  await sendAndRender('/predict', { method: 'POST', body: form }, showError);
  setCsvLoading(false);
});

/* ═══════════════════════════════════════════════════════
   ABA MANUAL - tabela dinâmica
════════════════════════════════════════════════════════ */
const CAMPOS = ['Months_after_giving_birth','IUFL','EUFL','IUFR','EUFR','IURL','EURL','IURR','EURR','Temperature'];
const EXEMPLO = { ID:'Vaca-001', Months_after_giving_birth:3.0, IUFL:12.4, EUFL:11.8, IUFR:13.1, EUFR:12.9, IURL:10.7, EURL:10.5, IURR:11.3, EURR:11.0, Temperature:38.6 };

const manualBody        = document.getElementById('manualBody');
const addRowBtn         = document.getElementById('addRowBtn');
const analyzeManualBtn  = document.getElementById('analyzeManualBtn');
const manualBtnText     = document.getElementById('manualBtnText');
const manualBtnSpinner  = document.getElementById('manualBtnSpinner');
const resultErrorManual = document.getElementById('resultErrorManual');
const errorMsgManual    = document.getElementById('errorMsgManual');

let rowCounter = 0;

function createRow(defaults = {}) {
  rowCounter++;
  const tr = document.createElement('tr');
  tr.dataset.rowId = rowCounter;

  // ID
  const tdId = document.createElement('td');
  const inId = document.createElement('input');
  inId.type = 'text';
  inId.value = defaults.ID || `Vaca-${String(rowCounter).padStart(3,'0')}`;
  inId.dataset.field = 'ID';
  tdId.appendChild(inId);
  tr.appendChild(tdId);

  // Campos numéricos
  CAMPOS.forEach(campo => {
    const td = document.createElement('td');
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = 'any';
    inp.value = defaults[campo] !== undefined ? defaults[campo] : '';
    inp.dataset.field = campo;
    td.appendChild(inp);
    tr.appendChild(td);
  });

  // Botão remover
  const tdX = document.createElement('td');
  const btnX = document.createElement('button');
  btnX.type = 'button';
  btnX.className = 'btn-remove-row';
  btnX.title = 'Remover linha';
  btnX.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/></svg>`;
  btnX.addEventListener('click', () => {
    tr.remove();
    // Mantém pelo menos uma linha
    if (manualBody.querySelectorAll('tr').length === 0) addRow(EXEMPLO);
  });
  tdX.appendChild(btnX);
  tr.appendChild(tdX);

  return tr;
}

function addRow(defaults = {}) {
  manualBody.appendChild(createRow(defaults));
}

// Linha inicial com dados de exemplo
addRow(EXEMPLO);

addRowBtn.addEventListener('click', () => addRow());

function getManualRows() {
  return Array.from(manualBody.querySelectorAll('tr')).map(tr => {
    const obj = {};
    tr.querySelectorAll('input').forEach(inp => { obj[inp.dataset.field] = inp.value; });
    return obj;
  });
}

function showErrorManual(msg) {
  errorMsgManual.textContent          = msg;
  setHidden(resultErrorManual, false);
}

function hideErrorManual() {
  setHidden(resultErrorManual, true);
  errorMsgManual.textContent = '';
}

function setManualLoading(on) {
  analyzeManualBtn.disabled           = on;
  manualBtnText.style.display         = on ? 'none' : '';
  manualBtnSpinner.style.display      = on ? 'flex' : 'none';
}

analyzeManualBtn.addEventListener('click', async () => {
  const rows = getManualRows();
  if (rows.length === 0) { showErrorManual('Adicione pelo menos um animal.'); return; }

  hideErrorManual();
  setManualLoading(true);
  await sendAndRender('/predict/manual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ registros: rows }),
  }, showErrorManual);
  setManualLoading(false);
});

/* ═══════════════════════════════════════════════════════
   ENVIAR E RENDERIZAR (compartilhado pelas duas abas)
════════════════════════════════════════════════════════ */
let lastResultData = null;   // armazena para download CSV

function getTotalPages(totalItems) {
  return Math.max(1, Math.ceil(totalItems / pageSize));
}

function renderPagination(totalItems) {
  const totalPages = getTotalPages(totalItems);
  const start = totalItems === 0 ? 0 : ((currentPage - 1) * pageSize) + 1;
  const end = Math.min(currentPage * pageSize, totalItems);

  resultsMeta.textContent = `Mostrando ${start} a ${end} de ${totalItems} animais`;
  paginationInfo.textContent = `Página ${currentPage} de ${totalPages}`;
  prevPageBtn.disabled = currentPage === 1;
  nextPageBtn.disabled = currentPage === totalPages;
  setHidden(resultsPagination, totalItems <= pageSize);
}

function syncPageSizeButtons() {
  pageSizeOptions.querySelectorAll('.page-size-btn').forEach((button) => {
    const isActive = Number(button.dataset.pageSize) === pageSize;
    button.classList.toggle('page-size-btn-active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function renderResultsRows() {
  const resultados = lastResultData?.resultados || [];
  const totalPages = getTotalPages(resultados.length);
  currentPage = Math.min(currentPage, totalPages);

  const startIndex = (currentPage - 1) * pageSize;
  const pageRows = resultados.slice(startIndex, startIndex + pageSize);

  resultsBody.innerHTML = '';
  pageRows.forEach(row => {
    const isMastite = row.classe_prevista === 'Mastite';
    const tr = document.createElement('tr');
    tr.className = isMastite ? 'mastite' : 'saudavel';
    const probM = (row.prob_mastite * 100).toFixed(1) + '%';
    const probS = row.prob_saudavel !== undefined ? (row.prob_saudavel * 100).toFixed(1) + '%' : 'N/A';
    tr.innerHTML = `
      <td>${esc(row.id)}</td>
      <td><span class="pill ${isMastite ? 'pill-mastite' : 'pill-saudavel'}">${esc(row.classe_prevista)}</span></td>
      <td>${probM}</td>
      <td>${probS}</td>
    `;
    resultsBody.appendChild(tr);
  });

  renderPagination(resultados.length);
}

async function parseResponse(res) {
  const contentType = res.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return res.json();
  }

  const text = await res.text();
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text || 'Erro inesperado ao processar a requisição.' };
  }
}

async function sendAndRender(url, requestInit, errorFn) {
  try {
    const res  = await fetch(url, requestInit);
    const data = await parseResponse(res);
    if (!res.ok) {
      errorFn(normalizeErrorMessage(data.detail));
      return;
    }
    lastResultData = data;
    renderResults(data);
  } catch (err) {
    errorFn('Não foi possível conectar ao servidor. Verifique se ele está rodando.');
    console.error(err);
  }
}

function renderResults(data) {
  document.getElementById('summaryTotal').textContent    = data.total;
  document.getElementById('summaryMastite').textContent  = data.mastite;
  document.getElementById('summarySaudavel').textContent = data.saudavel;
  currentPage = 1;
  renderResultsRows();
  setHidden(resultsEl, false);
  resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

pageSizeOptions.querySelectorAll('.page-size-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const nextSize = Number(button.dataset.pageSize);
    if (nextSize === pageSize) return;
    pageSize = nextSize;
    currentPage = 1;
    syncPageSizeButtons();
    if (lastResultData) renderResultsRows();
  });
});

prevPageBtn.addEventListener('click', () => {
  if (currentPage === 1) return;
  currentPage -= 1;
  renderResultsRows();
});

nextPageBtn.addEventListener('click', () => {
  const totalPages = getTotalPages(lastResultData?.resultados?.length || 0);
  if (currentPage >= totalPages) return;
  currentPage += 1;
  renderResultsRows();
});

/* ═══════════════════════════════════════════════════════
   DOWNLOAD CSV DE RESULTADO
════════════════════════════════════════════════════════ */
document.getElementById('downloadCsvBtn').addEventListener('click', () => {
  if (!lastResultData) return;
  const headers = ['id', 'classe_prevista', 'prob_mastite', 'prob_saudavel'];
  const lines   = [headers.join(',')];
  lastResultData.resultados.forEach(r => {
    lines.push([
      `"${r.id}"`,
      r.classe_prevista,
      r.prob_mastite,
      r.prob_saudavel !== undefined ? r.prob_saudavel : '',
    ].join(','));
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'resultado_classificacao.csv';
  a.click();
  URL.revokeObjectURL(url);
});

/* ═══════════════════════════════════════════════════════
   NOVA ANÁLISE
════════════════════════════════════════════════════════ */
document.getElementById('newAnalysisBtn').addEventListener('click', () => {
  resetUpload();
  hideErrorManual();
  hideResults();
  lastResultData = null;
  currentPage = 1;
});

syncPageSizeButtons();
