const nav = document.getElementById('nav');
const navBurger = document.getElementById('navBurger');
const navMobile = document.getElementById('navMobile');
const tabCsv = document.getElementById('tabCsv');
const tabManual = document.getElementById('tabManual');
const resultsEl = document.getElementById('results');
const resultsBody = document.getElementById('resultsBody');
const resultsMeta = document.getElementById('resultsMeta');
const pageSizeOptions = document.getElementById('pageSizeOptions');
const resultsPagination = document.getElementById('resultsPagination');
const paginationInfo = document.getElementById('paginationInfo');
const prevPageBtn = document.getElementById('prevPageBtn');
const nextPageBtn = document.getElementById('nextPageBtn');
const resultsPolicy = document.getElementById('resultsPolicy');

const FIELD_META = [
  {
    key: 'Months_after_giving_birth',
    code: 'Months_after_giving_birth',
    label: 'Meses pós-parto',
    hint: 'Tempo desde o parto',
    placeholder: '3.0',
  },
  {
    key: 'IUFL',
    code: 'IUFL',
    label: 'Leitura interna do quarto mamário anterior esquerdo',
    hint: 'Canal anterior esquerdo interno',
    placeholder: '12.4',
  },
  {
    key: 'EUFL',
    code: 'EUFL',
    label: 'Leitura externa do quarto mamário anterior esquerdo',
    hint: 'Canal anterior esquerdo externo',
    placeholder: '11.8',
  },
  {
    key: 'IUFR',
    code: 'IUFR',
    label: 'Leitura interna do quarto mamário anterior direito',
    hint: 'Canal anterior direito interno',
    placeholder: '13.1',
  },
  {
    key: 'EUFR',
    code: 'EUFR',
    label: 'Leitura externa do quarto mamário anterior direito',
    hint: 'Canal anterior direito externo',
    placeholder: '12.9',
  },
  {
    key: 'IURL',
    code: 'IURL',
    label: 'Leitura interna do quarto mamário posterior esquerdo',
    hint: 'Canal posterior esquerdo interno',
    placeholder: '10.7',
  },
  {
    key: 'EURL',
    code: 'EURL',
    label: 'Leitura externa do quarto mamário posterior esquerdo',
    hint: 'Canal posterior esquerdo externo',
    placeholder: '10.5',
  },
  {
    key: 'IURR',
    code: 'IURR',
    label: 'Leitura interna do quarto mamário posterior direito',
    hint: 'Canal posterior direito interno',
    placeholder: '11.3',
  },
  {
    key: 'EURR',
    code: 'EURR',
    label: 'Leitura externa do quarto mamário posterior direito',
    hint: 'Canal posterior direito externo',
    placeholder: '11.0',
  },
  {
    key: 'Temperature',
    code: 'Temperature',
    label: 'Temperatura corporal',
    hint: 'Leitura em graus Celsius',
    placeholder: '38.6',
  },
];

const FIELD_LABELS = FIELD_META.reduce((acc, field) => {
  acc[field.key] = field.label;
  return acc;
}, { ID: 'ID' });

const CAMPOS = FIELD_META.map((field) => field.key);
const EXEMPLO = {
  ID: 'Vaca-001',
  Months_after_giving_birth: 3.0,
  IUFL: 12.4,
  EUFL: 11.8,
  IUFR: 13.1,
  EUFR: 12.9,
  IURL: 10.7,
  EURL: 10.5,
  IURR: 11.3,
  EURR: 11.0,
  Temperature: 38.6,
};

let selectedFile = null;
let currentPage = 1;
let pageSize = 10;
let filterText = '';
let filterClasse = 'todos';
let exportFormat = 'csv';
let exportDelimiter = ',';
let rowCounter = 0;
let lastResultData = null;
let lastInputData = null;
let chartTemperatura = null;
let chartMeses = null;

window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 10);
}, { passive: true });

navBurger.addEventListener('click', () => {
  const isOpen = navMobile.classList.toggle('open');
  navBurger.setAttribute('aria-expanded', String(isOpen));
});

navMobile.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navMobile.classList.remove('open');
    navBurger.setAttribute('aria-expanded', 'false');
  });
});

document.querySelectorAll('.tab-btn').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach((item) => item.classList.remove('tab-active'));
    button.classList.add('tab-active');

    const tab = button.dataset.tab;
    setHidden(tabCsv, tab !== 'csv');
    setHidden(tabManual, tab !== 'manual');
    hideError();
    hideErrorManual();
    hideResults();
  });
});

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setHidden(element, hidden) {
  element.hidden = hidden;
}

function getFieldLabel(fieldName) {
  return FIELD_LABELS[fieldName] || fieldName;
}

function getValidationMessage(item) {
  if (!item || typeof item !== 'object') {
    return 'Erro de validação.';
  }

  if (item.type === 'missing') {
    return 'Campo obrigatório.';
  }

  if (item.type === 'float_parsing' || item.type === 'int_parsing') {
    return 'Informe um número válido.';
  }

  if (item.type === 'string_type') {
    return 'Informe um texto válido.';
  }

  return item.msg || 'Erro de validação.';
}

function normalizeErrorMessage(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (!item || typeof item !== 'object') {
        return 'Erro de validação.';
      }

      const loc = Array.isArray(item.loc) ? item.loc : [];
      const fieldName = loc[loc.length - 1];
      const registrosIndex = loc.indexOf('registros');
      const parts = [];

      if (registrosIndex !== -1 && typeof loc[registrosIndex + 1] === 'number') {
        parts.push(`Animal ${loc[registrosIndex + 1] + 1}`);
      }

      if (typeof fieldName === 'string') {
        parts.push(getFieldLabel(fieldName));
      }

      const prefix = parts.length ? `${parts.join(' - ')}: ` : '';
      return `${prefix}${getValidationMessage(item)}`;
    }).join(' ');
  }

  if (detail && typeof detail === 'object') {
    return detail.msg || 'Erro inesperado ao processar a requisição.';
  }

  return detail || 'Erro inesperado ao processar a requisição.';
}

function esc(value) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(String(value)));
  return div.innerHTML;
}

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const fileSelected = document.getElementById('fileSelected');
const fileNameEl = document.getElementById('fileName');
const fileSizeEl = document.getElementById('fileSize');
const fileRemove = document.getElementById('fileRemove');
const analyzeBtn = document.getElementById('analyzeBtn');
const analyzeBtnText = document.getElementById('analyzeBtnText');
const analyzeBtnSpinner = document.getElementById('analyzeBtnSpinner');
const resultError = document.getElementById('resultError');
const errorMsg = document.getElementById('errorMsg');
const downloadExemplo = document.getElementById('downloadExemplo');

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

function showError(message) {
  errorMsg.textContent = message;
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

function setCsvLoading(loading) {
  analyzeBtn.disabled = loading;
  analyzeBtnText.style.display = loading ? 'none' : '';
  analyzeBtnSpinner.style.display = loading ? 'flex' : 'none';
}

uploadBtn.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('click', (event) => {
  if (event.target.closest('#downloadExemplo') || event.target.closest('#uploadBtn')) {
    return;
  }
  fileInput.click();
});

downloadExemplo.addEventListener('click', (event) => event.stopPropagation());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    showFileSelected(fileInput.files[0]);
  }
});

uploadArea.addEventListener('dragover', (event) => {
  event.preventDefault();
  uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
  uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (event) => {
  event.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) {
    showError('Apenas arquivos .csv são aceitos.');
    return;
  }
  showFileSelected(file);
});

fileRemove.addEventListener('click', resetUpload);

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  if (selectedFile.size > 5 * 1024 * 1024) {
    showError(`Arquivo muito grande (${formatBytes(selectedFile.size)}). Limite: 5 MB.`);
    return;
  }

  hideError();
  setCsvLoading(true);

  try {
    const csvText = await readFileAsText(selectedFile);
    const csvRows = parseCSVText(csvText);
    lastInputData = buildInputMap(csvRows);
  } catch {
    lastInputData = null;
  }

  const formData = new FormData();
  formData.append('arquivo', selectedFile);

  await sendAndRender('/predict', { method: 'POST', body: formData }, showError);

  setCsvLoading(false);
});

const manualBody = document.getElementById('manualBody');
const addRowBtn = document.getElementById('addRowBtn');
const analyzeManualBtn = document.getElementById('analyzeManualBtn');
const manualBtnText = document.getElementById('manualBtnText');
const manualBtnSpinner = document.getElementById('manualBtnSpinner');
const resultErrorManual = document.getElementById('resultErrorManual');
const errorMsgManual = document.getElementById('errorMsgManual');

function bindNumericInputGuards(input) {
  input.addEventListener('keydown', (event) => {
    if (event.key === '-') {
      event.preventDefault();
    }
  });

  input.addEventListener('input', () => {
    if (input.value.trim().startsWith('-')) {
      input.value = input.value.replace('-', '');
    }
    clearFieldError(input);
    hideErrorManual();
  });
}

function createFieldShell({ field, code, label, hint, type = 'number', placeholder = '', value = '' }) {
  const wrapper = document.createElement('div');
  wrapper.className = `manual-field${field === 'ID' ? ' manual-field-id' : ''}`;

  const fieldLabel = document.createElement('label');
  fieldLabel.className = 'manual-field-label';
  fieldLabel.setAttribute('for', `${field}-${rowCounter}`);
  fieldLabel.innerHTML = code
    ? `<span class="manual-field-code">${esc(code)}</span>${esc(label)}`
    : esc(label);

  const input = document.createElement('input');
  input.id = `${field}-${rowCounter}`;
  input.type = type;
  input.dataset.field = field;
  input.placeholder = placeholder;
  input.value = value ?? '';
  input.autocomplete = 'off';

  if (type === 'number') {
    input.step = 'any';
    input.min = '0';
    input.inputMode = 'decimal';
    bindNumericInputGuards(input);
  } else {
    input.addEventListener('input', () => {
      clearFieldError(input);
      hideErrorManual();
    });
  }

  const hintEl = document.createElement('span');
  hintEl.className = 'manual-field-hint';
  hintEl.textContent = hint;

  const errorEl = document.createElement('span');
  errorEl.className = 'manual-field-error';
  errorEl.hidden = true;

  wrapper.append(fieldLabel, input, hintEl, errorEl);
  return wrapper;
}

function createRow(defaults = {}) {
  rowCounter += 1;

  const card = document.createElement('article');
  card.className = 'manual-card';
  card.dataset.rowId = String(rowCounter);

  const header = document.createElement('div');
  header.className = 'manual-card-head';

  const titleWrap = document.createElement('div');
  titleWrap.className = 'manual-card-title-wrap';

  const badge = document.createElement('span');
  badge.className = 'manual-card-badge';

  const title = document.createElement('h3');
  title.className = 'manual-card-title';
  title.textContent = 'Registro manual';

  const subtitle = document.createElement('p');
  subtitle.className = 'manual-card-subtitle';
  subtitle.textContent = 'Campos obrigatórios para o modelo de classificação.';

  titleWrap.append(badge, title, subtitle);

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'btn-remove-row';
  removeBtn.title = 'Remover animal';
  removeBtn.setAttribute('aria-label', 'Remover animal');
  removeBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12"/>
    </svg>
  `;
  removeBtn.addEventListener('click', () => {
    card.remove();
    if (manualBody.querySelectorAll('.manual-card').length === 0) {
      addRow(EXEMPLO);
    }
    syncManualCards();
    hideErrorManual();
  });

  header.append(titleWrap, removeBtn);

  const grid = document.createElement('div');
  grid.className = 'manual-card-grid';

  grid.appendChild(createFieldShell({
    field: 'ID',
    label: 'Identificador do animal',
    hint: 'Use um código único para identificar o animal.',
    type: 'text',
    placeholder: `Vaca-${String(rowCounter).padStart(3, '0')}`,
    value: defaults.ID || `Vaca-${String(rowCounter).padStart(3, '0')}`,
  }));

  FIELD_META.forEach((meta) => {
    grid.appendChild(createFieldShell({
      field: meta.key,
      code: meta.code,
      label: meta.label,
      hint: meta.hint,
      placeholder: meta.placeholder,
      value: defaults[meta.key] ?? '',
    }));
  });

  card.append(header, grid);
  return card;
}

function syncManualCards() {
  manualBody.querySelectorAll('.manual-card').forEach((card, index) => {
    const badge = card.querySelector('.manual-card-badge');
    if (badge) {
      badge.textContent = `Animal ${String(index + 1).padStart(2, '0')}`;
    }
  });
}

function addRow(defaults = {}) {
  manualBody.appendChild(createRow(defaults));
  syncManualCards();
}

function getFieldWrapper(input) {
  return input.closest('.manual-field');
}

function setFieldError(input, message) {
  const wrapper = getFieldWrapper(input);
  if (!wrapper) return;

  wrapper.classList.add('manual-field-invalid');
  input.setAttribute('aria-invalid', 'true');

  const errorEl = wrapper.querySelector('.manual-field-error');
  if (errorEl) {
    errorEl.textContent = message;
    errorEl.hidden = false;
  }
}

function clearFieldError(input) {
  const wrapper = getFieldWrapper(input);
  if (!wrapper) return;

  wrapper.classList.remove('manual-field-invalid');
  input.removeAttribute('aria-invalid');

  const errorEl = wrapper.querySelector('.manual-field-error');
  if (errorEl) {
    errorEl.textContent = '';
    errorEl.hidden = true;
  }
}

function clearAllManualFieldErrors() {
  manualBody.querySelectorAll('.manual-field input').forEach((input) => clearFieldError(input));
}

function getManualRows() {
  return Array.from(manualBody.querySelectorAll('.manual-card')).map((card) => {
    const row = {};
    card.querySelectorAll('input').forEach((input) => {
      row[input.dataset.field] = input.value.trim();
    });
    return row;
  });
}

function validateManualRows() {
  const rows = [];
  let firstInvalidInput = null;

  clearAllManualFieldErrors();

  manualBody.querySelectorAll('.manual-card').forEach((card) => {
    const row = {};

    card.querySelectorAll('input').forEach((input) => {
      row[input.dataset.field] = input.value.trim();
    });

    const idInput = card.querySelector('input[data-field="ID"]');
    const cleanId = row.ID.trim();
    if (!cleanId) {
      setFieldError(idInput, 'Informe um ID para o animal.');
      firstInvalidInput = firstInvalidInput || idInput;
    }
    row.ID = cleanId;

    FIELD_META.forEach((meta) => {
      const input = card.querySelector(`input[data-field="${meta.key}"]`);
      const rawValue = row[meta.key];

      if (rawValue === '') {
        setFieldError(input, 'Campo obrigatório.');
        firstInvalidInput = firstInvalidInput || input;
        return;
      }

      const numericValue = Number(rawValue);
      if (!Number.isFinite(numericValue)) {
        setFieldError(input, 'Informe um número válido.');
        firstInvalidInput = firstInvalidInput || input;
        return;
      }

      if (numericValue < 0) {
        setFieldError(input, 'Valores negativos não são aceitos.');
        firstInvalidInput = firstInvalidInput || input;
        return;
      }

      row[meta.key] = numericValue;
    });

    rows.push(row);
  });

  return {
    hasErrors: Boolean(firstInvalidInput),
    firstInvalidInput,
    rows,
  };
}

function showErrorManual(message) {
  errorMsgManual.textContent = message;
  setHidden(resultErrorManual, false);
}

function showTabScopedError(message) {
  if (!tabManual.hidden) {
    hideError();
    showErrorManual(message);
    return;
  }

  hideErrorManual();
  showError(message);
}

function hideErrorManual() {
  setHidden(resultErrorManual, true);
  errorMsgManual.textContent = '';
}

function setManualLoading(loading) {
  analyzeManualBtn.disabled = loading;
  manualBtnText.style.display = loading ? 'none' : '';
  manualBtnSpinner.style.display = loading ? 'flex' : 'none';
}

addRow(EXEMPLO);

addRowBtn.addEventListener('click', () => addRow());

analyzeManualBtn.addEventListener('click', async () => {
  const currentRows = getManualRows();
  if (currentRows.length === 0) {
    showErrorManual('Adicione pelo menos um animal.');
    return;
  }

  const validation = validateManualRows();
  if (validation.hasErrors) {
    showErrorManual('Revise os campos destacados antes de analisar.');
    validation.firstInvalidInput?.focus();
    validation.firstInvalidInput?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }

  hideErrorManual();
  setManualLoading(true);

  lastInputData = buildInputMap(validation.rows.map((r) => ({
    ID: r.ID,
    Temperature: r.Temperature,
    Months_after_giving_birth: r.Months_after_giving_birth,
  })));

  await sendAndRender('/predict/manual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ registros: validation.rows }),
  }, showErrorManual);

  setManualLoading(false);
});

const filterIdInput = document.getElementById('filterIdInput');
const filterClasseSelect = document.getElementById('filterClasseSelect');
const clearFiltersBtn = document.getElementById('clearFiltersBtn');
const filterSummary = document.getElementById('filterSummary');
const exportFormatToggle = document.getElementById('exportFormatToggle');
const delimiterGroup = document.getElementById('delimiterGroup');
const delimiterSelect = document.getElementById('delimiterSelect');
const downloadResultsBtn = document.getElementById('downloadResultsBtn');
const downloadResultsText = document.getElementById('downloadResultsText');

function getFilteredResults() {
  const resultados = lastResultData?.resultados || [];
  return resultados.filter((row) => {
    const matchId = filterText === '' || row.id.toLowerCase().includes(filterText.toLowerCase());
    const matchClasse = filterClasse === 'todos' || row.nivel_risco === filterClasse;
    return matchId && matchClasse;
  });
}

function syncFilterSummary() {
  const activeFilters = [];

  if (filterText !== '') {
    activeFilters.push(`ID contém "${filterText}"`);
  }

  if (filterClasse !== 'todos') {
    activeFilters.push(`Risco: ${filterClasse}`);
  }

  filterSummary.textContent = activeFilters.length
    ? `Filtros ativos: ${activeFilters.join(' | ')}`
    : 'Sem filtros aplicados';
}

function getNormalizedDelimiter() {
  return exportDelimiter === 'tab' ? '\t' : exportDelimiter;
}

function getExportFileExtension() {
  return exportFormat === 'xlsx' ? 'xlsx' : 'csv';
}

function getExportButtonLabel() {
  return `Baixar resultado em ${exportFormat.toUpperCase()}`;
}

function syncExportControls() {
  exportFormatToggle.querySelectorAll('.export-format-btn').forEach((button) => {
    const isActive = button.dataset.exportFormat === exportFormat;
    button.classList.toggle('export-format-btn-active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });

  delimiterGroup.hidden = exportFormat !== 'csv';
  downloadResultsText.textContent = getExportButtonLabel();
}

function syncClearFiltersBtn() {
  const hasFilter = filterText !== '' || filterClasse !== 'todos';
  clearFiltersBtn.hidden = !hasFilter;
  syncFilterSummary();
}

filterIdInput.addEventListener('input', () => {
  filterText = filterIdInput.value.trim();
  currentPage = 1;
  syncClearFiltersBtn();
  if (lastResultData) {
    renderResultsRows();
  }
});

filterClasseSelect.addEventListener('change', () => {
  filterClasse = filterClasseSelect.value;
  currentPage = 1;
  syncClearFiltersBtn();
  if (lastResultData) {
    renderResultsRows();
  }
});

clearFiltersBtn.addEventListener('click', () => {
  filterText = '';
  filterClasse = 'todos';
  filterIdInput.value = '';
  filterClasseSelect.value = 'todos';
  currentPage = 1;
  syncClearFiltersBtn();
  if (lastResultData) {
    renderResultsRows();
  }
});

function getTotalPages(totalItems) {
  return Math.max(1, Math.ceil(totalItems / pageSize));
}

function renderPagination(filteredCount) {
  const total = lastResultData?.resultados?.length || 0;
  const totalPages = getTotalPages(filteredCount);
  const start = filteredCount === 0 ? 0 : ((currentPage - 1) * pageSize) + 1;
  const end = Math.min(currentPage * pageSize, filteredCount);

  if ((filterText !== '' || filterClasse !== 'todos') && filteredCount !== total) {
    resultsMeta.textContent = `Mostrando ${start} a ${end} de ${filteredCount} animais filtrados (${total} no total)`;
  } else {
    resultsMeta.textContent = `Mostrando ${start} a ${end} de ${filteredCount} animais`;
  }

  paginationInfo.textContent = `Página ${currentPage} de ${totalPages}`;
  prevPageBtn.disabled = currentPage === 1;
  nextPageBtn.disabled = currentPage === totalPages;
  setHidden(resultsPagination, filteredCount <= pageSize);
}

function syncPageSizeButtons() {
  pageSizeOptions.querySelectorAll('.page-size-btn').forEach((button) => {
    const isActive = Number(button.dataset.pageSize) === pageSize;
    button.classList.toggle('page-size-btn-active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
}

function getRiskRowClass(row) {
  if (row.nivel_risco === 'Alta suspeita') return 'risk-high';
  if (row.nivel_risco === 'Monitorar com cautela') return 'risk-medium';
  return 'risk-low';
}

function getRiskPillClass(row) {
  if (row.nivel_risco === 'Alta suspeita') return 'pill-risk-high';
  if (row.nivel_risco === 'Monitorar com cautela') return 'pill-risk-medium';
  return 'pill-risk-low';
}

function renderResultsRows() {
  const filtered = getFilteredResults();
  const totalPages = getTotalPages(filtered.length);
  currentPage = Math.min(currentPage, totalPages);

  const startIndex = (currentPage - 1) * pageSize;
  const pageRows = filtered.slice(startIndex, startIndex + pageSize);

  resultsBody.innerHTML = '';

  pageRows.forEach((row) => {
    const tableRow = document.createElement('tr');
    tableRow.className = getRiskRowClass(row);

    const probMastite = `${(row.prob_mastite * 100).toFixed(1)}%`;
    const action = row.recomendacao || 'Sem recomendação disponível.';
    const modelSignal = row.classe_prevista ? `Sinal do modelo: ${row.classe_prevista}` : '';

    tableRow.innerHTML = `
      <td>${esc(row.id)}</td>
      <td><span class="pill ${getRiskPillClass(row)}">${esc(row.nivel_risco)}</span></td>
      <td>${probMastite}</td>
      <td>${esc(action)}${modelSignal ? `<br><small>${esc(modelSignal)}</small>` : ''}</td>
    `;

    resultsBody.appendChild(tableRow);
  });

  renderPagination(filtered.length);
}

async function parseResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }

  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text || 'Erro inesperado ao processar a requisição.' };
  }
}

async function sendAndRender(url, requestInit, errorFn) {
  try {
    const response = await fetch(url, requestInit);
    const data = await parseResponse(response);

    if (!response.ok) {
      errorFn(normalizeErrorMessage(data.detail));
      return;
    }

    lastResultData = data;
    renderResults(data);
  } catch (error) {
    errorFn('Não foi possível conectar ao servidor. Verifique se ele está rodando.');
    console.error(error);
  }
}

function renderResults(data) {
  if (data.politica_triagem) {
    const review = (data.politica_triagem.limiar_revisao * 100).toFixed(0);
    const high = (data.politica_triagem.limiar_alta_suspeita * 100).toFixed(0);
    resultsPolicy.textContent =
      `Política atual: revisar antes da liberação do leite a partir de ${review}% de probabilidade de mastite e tratar como alta suspeita a partir de ${high}%.`;
  }

  filterText = '';
  filterClasse = 'todos';
  filterIdInput.value = '';
  filterClasseSelect.value = 'todos';
  currentPage = 1;
  syncClearFiltersBtn();
  syncExportControls();
  renderResultsRows();

  resetDashboardCanvases();
  const inputMap = lastInputData || {};
  const merged = mergeResultsWithInput(data.resultados || [], inputMap);
  renderDashboard(data, merged);

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
    if (lastResultData) {
      renderResultsRows();
    }
  });
});

prevPageBtn.addEventListener('click', () => {
  if (currentPage === 1) return;
  currentPage -= 1;
  renderResultsRows();
});

nextPageBtn.addEventListener('click', () => {
  const totalPages = getTotalPages(getFilteredResults().length);
  if (currentPage >= totalPages) return;
  currentPage += 1;
  renderResultsRows();
});

exportFormatToggle.querySelectorAll('.export-format-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const nextFormat = button.dataset.exportFormat;
    if (!nextFormat || nextFormat === exportFormat) return;
    exportFormat = nextFormat;
    syncExportControls();
  });
});

delimiterSelect.addEventListener('change', () => {
  exportDelimiter = delimiterSelect.value;
});

downloadResultsBtn.addEventListener('click', async () => {
  if (!lastResultData) return;

  hideError();
  hideErrorManual();
  downloadResultsBtn.disabled = true;

  try {
    const response = await fetch('/export/results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        formato: exportFormat,
        delimitador: getNormalizedDelimiter(),
        resultados: lastResultData.resultados,
      }),
    });

    if (!response.ok) {
      const data = await parseResponse(response);
      showTabScopedError(normalizeErrorMessage(data.detail));
      return;
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `resultado_classificacao.${getExportFileExtension()}`;
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    showTabScopedError('Não foi possível gerar o arquivo de exportação no momento.');
    console.error(error);
  } finally {
    downloadResultsBtn.disabled = false;
  }
});

document.getElementById('newAnalysisBtn').addEventListener('click', () => {
  resetUpload();
  hideErrorManual();
  hideResults();
  clearAllManualFieldErrors();
  lastResultData = null;
  lastInputData = null;
  currentPage = 1;
  resetDashboardCanvases();
});

/* ═══════════════════════════════════════════════════════
   DASHBOARD
════════════════════════════════════════════════════════ */

function parseCSVText(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];

  const sep = lines[0].includes(';') ? ';' : ',';
  const headers = lines[0].split(sep).map((h) => h.trim().replace(/^["']|["']$/g, ''));

  return lines.slice(1).filter((l) => l.trim()).map((line) => {
    const values = line.split(sep).map((v) => v.trim().replace(/^["']|["']$/g, ''));
    const row = {};
    headers.forEach((h, i) => { row[h] = values[i] || ''; });
    return row;
  });
}

async function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function buildInputMap(rows) {
  const map = {};
  rows.forEach((row) => {
    const id = String(row.ID || '').trim();
    if (!id) return;
    map[id] = {
      temperatura: parseFloat(row.Temperature),
      meses: parseFloat(row.Months_after_giving_birth),
    };
  });
  return map;
}

function mergeResultsWithInput(resultados, inputMap) {
  return resultados.map((r) => {
    const input = inputMap[r.id] || {};
    return {
      ...r,
      temperatura: Number.isFinite(input.temperatura) ? input.temperatura : null,
      meses: Number.isFinite(input.meses) ? input.meses : null,
    };
  });
}

function renderDashboard(data, merged) {
  const total = data.total || 0;
  const alta = data.alta_suspeita || 0;
  const monitorar = data.monitorar || 0;
  const baixo = data.baixo_risco || 0;
  const revisao = data.requer_revisao || 0;
  const mastite = data.mastite || 0;

  const pct = (v) => total ? `${((v / total) * 100).toFixed(1)}%` : '0%';

  document.getElementById('dashAlta').textContent = alta;
  document.getElementById('dashAltaPct').textContent = `${pct(alta)} do total`;
  document.getElementById('dashMonitorar').textContent = monitorar;
  document.getElementById('dashMonitorarPct').textContent = `${pct(monitorar)} do total`;
  document.getElementById('dashBaixo').textContent = baixo;
  document.getElementById('dashBaixoPct').textContent = `${pct(baixo)} do total`;
  document.getElementById('dashRevisao').textContent = revisao;
  document.getElementById('dashRevisaoPct').textContent = `${pct(revisao)} do total`;
  document.getElementById('dashMastite').textContent = mastite;
  document.getElementById('dashMastitePct').textContent = `${pct(mastite)} do total`;

  const probMedia = total > 0
    ? (merged.reduce((sum, r) => sum + r.prob_mastite, 0) / total)
    : 0;
  document.getElementById('dashProbMedia').textContent = probMedia.toFixed(3);

  renderChartTemperatura(merged);
  renderUrgencyTable(merged);
  renderChartMeses(merged);
}

function renderChartTemperatura(merged) {
  const groups = { 'Alta suspeita': [], 'Monitorar com cautela': [], 'Baixo risco': [] };
  merged.forEach((r) => {
    if (r.temperatura != null && groups[r.nivel_risco]) {
      groups[r.nivel_risco].push(r.temperatura);
    }
  });

  const avg = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const labels = ['Alta suspeita', 'Monitorar com cautela', 'Baixo risco'];
  const values = labels.map((l) => parseFloat(avg(groups[l]).toFixed(2)));
  const colors = ['#E24B4A', '#EF9F27', '#1D9E75'];

  const hasData = values.some((v) => v > 0);

  const ctx = document.getElementById('chartTemperatura');
  if (chartTemperatura) { chartTemperatura.destroy(); chartTemperatura = null; }

  if (!hasData) {
    ctx.parentElement.innerHTML = '<p style="text-align:center;color:#9e9e9e;padding:2rem 0;font-size:.88rem;">Dados de temperatura não disponíveis para este conjunto.</p>';
    return;
  }

  chartTemperatura = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderRadius: 6,
        maxBarThickness: 72,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => `${c.parsed.y.toFixed(2)} °C`,
          },
        },
      },
      scales: {
        y: {
          min: 38,
          title: { display: true, text: '°C', font: { size: 12 } },
          ticks: { callback: (v) => `${v}°C` },
          grid: { color: 'rgba(0,0,0,.06)' },
        },
        x: {
          grid: { display: false },
        },
      },
    },
  });
}

function renderUrgencyTable(merged) {
  const body = document.getElementById('dashUrgencyBody');
  const note = document.getElementById('dashUrgencyNote');
  body.innerHTML = '';

  const highRisk = merged
    .filter((r) => r.nivel_risco === 'Alta suspeita')
    .sort((a, b) => b.prob_mastite - a.prob_mastite)
    .slice(0, 15);

  if (highRisk.length === 0) {
    body.innerHTML = '<tr><td colspan="4" style="text-align:center;color:#9e9e9e;padding:1.5rem;">Nenhum animal com alta suspeita nesta análise.</td></tr>';
    note.textContent = '';
    return;
  }

  highRisk.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.className = 'urgency-high';
    tr.innerHTML = `
      <td>${i + 1}</td>
      <td>${esc(r.id)}</td>
      <td>${(r.prob_mastite * 100).toFixed(1)}%</td>
      <td><span class="pill pill-risk-high">${esc(r.nivel_risco)}</span></td>
    `;
    body.appendChild(tr);
  });

  const totalAlta = merged.filter((r) => r.nivel_risco === 'Alta suspeita').length;
  note.textContent = totalAlta > 15
    ? `Mostrando os 15 mais urgentes de ${totalAlta} animais com alta suspeita.`
    : `${totalAlta} animal(is) com alta suspeita.`;
}

function renderChartMeses(merged) {
  const buckets = {};
  merged.forEach((r) => {
    if (r.meses == null) return;
    const mes = Math.max(1, Math.min(6, Math.round(r.meses)));
    if (!buckets[mes]) buckets[mes] = { 'Alta suspeita': 0, 'Monitorar com cautela': 0, 'Baixo risco': 0 };
    if (buckets[mes][r.nivel_risco] != null) {
      buckets[mes][r.nivel_risco]++;
    }
  });

  const labels = [1, 2, 3, 4, 5, 6].map((m) => `Mês ${m}`);
  const alta = [1, 2, 3, 4, 5, 6].map((m) => (buckets[m] || {})['Alta suspeita'] || 0);
  const monitorar = [1, 2, 3, 4, 5, 6].map((m) => (buckets[m] || {})['Monitorar com cautela'] || 0);
  const baixo = [1, 2, 3, 4, 5, 6].map((m) => (buckets[m] || {})['Baixo risco'] || 0);

  const hasData = [...alta, ...monitorar, ...baixo].some((v) => v > 0);

  const ctx = document.getElementById('chartMeses');
  if (chartMeses) { chartMeses.destroy(); chartMeses = null; }

  if (!hasData) {
    ctx.parentElement.innerHTML = '<p style="text-align:center;color:#9e9e9e;padding:2rem 0;font-size:.88rem;">Dados de meses pós-parto não disponíveis para este conjunto.</p>';
    return;
  }

  chartMeses = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Alta suspeita', data: alta, backgroundColor: '#E24B4A', borderRadius: 4, maxBarThickness: 36 },
        { label: 'Monitorar com cautela', data: monitorar, backgroundColor: '#EF9F27', borderRadius: 4, maxBarThickness: 36 },
        { label: 'Baixo risco', data: baixo, backgroundColor: '#1D9E75', borderRadius: 4, maxBarThickness: 36 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 12 } },
        },
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1, precision: 0 },
          grid: { color: 'rgba(0,0,0,.06)' },
        },
        x: {
          grid: { display: false },
        },
      },
    },
  });
}

function resetDashboardCanvases() {
  if (chartTemperatura) { chartTemperatura.destroy(); chartTemperatura = null; }
  if (chartMeses) { chartMeses.destroy(); chartMeses = null; }

  ['chartTemperatura', 'chartMeses'].forEach((id) => {
    const container = document.getElementById(id)?.parentElement;
    if (container && !container.querySelector('canvas')) {
      container.innerHTML = `<canvas id="${id}"></canvas>`;
    }
  });
}

syncPageSizeButtons();
syncClearFiltersBtn();
syncExportControls();
