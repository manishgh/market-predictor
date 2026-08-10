document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.querySelector("#model-select");
  const tickerInput = document.querySelector("#tickers");
  const executeButton = document.querySelector("#execute-btn");
  const modelStatus = document.querySelector("#model-status");
  const requestStatus = document.querySelector("#request-status");
  const resultContext = document.querySelector("#result-context");
  const resultCount = document.querySelector("#result-count");
  const predictionRows = document.querySelector("#prediction-rows");
  const rejectionList = document.querySelector("#rejection-list");
  let models = [];

  function selectedModel() {
    return models.find((model) => model.id === modelSelect.value);
  }

  function updateModelState() {
    const model = selectedModel();
    if (!model) {
      modelStatus.textContent = "Model catalog is unavailable.";
      executeButton.disabled = true;
      return;
    }
    modelStatus.textContent = `${model.training_status}: ${model.reason}`;
    executeButton.disabled = !model.research_scoring_available;
  }

  function parseTickers() {
    return [...new Set(
      tickerInput.value
        .split(/[\s,]+/)
        .map((ticker) => ticker.trim().toUpperCase())
        .filter(Boolean)
    )];
  }

  function setStatus(message, isError = false) {
    requestStatus.textContent = message;
    requestStatus.classList.toggle("error", isError);
  }

  function appendCell(row, value, className = "") {
    const cell = document.createElement("td");
    cell.textContent = value;
    if (className) cell.className = className;
    row.appendChild(cell);
    return cell;
  }

  function featureDetails(features) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const entries = Object.entries(features);
    summary.textContent = `${entries.length} calculated features`;
    details.appendChild(summary);
    const list = document.createElement("ul");
    list.className = "feature-list";
    entries.forEach(([name, value]) => {
      const item = document.createElement("li");
      const code = document.createElement("code");
      const number = document.createElement("span");
      code.textContent = name;
      number.textContent = Number(value).toPrecision(6);
      item.append(code, number);
      list.appendChild(item);
    });
    details.appendChild(list);
    return details;
  }

  function renderResult(data) {
    predictionRows.replaceChildren();
    rejectionList.replaceChildren();
    resultCount.textContent = String(data.predictions.length);
    resultContext.textContent = `${data.model.label}; ${data.as_of_utc}; non-actionable research output.`;

    if (data.predictions.length === 0) {
      const row = document.createElement("tr");
      const cell = appendCell(row, "No current feature rows were scored.", "empty");
      cell.colSpan = 5;
      predictionRows.appendChild(row);
    } else {
      data.predictions.forEach((prediction) => {
        const row = document.createElement("tr");
        appendCell(row, String(prediction.rank));
        appendCell(row, prediction.ticker);
        appendCell(row, Number(prediction.score).toFixed(6), "score");
        appendCell(row, prediction.feature_available_at_utc || "Unavailable");
        const featureCell = document.createElement("td");
        featureCell.appendChild(featureDetails(prediction.features));
        row.appendChild(featureCell);
        predictionRows.appendChild(row);
      });
    }

    if (data.rejected.length === 0) {
      const item = document.createElement("li");
      item.textContent = "None";
      rejectionList.appendChild(item);
    } else {
      data.rejected.forEach((rejection) => {
        const item = document.createElement("li");
        item.textContent = `${rejection.ticker}: ${rejection.reason}`;
        rejectionList.appendChild(item);
      });
    }
  }

  async function loadModels() {
    try {
      const response = await fetch("/v1/research/models");
      if (!response.ok) throw new Error(`Model catalog request failed (${response.status}).`);
      const data = await response.json();
      models = data.models;
      modelSelect.replaceChildren();
      models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.label;
        modelSelect.appendChild(option);
      });
      modelSelect.disabled = false;
      const available = models.find((model) => model.research_scoring_available);
      if (available) modelSelect.value = available.id;
      updateModelState();
    } catch (error) {
      setStatus(error.message, true);
    }
  }

  modelSelect.addEventListener("change", updateModelState);
  executeButton.addEventListener("click", async () => {
    const tickers = parseTickers();
    if (tickers.length === 0) {
      setStatus("Enter at least one valid US ticker.", true);
      tickerInput.focus();
      return;
    }
    const model = selectedModel();
    if (!model || !model.research_scoring_available) {
      setStatus("The selected model has no verified research candidate.", true);
      return;
    }
    executeButton.disabled = true;
    executeButton.textContent = "Running...";
    setStatus("Loading registered causal features...");
    try {
      const response = await fetch("/v1/research/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: model.id, tickers }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Prediction failed (${response.status}).`);
      renderResult(data);
      setStatus(data.warning);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      executeButton.textContent = "Run research score";
      executeButton.disabled = !selectedModel()?.research_scoring_available;
    }
  });

  loadModels();
});
