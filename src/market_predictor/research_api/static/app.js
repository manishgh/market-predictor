document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.querySelector("#model-select");
  const sessionInput = document.querySelector("#session-date");
  const queryMode = document.querySelector("#query-mode");
  const accessToken = document.querySelector("#access-token");
  const tickerInput = document.querySelector("#tickers");
  const executeButton = document.querySelector("#execute-btn");
  const modelStatus = document.querySelector("#model-status");
  const requestStatus = document.querySelector("#request-status");
  const predictionRows = document.querySelector("#prediction-rows");
  let models = [];
  const selectedModel = () => models.find((model) => model.id === modelSelect.value);

  function updateModelState() {
    const model = selectedModel();
    modelStatus.textContent = model ? model.reason : "Model catalog is unavailable.";
    const fresh = queryMode.value === "forecast";
    executeButton.disabled = !model?.integrity_verified;
    executeButton.textContent = fresh ? "Request fresh forecast" : "Read saved predictions";
    sessionInput.disabled = fresh;
    sessionInput.required = !fresh;
    if (fresh && model && !model.fresh_inputs_published) {
      modelStatus.textContent = "Model trained. Current observed stock/news features have not been published.";
    }
    sessionInput.min = model?.validation_start || "";
    sessionInput.max = model?.validation_end || "";
    if (!sessionInput.value && model?.validation_start) sessionInput.value = model.validation_start;
  }

  function appendCell(row, value) {
    const cell = document.createElement("td");
    cell.textContent = value;
    row.appendChild(cell);
  }

  function renderResult(data) {
    predictionRows.replaceChildren();
    document.querySelector("#result-count").textContent = String(data.predictions.length);
    document.querySelector("#result-context").textContent =
      data.model.label + "; " + data.session_date + "; " + data.prediction_basis;
    for (const prediction of data.predictions) {
      const row = document.createElement("tr");
      appendCell(row, prediction.ticker);
      appendCell(row, prediction.predicted_excess_return === null
        ? "Unavailable"
        : (prediction.predicted_excess_return * 100).toFixed(3) + " pp");
      appendCell(row, prediction.prediction_status);
      appendCell(row, prediction.decision_time_utc);
      predictionRows.appendChild(row);
    }
    if (!data.predictions.length) {
      const row = document.createElement("tr");
      appendCell(row, "No saved rows for this exact session and ticker selection.");
      row.firstChild.colSpan = 4;
      predictionRows.appendChild(row);
    }
    const rejections = document.querySelector("#rejection-list");
    rejections.replaceChildren();
    for (const rejection of data.rejected) {
      const item = document.createElement("li");
      item.textContent = rejection.ticker + ": " + rejection.reason;
      rejections.appendChild(item);
    }
    if (!data.rejected.length) {
      const item = document.createElement("li");
      item.textContent = "None";
      rejections.appendChild(item);
    }
    document.querySelector("#provenance").textContent = JSON.stringify(data.provenance, null, 2);
  }

  async function loadModels() {
    try {
      const response = await fetch("/v1/research/models", { headers: requestHeaders() });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message || "Model catalog request failed (" + response.status + ").");
      models = data.models;
      modelSelect.replaceChildren();
      for (const model of models) {
        const option = document.createElement("option");
        option.value = model.id;
        option.textContent = model.label;
        modelSelect.appendChild(option);
      }
      modelSelect.disabled = false;
      updateModelState();
    } catch (error) {
      requestStatus.textContent = error.message;
    }
  }

  modelSelect.addEventListener("change", updateModelState);
  queryMode.addEventListener("change", updateModelState);
  accessToken.addEventListener("change", loadModels);
  function requestHeaders() {
    const headers = { "Content-Type": "application/json" };
    if (accessToken.value.trim()) headers.Authorization = "Bearer " + accessToken.value.trim();
    return headers;
  }
  executeButton.addEventListener("click", async () => {
    const tickers = [...new Set(tickerInput.value.split(/[\s,]+/).map((ticker) => ticker.trim().toUpperCase()).filter(Boolean))];
    const fresh = queryMode.value === "forecast";
    if (!tickers.length || (!fresh && !sessionInput.value)) {
      requestStatus.textContent = "Enter US tickers and, for historical queries, an exact session.";
      return;
    }
    executeButton.disabled = true;
    requestStatus.textContent = fresh ? "Checking current inputs and calculating forecast..." : "Reading verified saved predictions...";
    try {
      const payload = { model_id: modelSelect.value, tickers };
      if (!fresh) payload.session_date = sessionInput.value;
      const response = await fetch(fresh ? "/v1/research/forecast" : "/v1/research/predict", {
        method: "POST",
        headers: requestHeaders(),
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error?.message || "The return prediction request failed (" + response.status + ").");
      renderResult(data);
      requestStatus.textContent = data.warning;
    } catch (error) {
      requestStatus.textContent = error.message;
    } finally {
      updateModelState();
    }
  });
  loadModels();
});
