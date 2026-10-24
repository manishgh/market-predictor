document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = document.querySelector("#model-select");
  const sessionInput = document.querySelector("#session-date");
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
    executeButton.disabled = !model?.historical_predictions_available;
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
      data.model.label + "; " + data.session_date + "; saved temporal validation, not live.";
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
      const response = await fetch("/v1/research/models");
      if (!response.ok) throw new Error("Model catalog request failed (" + response.status + ").");
      models = (await response.json()).models;
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
  executeButton.addEventListener("click", async () => {
    const tickers = [...new Set(tickerInput.value.split(/[\s,]+/).map((ticker) => ticker.trim().toUpperCase()).filter(Boolean))];
    if (!tickers.length || !sessionInput.value) {
      requestStatus.textContent = "Enter US tickers and an exact historical session.";
      return;
    }
    executeButton.disabled = true;
    requestStatus.textContent = "Reading verified saved predictions...";
    try {
      const response = await fetch("/v1/research/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelSelect.value, tickers, session_date: sessionInput.value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Invalid historical prediction request.");
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
