const API_BASE = "";
const form = document.querySelector("#prediction-form");
const errorBox = document.querySelector("#error-message");
const submitButton = document.querySelector("#submit-button");
const apiStatus = document.querySelector("#api-status");
const apiPill = document.querySelector(".api-pill");
const resultTitle = document.querySelector("#result-title");
const resultSummary = document.querySelector("#result-summary");
const probabilityValue = document.querySelector("#probability-value");
const probabilityMeter = document.querySelector("#probability-meter");
const resultSymbol = document.querySelector(".result-symbol");

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/`, { method: "GET" });
    if (!response.ok) throw new Error("API unavailable");
    apiStatus.textContent = "API connected";
    apiPill.classList.add("connected");
    apiPill.classList.remove("offline");
  } catch {
    apiStatus.textContent = "API offline";
    apiPill.classList.add("offline");
    apiPill.classList.remove("connected");
  }
}

function clearResult() {
  errorBox.hidden = true;
  resultTitle.textContent = "Your estimate will appear here.";
  resultSummary.textContent = "Complete the indicators above, then select “Get estimate”.";
  probabilityValue.textContent = "—";
  probabilityMeter.style.width = "0%";
  resultSymbol.textContent = "↗";
  resultSymbol.style.color = "var(--accent)";
}

document.querySelector("#randomize-inputs").addEventListener("click", () => {
  // Generate plausible values within each field's allowed range.
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const min = Number(slider.min);
    const max = Number(slider.max);
    const step = Number(slider.step) || 1;
    const steps = Math.floor((max - min) / step);
    slider.value = String(min + Math.floor(Math.random() * (steps + 1)) * step);
    updateRange(slider);
  });
  document.querySelectorAll('input[type="checkbox"]').forEach((input) => {
    input.checked = Math.random() >= 0.5;
  });
  const sexOptions = [...document.querySelectorAll('input[type="radio"][name="Sex"]')];
  sexOptions.forEach((input) => { input.checked = Math.random() >= 0.5 ? input.value === "1" : input.value === "0"; });
  clearResult();
});

document.querySelector("#reset-form").addEventListener("click", () => {
  form.reset();
  document.querySelectorAll('input[type="range"]').forEach(updateRange);
  clearResult();
});

function updateRange(slider) {
  const output = slider.parentElement.querySelector(".range-value");
  if (!output) return;
  const labels = Object.fromEntries((slider.dataset.labels || "").split("; ").filter(Boolean).map((entry) => {
    const index = entry.indexOf(":");
    return [entry.slice(0, index), entry.slice(index + 1)];
  }));
  let value = labels[slider.value] || slider.value;
  if ((slider.name === "MentHlth" || slider.name === "PhysHlth") && !labels[slider.value]) value += Number(slider.value) === 1 ? " day" : " days";
  output.textContent = value;
}

document.querySelectorAll('input[type="range"]').forEach((slider) => {
  slider.addEventListener("input", () => updateRange(slider));
  updateRange(slider);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;

  const features = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    if (element.type === "checkbox") features[element.name] = element.checked ? 1 : 0;
    else if (element.type === "radio" && element.checked) features[element.name] = Number(element.value);
    else if (element.type === "range" || element.tagName === "SELECT") {
      // UI presents General Health from Poor (1) to Excellent (5),
      // while the CDC model encoding is Excellent (1) to Poor (5).
      features[element.name] = element.name === "GenHlth" ? 6 - Number(element.value) : Number(element.value);
    }
  }

  submitButton.disabled = true;
  submitButton.innerHTML = 'Calculating… <span>↻</span>';

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features })
    });
    let payload;
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : "The API rejected the request. Check the backend logs.";
      throw new Error(detail);
    }

    const probability = Number(payload.probability);
    if (!Number.isFinite(probability)) throw new Error("The API response did not include a valid probability.");
    const pct = (probability * 100).toFixed(2);
    probabilityValue.textContent = `${pct}%`;
    probabilityMeter.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;

    if (Number(payload.prediction) === 1) {
      resultTitle.textContent = "Model flagged class 1.";
      resultSummary.textContent = `The model estimates ${pct}% probability for class 1, at or above its configured decision threshold. This is not a diagnosis.`;
      resultSymbol.textContent = "!";
      resultSymbol.style.color = "var(--accent-dark)";
    } else {
      resultTitle.textContent = "Model predicted class 0.";
      resultSummary.textContent = `The model estimates ${pct}% probability for class 1, below its configured threshold. This does not rule out diabetes.`;
      resultSymbol.textContent = "✓";
      resultSymbol.style.color = "var(--success)";
    }
    document.querySelector("#result-panel").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    errorBox.textContent = error instanceof TypeError
      ? "Can't reach FastAPI. Start the backend and check its CORS settings for this page's origin."
      : (error.message || "Something went wrong. Please try again.");
    errorBox.hidden = false;
    await checkApi();
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Get estimate <span>↗</span>';
  }
});

checkApi();
