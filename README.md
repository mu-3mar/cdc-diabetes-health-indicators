# HealthLens — CDC Diabetes Health Indicators

**Author:** Muhammad Ammar  
**Project type:** Machine Learning classification project with a web interface and REST API  
**Status:** Local prediction flow reported working by the project owner; deployment is the next phase.

> **Medical disclaimer:** HealthLens is an educational machine-learning demonstration, not a medical device. Its output is not a diagnosis, does not rule diabetes in or out, and must not be used to make treatment or medical decisions. Consult a qualified healthcare professional for medical advice.

---

## Table of contents

1. [Project overview](#project-overview)
2. [Goals](#goals)
3. [Current architecture](#current-architecture)
4. [Dataset and prediction target](#dataset-and-prediction-target)
5. [Input features](#input-features)
6. [Exploratory analysis and preparation](#exploratory-analysis-and-preparation)
7. [Modeling and evaluation](#modeling-and-evaluation)
8. [Final candidate model and reported evaluation](#final-candidate-model-and-reported-evaluation)
9. [Preprocessing and feature consistency](#preprocessing-and-feature-consistency)
10. [Web interface](#web-interface)
11. [FastAPI API contract](#fastapi-api-contract)
12. [Run locally](#run-locally)
13. [Configuration and CORS](#configuration-and-cors)
14. [Randomized inputs](#randomized-inputs)
15. [Troubleshooting](#troubleshooting)
16. [Deployment plan](#deployment-plan)
17. [Limitations and responsible use](#limitations-and-responsible-use)
18. [Potential next steps](#potential-next-steps)
19. [Project structure](#project-structure)
20. [Reproducibility notes and TODOs](#reproducibility-notes-and-todos)

---

## Project overview

HealthLens is a web-based demonstration for a binary classification model trained using the CDC Diabetes Health Indicators data. A user supplies health, demographic, lifestyle, and healthcare-access indicators through a browser form. The frontend sends the values to a FastAPI endpoint, which returns a model prediction and a probability for class `1`.

The project brings together the main stages of a practical ML application:

- data preparation and feature engineering;
- model comparison and validation;
- classification with an imbalanced target;
- a trained model and its preprocessing logic;
- an HTTP inference API;
- a browser-based user interface;
- a planned deployment to a publicly accessible host.

The frontend is branded **HealthLens** and uses a warm cream and orange visual identity. The interface includes an animated logo, grouped form fields, binary switches, sliders, a random-input button, API status, and a prediction result panel.

### What the application does

1. Collects 21 input indicators.
2. Converts the selected values into the numeric representation expected by the model.
3. Sends a JSON request to the backend.
4. Receives a class prediction and a probability for class `1`.
5. Displays the probability and a threshold-based result message.

### What it does not do

- It is not a clinical diagnostic tool.
- It does not replace laboratory testing or professional medical assessment.
- A generated/randomized example is not a real person's health profile.
- The probability is a model output and should not automatically be interpreted as a calibrated real-world personal risk.

---

## Goals

The project's practical goals are to:

- build and compare classification baselines for `Diabetes_binary`;
- account for class imbalance rather than relying on accuracy alone;
- keep preprocessing within model pipelines to reduce data leakage during cross-validation;
- evaluate candidate models using precision, recall, F1-score, and Average Precision;
- expose inference through a simple API;
- provide a straightforward web interface for experimenting with inputs;
- document local setup and prepare for deployment.

---

## Current architecture

```text
User's browser
    |
    |  HTML / CSS / JavaScript
    v
HealthLens frontend
    |
    |  GET  http://127.0.0.1:8000/
    |  POST http://127.0.0.1:8000/predict
    v
FastAPI backend
    |
    |  validate / map features
    |  preprocessing + trained estimator
    v
Prediction and probability
    |
    v
Result displayed in browser
```

In local development, the frontend currently defines the API base URL as `http://127.0.0.1:8000`. That address refers to the computer running the browser, so it must be changed to the deployed API's public URL when the backend is hosted remotely.

**Current status:** The project owner reports that the website and prediction/estimate flow are working locally. The README documents the available project information and notebook results; it does not claim that a public deployment has already been completed.

---

## Dataset and prediction target

The project uses the **CDC Diabetes Health Indicators** dataset. The prepared modeling data in the modeling notebook is loaded from:

```text
../data/processed/prepared/
```

The notebook expects these files:

```text
X_train.csv
X_test.csv
y_train.csv
y_test.csv
```

The target used in the modeling notebook is **`Diabetes_binary`**, with the positive class defined as `1`.

The prepared train/test split is used for model development and evaluation. The modeling notebook's narrative describes the target as imbalanced, with approximately 84.7% of examples in class `0` and 15.3% in class `1` in the data discussed there. Exact row counts and split proportions should be confirmed from the current prepared CSV files before being reported as final dataset statistics.

### Dataset provenance

The dataset is associated with CDC's Behavioral Risk Factor Surveillance System (BRFSS) health-indicator data. If redistributing the data, check the original source's terms, citation requirements, and current documentation. This repository README does not include the dataset itself.

**Source link to add/verify before publication:** CDC/UCI dataset page used to obtain the specific project copy. Add the exact URL and access date here once confirmed from the original download/source notebook.

---

## Input features

The current frontend sends these 21 features. The names and encodings must remain consistent with the model's training columns and preprocessing pipeline.

| Feature | Frontend control | Expected values / meaning |
|---|---|---|
| `HighBP` | Switch | `0` = No, `1` = Yes |
| `HighChol` | Switch | `0` = No, `1` = Yes |
| `CholCheck` | Switch | `0` = No, `1` = Yes |
| `BMI` | Slider | Numeric BMI; UI range currently 12–60 in steps of 0.5 |
| `Smoker` | Switch | `0` = No, `1` = Yes |
| `Stroke` | Switch | `0` = No, `1` = Yes |
| `HeartDiseaseorAttack` | Switch | `0` = No, `1` = Yes |
| `PhysActivity` | Switch | `0` = No, `1` = Yes |
| `Fruits` | Switch | `0` = No, `1` = Yes |
| `Veggies` | Switch | `0` = No, `1` = Yes |
| `HvyAlcoholConsump` | Switch | `0` = No, `1` = Yes |
| `AnyHealthcare` | Switch | `0` = No, `1` = Yes |
| `NoDocbcCost` | Switch | `0` = No, `1` = Yes |
| `GenHlth` | Slider | Ordinal general-health response; model encoding: `1` = Excellent, `2` = Very good, `3` = Good, `4` = Fair, `5` = Poor |
| `MentHlth` | Slider | Number of days of poor mental health in the past 30 days, `0`–`30` |
| `PhysHlth` | Slider | Number of days of poor physical health in the past 30 days, `0`–`30` |
| `DiffWalk` | Switch | `0` = No, `1` = Yes |
| `Sex` | Mutually exclusive choice | Current UI uses `0` = Female and `1` = Male; confirm against the exact training data/code before deployment |
| `Age` | Slider | Encoded age group, `1`–`13` |
| `Education` | Slider | Ordinal education category, `1`–`6` |
| `Income` | Slider | Ordinal income category, `1`–`8` |

### Ordinal labels used by the interface

- **Age:** `1` = 18–24, `2` = 25–29, `3` = 30–34, `4` = 35–39, `5` = 40–44, `6` = 45–49, `7` = 50–54, `8` = 55–59, `9` = 60–64, `10` = 65–69, `11` = 70–74, `12` = 75–79, `13` = 80+ years.
- **General health:** the form presents the human-readable scale from Poor to Excellent. The underlying model encoding remains `1` = Excellent through `5` = Poor. The JavaScript converts the displayed slider value before sending it to the API.
- **Education:** `1` = Never attended/kindergarten; `2` = Grades 1–8; `3` = Grades 9–11; `4` = Grade 12/GED; `5` = College 1–3 years; `6` = College 4+ years.
- **Income:** eight ordered income categories, currently displayed from under `$10,000` through `$75,000+`. Confirm the exact bracket labels against the source dataset documentation before public release.

**Important:** The API/model is the source of truth for accepted feature names, order, types, and encodings. Before deployment, confirm the backend validates all 21 keys and applies the same preprocessing object/pipeline used during training.

---

## Exploratory analysis and preparation

The modeling notebook identifies the prepared-data output of `06_feature_engineering_preparation.ipynb` as the input to the modeling stage. The following modeling inputs are explicitly documented:

- `X_train` and `X_test` are read from prepared CSV files.
- `y_train` and `y_test` are read from the corresponding target CSV files, selecting `Diabetes_binary`.
- Feature groups are separated into numeric, binary, and ordinal variables.
- The target class `1` is treated as the positive class for evaluation.

The notebook's documented feature groups are:

- **Numeric:** `BMI`, `MentHlth`, `PhysHlth`.
- **Ordinal:** `GenHlth`, `Education`, `Income`, `Age`.
- **Binary:** remaining columns identified from the training data by checking for two unique values.

The notebook code shown for this stage uses `ColumnTransformer` and `StandardScaler` for selected numeric features. Other features are passed through unchanged. For models that are sensitive to feature scale, scaling is included inside the pipeline. Tree-based models are evaluated without requiring numeric scaling.

The full EDA and feature-engineering details should be expanded from the corresponding earlier notebooks before the README is considered a complete record of the entire data workflow. This README does not infer missing EDA findings.

---

## Modeling and evaluation

The organized modeling notebook documents the following baseline model configurations:

1. Logistic Regression.
2. Logistic Regression with `class_weight="balanced"`.
3. K-Nearest Neighbors (KNN).
4. Decision Tree.
5. Random Forest.

The experiments then investigate class-imbalance strategies, including:

- `class_weight="balanced"`;
- Random Oversampling;
- SMOTE;
- Random Undersampling.

Resampling is integrated into an imbalanced-learn pipeline so it is applied within training folds, rather than before cross-validation. This is intended to prevent validation-fold information from influencing the resampling step.

### Validation approach

The notebook uses five-fold `StratifiedKFold` cross-validation, with shuffling and `random_state=42` documented in the modeling workflow. Stratification helps preserve approximately the same class proportions in each fold.

Metrics used include:

- **Accuracy:** overall fraction of correct predictions.
- **Precision for class 1:** fraction of predicted positive cases that are positive in the labels.
- **Recall for class 1:** fraction of actual positive cases identified by the model.
- **F1-score for class 1:** harmonic mean of precision and recall.
- **Average Precision:** summary of precision-recall ranking performance.
- **Confusion matrix:** counts of true negatives, false positives, false negatives, and true positives.

Because class `1` is the minority class, Accuracy alone is not sufficient to characterize the model's performance.

### Methodological caveat

The organized notebook explicitly notes that the test set was used for initial exploratory model comparisons. It also describes later validation-based threshold selection and a final evaluation on the same held-out test split. As a result, the reported test metrics are useful for documenting the current experiment, but they should **not** be described as a completely untouched, independent estimate of generalization. For a stronger final report, preserve a genuinely untouched test set or use a separate external validation set for the final confirmation.

---

## Final candidate model and reported evaluation

The organized modeling notebook identifies **Balanced Logistic Regression** as the leading candidate based on the cross-validated F1-score and recall trade-off. The notebook tests classification thresholds from `0.3` to `0.7` on validation data. Among the tested thresholds, `0.6` achieved the highest validation F1-score.

### Validation threshold results

For the selected threshold `0.6`, the notebook reports:

- Precision: `0.371`
- Recall: `0.633`
- F1-score: `0.468`

This threshold is a validation-selected candidate, not a universal clinical threshold. Increasing or lowering it changes the balance between false positives and false negatives.

### Reported held-out test results at threshold 0.6

The notebook's final evaluation section reports the following results for the selected Balanced Logistic Regression configuration:

| Metric | Reported value |
|---|---:|
| Accuracy | `0.775` |
| Precision for class 1 | `0.366` |
| Recall for class 1 | `0.642` |
| F1-score for class 1 | `0.466` |
| Average Precision | `0.412` |

Reported confusion matrix, in scikit-learn's default class order `[[TN, FP], [FN, TP]]`:

```text
[[31058, 7818],
 [ 2512, 4507]]
```

Interpretation from these counts:

- 4,507 positive cases were correctly identified.
- 2,512 positive cases were missed (false negatives).
- 7,818 negative cases were flagged as positive (false positives).

These values are transcribed from the organized notebook. Re-run the notebook against the exact current data and final saved estimator before publishing them as definitive metrics. The notebook itself notes that the test set had already been used for earlier comparisons.

---

## Preprocessing and feature consistency

The modeling notebook uses a scikit-learn preprocessing pipeline that standardizes `BMI`, `MentHlth`, and `PhysHlth` with `StandardScaler`, while passing the remaining columns through unchanged. The scaling is fitted within the pipeline during training/cross-validation.

For inference, the production API must use the same preprocessing parameters fitted during training. Do not independently fit a new scaler on incoming user input, and do not manually reorder or rename features without matching the model's training schema.

Before release, verify these items in the backend code:

1. The trained estimator and scaler/pipeline are loaded from the intended artifacts.
2. The exact 21 feature names match the training columns.
3. The feature order passed to the estimator matches training order.
4. `GenHlth` is converted to the original CDC model encoding correctly.
5. The `Sex` encoding matches the source training data.
6. The API's returned probability corresponds to class `1`.
7. The decision threshold in the backend matches the documented threshold of `0.6`, if that threshold is indeed configured in the deployed estimator/API.

The frontend currently expects the response to include a `prediction` field and a numeric `probability` field. The precise backend implementation and artifact filenames were not available in the frontend bundle used to prepare this README, so those details should be confirmed from `src/main.py` and the model-loading code.

---

## Web interface

The current frontend bundle contains:

- `index.html` — page structure and input form;
- `styles.css` — layout, responsive styles, colors, switches, sliders, result card, and footer;
- `app.js` — API status, input mapping, randomization, request handling, and result rendering;
- `logo-animated.html` — animated brand logo embedded in the header.

### User interface features

- Light-only, warm cream background and orange-red accent.
- HealthLens brand header and animated logo.
- API status indicator.
- A brief educational-use disclaimer.
- Three input sections: About you, Medical history, and Lifestyle & care.
- Sliders for numeric and ordinal fields.
- Switches for binary indicators.
- Mutually exclusive Female/Male options.
- Randomize inputs button.
- Reset answers button.
- Get estimate button.
- Result panel with class prediction, probability, progress meter, and disclaimer.
- Footer credit: **Designed & built by Muhammad Ammar**.

### Frontend input mapping

The JavaScript builds an object called `features` and submits it as JSON inside a wrapper object. Checkbox values are mapped to `1` when checked and `0` when unchecked. The selected Sex radio value is sent as a number. Slider values are sent as numbers. For `GenHlth`, the UI displays a reversed, user-friendly scale and converts it to the dataset encoding using `6 - displayedValue`.

### API URL

The frontend currently has this constant in `app.js`:

```javascript
const API_BASE = "http://127.0.0.1:8000";
```

For local development, this should point to the local FastAPI server. For a hosted website, replace it with the public HTTPS URL of the deployed API.

---

## FastAPI API contract

The frontend makes two requests.

### `GET /`

Used by the frontend to check whether the API is reachable. Any successful HTTP response is treated as connected by the current frontend.

### `POST /predict`

The frontend sends JSON in this shape:

```json
{
  "features": {
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 1,
    "BMI": 27,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Sex": 0,
    "Age": 9,
    "Education": 6,
    "Income": 5
  }
}
```

This is an illustrative example only, not a medically meaningful profile. Values and feature names must match the backend schema. The frontend expects a successful response to contain at least:

```json
{
  "prediction": 0,
  "probability": 0.18
}
```

Here, `prediction` is expected to be a class label, and `probability` is expected to be a number between `0` and `1` representing the model's probability for class `1`. Confirm this interpretation against the actual `src/main.py` implementation before deploying. The frontend converts the probability to a percentage for display.

### Local API development command

From the project root, the command used by the project owner is:

```bash
uvicorn src.main:app --reload
```

The API is then available locally at `http://127.0.0.1:8000`, and FastAPI's interactive documentation is normally available at `http://127.0.0.1:8000/docs` if the app has the default documentation enabled.

**Backend implementation TODO:** document the actual Pydantic request model, response schema, model artifact path, class-probability extraction method, exception behavior, and configured decision threshold after inspecting the backend source.

---

## Run locally

### Prerequisites

- Python environment with the project's required packages installed.
- The trained model and preprocessing artifact(s) at the paths expected by the backend.
- The project source containing `src/main.py`.
- A modern browser.
- VS Code Live Server or another local static HTTP server for the frontend.

The project owner has used a Conda environment named `cdc`; environment names are local preferences and may differ on another machine.

### 1. Start the backend

Activate the environment and move to the project root, then run:

```bash
uvicorn src.main:app --reload
```

Keep this terminal open. Check the startup logs and visit `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/docs` to confirm the API starts.

### 2. Start the frontend

Open the frontend folder in VS Code and run `index.html` using Live Server. The exact local origin depends on the Live Server settings, commonly:

```text
http://127.0.0.1:5500
```

Do not assume the port is always `5500`; use the URL shown by the editor.

### 3. Test the prediction flow

1. Confirm the header says the API is connected.
2. Enter values or click **Randomize inputs**.
3. Click **Get estimate**.
4. Confirm the result panel displays a prediction and probability.
5. Check the browser console and Uvicorn terminal if a request fails.

### 4. Stop local services

Press `Ctrl+C` in the terminal running Uvicorn. Stop Live Server from VS Code when finished.

---

## Configuration and CORS

When the frontend and API run on different origins, the browser may send a CORS preflight request using the `OPTIONS` method. If the backend rejects the preflight, the browser cannot send the actual prediction request.

The FastAPI app should configure `CORSMiddleware` with the specific frontend origins used in development and deployment. Avoid relying on `allow_origins=["*"]` together with credentials in a production application. Add the exact Live Server origin during local development and the deployed frontend origin in production.

Example configuration (adjust origins to the actual URLs):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        # Add the real deployed frontend origin here.
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
```

If `index.html` is opened directly as a `file://` URL, the browser may send an origin of `null`. Prefer a local HTTP server such as Live Server instead of opening the file directly.

---

## Randomized inputs

The **Randomize inputs** button chooses random values within the current UI controls' allowed ranges. It randomizes slider values and binary switches and selects one of the mutually exclusive Sex options. It then clears the previous result so that the user can press **Get estimate** to request a new prediction.

Randomized values are for interface testing only. They are not sampled from the real population distribution, are not guaranteed to represent realistic combinations, and should not be used to report model performance.

---

## Troubleshooting

### API indicator says offline

- Make sure Uvicorn is still running.
- Confirm `API_BASE` in `app.js` matches the backend address.
- Open the API root URL in the browser.
- Check whether the backend crashed while loading the model.

### `OPTIONS /predict` returns `400 Bad Request`

- Check FastAPI CORS middleware and allowed origins.
- Run the frontend through Live Server and use its exact origin in `allow_origins`.
- Do not guess the origin; read it from the browser address bar.

### Browser reports a CORS error

- Confirm the API permits the frontend origin, `POST`, and the `Content-Type` header.
- Confirm the frontend uses the public API URL in deployment.
- Inspect the backend logs for the preflight request.

### API responds with a validation error

- Check all 21 feature keys and their types.
- Ensure numeric values are sent as numbers rather than labels such as `"Good"`.
- Verify the `GenHlth` conversion and Sex encoding.
- Compare the incoming columns to the model's training feature names.

### Prediction succeeds but result looks wrong

- Confirm the API returns probability for class `1` rather than class `0`.
- Confirm the same preprocessing pipeline used for training is loaded at inference time.
- Confirm the configured decision threshold and response interpretation.
- Remember that the model is experimental and is not a clinical tool.

### Animated logo does not display

- Keep `logo-animated.html` in the same frontend directory as `index.html`, `styles.css`, and `app.js`.
- Serve the page over Live Server rather than opening it directly from the file system.
- Check the browser console for iframe or file path errors.

---

## Deployment plan

Deployment has not been documented as completed yet. The intended plan is to publish the frontend and make the API reachable over HTTPS.

### Option A: Vercel

Vercel is convenient for static frontend hosting and also supports Python Functions, but the suitability of its function runtime depends on the project's actual dependency bundle, model artifact size, startup time, memory needs, execution limits, and runtime behavior. Verify the current limits for the selected plan before choosing it for the inference backend.

Suggested separation:

- Deploy the static frontend from the frontend directory.
- Deploy the API as a Python function only if the model and dependencies fit the current Vercel runtime constraints.
- Set `API_BASE` to the public API URL.
- Configure CORS to allow the exact public frontend origin.
- Test `/` and `/predict` after deployment.

Official references to re-check before deploying:

- https://vercel.com/docs/plans/hobby
- https://vercel.com/kb/guide/ship-a-fastapi-app-on-vercel

### Option B: Oracle Cloud Infrastructure (OCI)

An Oracle Cloud VM can provide a more conventional always-running Linux environment where FastAPI can run as a service. It is also a possible future home for Dockerized API and frontend components. Free resources, region availability, eligibility, and inactivity policies may change; check the current official terms before relying on the free tier.

Possible layout on a VM:

```text
Internet
   |
   v
HTTPS reverse proxy (e.g. Nginx/Caddy)
   |                         \
   v                          v
Static frontend             FastAPI service
                              |
                              v
                        Model + preprocessing
```

Before exposing a VM publicly, configure firewall/security rules, HTTPS, process supervision, updates, logs, and secrets. Do not expose an unsecured development server as a production service.

Official reference:

- https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm

### Deployment checklist

- [ ] Commit the final code and README to GitHub.
- [ ] Confirm model and preprocessing artifacts are included or safely downloaded at startup.
- [ ] Confirm model artifact sizes and Python dependency sizes.
- [ ] Configure environment variables instead of hardcoding environment-specific URLs.
- [ ] Deploy the API and confirm its health endpoint.
- [ ] Deploy the frontend and set its public API URL.
- [ ] Configure CORS for the deployed origin.
- [ ] Verify a real `/predict` request from the public frontend.
- [ ] Check logs and failure responses.
- [ ] Confirm that no private data, credentials, or large raw data files were accidentally committed.
- [ ] Document the live URLs and deployment steps after they exist.

Dockerization can be added after choosing the deployment target. It is not required for a static frontend, and it is optional for some serverless API deployments, but it is useful for packaging a conventional API service on a VM.

---

## Limitations and responsible use

- The target is imbalanced, so model performance depends on the chosen metric and threshold.
- The reported test metrics are not a fully independent estimate because the notebook states the test set was used in earlier exploratory comparisons.
- The model can produce false positives and false negatives; the reported confusion matrix shows both types of errors.
- A model probability is not automatically a calibrated estimate of an individual's actual probability of developing diabetes.
- The training data may not represent all current populations or contexts.
- Randomized UI inputs are not real data and do not demonstrate clinical validity.
- The application is an educational project and must not be used for diagnosis or treatment decisions.
- The backend source and final artifact configuration should be reviewed before public deployment.

---

## Potential next steps

1. Finalize this README with the exact dataset URL, repository tree, package list, and artifact filenames.
2. Verify backend request/response schemas directly from `src/main.py`.
3. Add a dependency file (`requirements.txt` or an environment specification) based on the actual working environment.
4. Add a small automated API smoke test for `/` and `/predict`.
5. Add validation for missing keys, unexpected features, out-of-range values, and non-finite numbers.
6. Add a versioned model artifact and record the training configuration.
7. Re-evaluate using an untouched test set if a stronger final performance claim is needed.
8. Publish the repository and screenshots.
9. Deploy the frontend and backend, then test the public URL.
10. Add monitoring/logging that avoids storing sensitive health inputs unnecessarily.

---

## Project structure

The frontend bundle prepared during this project contains:

```text
frontend/
├── index.html
├── styles.css
├── app.js
└── logo-animated.html
```

The modeling notebook refers to a project structure that includes:

```text
data/processed/prepared/
├── X_train.csv
├── X_test.csv
├── y_train.csv
└── y_test.csv
```

The backend command references:

```text
src/main.py
```

The full repository tree, model artifact filenames, exact dependency file, and data/ notebook names beyond those explicitly referenced above must be filled in from the actual project directory. These paths are not guessed here.

---

## Reproducibility notes and TODOs

The following details must be verified from the full repository before treating this document as a complete, release-ready technical record:

- [ ] Add the repository URL and default branch.
- [ ] Add the exact dataset download/source URL and access date.
- [ ] Add the dataset's exact row/column counts and train/test split details from the current data files.
- [ ] Summarize the EDA and feature-engineering notebooks using their actual outputs.
- [ ] Add the exact saved estimator and preprocessing artifact filenames/paths.
- [ ] Add the actual Python and library versions from the working environment.
- [ ] Add the actual API request/response models and status/error behavior from `src/main.py`.
- [ ] Verify whether threshold `0.6` is enforced in the API or only documented in the modeling notebook.
- [ ] Verify the Sex encoding against the final training data and backend mapping.
- [ ] Confirm the current frontend feature names and order against the model's `feature_names_in_` or equivalent schema.
- [ ] Add screenshots and live deployment URLs after deployment.
- [ ] Re-run the final evaluation workflow if a fully untouched test-set estimate is required.

**Author credit:** Designed and built by **Muhammad Ammar**.
