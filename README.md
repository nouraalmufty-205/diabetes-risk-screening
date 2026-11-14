# 🩺 Diabetes Risk Screening Tool

> Many people have diabetes or prediabetes and don't know it. This tool turns a short health questionnaire into a risk level that tells people when it is worth seeing a doctor. **A screening aid, not a medical diagnosis.**

**🔗 Live app:** [ADD YOUR STREAMLIT / HUGGING FACE LINK]

![App screenshot](ADD_SCREENSHOT_OR_GIF.png)

## Results

All numbers come from `python train.py` (test set = 20% of the data, never used for tuning).

| Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|
| Baseline (majority class) | [X] | [X] | [X] | [X] |
| Logistic Regression | [X] | [X] | [X] | [X] |
| Random Forest | [X] | [X] | [X] | [X] |
| Gradient Boosting | [X] | [X] | [X] | [X] |
| Best model, short form ([N] questions) | [X] | [X] | [X] | [X] |

Copy these from `outputs/results.csv` after you run the training. Never write a number you did not measure.

**Key finding:** [one sentence, e.g. "the questionnaire can be cut to N questions while keeping recall at X%"]

## What it does
- Cleans and explores ~250K CDC survey responses (`Diabetes_binary` target, 21 health and lifestyle features)
- Compares a baseline with Logistic Regression, Random Forest and Gradient Boosting (5-fold CV + randomized search, leakage-free scikit-learn pipelines)
- Tunes the decision threshold on validation data to reach a recall goal, because for screening a missed case costs more than a false alarm
- Finds the smallest set of questions that keeps most of the performance and serves it in a Streamlit app that shows the answers that raise the score

## Run it
```bash
pip install -r requirements.txt
# put the CSV in data/ (or let the code download it from UCI)
python train.py --k 7              # trains, tunes, evaluates, saves models/ and outputs/
streamlit run app.py               # local app
jupyter notebook diabetes_risk_screening.ipynb
```

## Files
| File | Purpose |
|---|---|
| `diabetes_risk_screening.ipynb` | Full analysis: framing, EDA, modeling, evaluation, business insight |
| `train.py` | Reproducible training script, writes `models/*.joblib` and `outputs/*` |
| `utils.py` | Schema, preprocessing pipeline, threshold helper |
| `app.py` | Streamlit web app |

## Limitations
Trained on US self-reported survey data (2015); prediabetes and diabetes are grouped together; the risk score is a model estimate, not a clinical probability; the threshold accepts some false alarms to catch more true cases.

## Data
CDC Diabetes Health Indicators, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/891

Built by [names] during the NextGen Institute ML internship.
