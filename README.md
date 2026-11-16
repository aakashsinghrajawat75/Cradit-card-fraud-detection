# Credit Card Fraud Detection

**Author:** Aakash (GitHub: [@aakashrajawat75](https://github.com/aakashrajawat75))

A machine-learning project for exploring anonymized credit-card transactions,
comparing fraud classifiers, and reviewing transactions flagged by a trained
model. It includes a Jupyter notebook for analysis and a Streamlit dashboard
for interactive exploration and scoring.

> **Project status:** The Streamlit app, notebook, requirements, and model
> artifact files are present. The app's model path has been set up for this
> repository layout. The data-download cell in the notebook previously failed
> because this Python environment could not verify OpenML's TLS certificate.
> No raw or cleaned CSV files are currently included. The model bundle was
> loaded successfully and a sample inference plus Streamlit smoke test passed;
> this is a basic integration check, not an independent model-performance
> validation.

## Features

- **Overview:** transaction and feature counts, fraud distribution, amount
  histogram, and a sample of uploaded rows.
- **Explore data:** numeric feature distributions, optional transaction-time
  trends, and feature correlations.
- **Model evaluation:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and a
  confusion matrix for a labeled input CSV.
- **Score transactions:** predicted fraud probabilities, review flags, and a
  downloadable CSV of the scored rows.
- **Notebook workflow:** dataset exploration, preprocessing, model comparison,
  threshold selection, and model export.

Fraud scores are decision-support signals, not definitive findings. Review
model performance and operational consequences before using predictions.

## Repository layout

```text
Credit-Card-Fraud-Detection/
├── app/
│   └── app.py
├── data/
│   ├── credit_card_fraud_raw.csv       # not yet generated
│   └── credit_card_fraud_cleaned.csv   # not yet generated
├── fraud_model/
│   ├── fraud_detection_model.pkl
│   ├── threshold.pkl
│   ├── feature_names.pkl
│   ├── imputer.pkl                     # additional notebook artifact
│   └── scaler.pkl                      # additional notebook artifact
├── CraditCardFraudDetection.ipynb
├── requirements.txt
└── README.md
```

The notebook uses OpenML dataset ID `1597`. Its dataset-loading cell calls
`fetch_openml(data_id=1597, as_frame=True)`. In the current environment, that
call encountered an SSL certificate verification error. The `data/` CSV files
will need to be created after TLS certificate verification is configured or a
trusted source CSV is provided.

The notebook's saved exploration outputs report 284,807 rows, 492 fraud cases,
no missing values, and 9,144 exact duplicate rows. These are notebook outputs,
not a fresh verification of the currently available source data. The notebook
counts duplicates; confirm the intended duplicate policy before treating a
deduplicated export as canonical.

## Requirements

- Python 3.10 or later is recommended.
- Dependencies are listed in `requirements.txt`.
- Model files are serialized with joblib/pickle. Keep the runtime compatible
  with the library versions used to create them.

## Run the Streamlit app

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

The app loads model artifacts from the root `fraud_model/` directory. The
dashboard itself starts without an uploaded transaction CSV; upload a CSV
through the sidebar to use the analysis pages.

## Input CSV format

For model scoring, include the model's feature columns in the saved order.
For this project these are expected to be the anonymized features `V1` through
`V28` and `Amount`, as recorded in `feature_names.pkl`. The optional `Class`
column is the target label (`0` = genuine, `1` = fraud) and enables evaluation.
The optional `Time` column enables a time-based chart when present.

CSV uploads are read into memory by the Streamlit process. Do not upload
cardholder data or other sensitive information to a public or unapproved
deployment.

## Model files

- `fraud_detection_model.pkl` — trained estimator/pipeline; must implement
  `predict_proba()`.
- `threshold.pkl` — probability cutoff used to flag fraud.
- `feature_names.pkl` — ordered feature names expected by the model.
- `imputer.pkl` and `scaler.pkl` — additional preprocessing artifacts saved by
  the notebook; the current dashboard uses the trained model bundle and expects
  its model to handle its own preprocessing.

Only load trusted model artifacts. Pickle/joblib files can execute code when
loaded. The files currently present should be validated against a clean app
startup and a known-good data sample before relying on them.

## Notebook workflow

Open `CraditCardFraudDetection.ipynb` in VS Code or Jupyter and run the cells in
order:

1. Load and inspect OpenML dataset 1597.
2. Review missing values, class imbalance, and duplicates.
3. Split the labeled data into training and test partitions.
4. Fit preprocessing and SMOTE within the training workflow.
5. Compare classifiers and tune the Random Forest using cross-validation.
6. Choose a probability threshold on validation data.
7. Export the model, threshold, and ordered feature names into `fraud_model/`.

The notebook's model-export cell now rebuilds the final model from the fitted
hyperparameter search and the training split, rather than assuming
`best_model` is still present in the active kernel.

## Git and data handling

The model files are included in this project folder so they can be distributed
with the application. Review file sizes and Git hosting limits before pushing
them. Keep raw/cleaned CSVs and private transaction data out of public Git
repositories unless you have confirmed the data license and have approval to
publish them.

## Deployment

For Streamlit Community Cloud or another hosting service:

1. Push the project to a Git repository.
2. Configure the app entry point as `app/app.py`.
3. Ensure the `fraud_model/` files are included or provisioned securely.
4. Install dependencies from `requirements.txt`.
5. Test the deployed app with an approved, non-sensitive CSV.

## Author

**Aakash** — [@aakashrajawat75](https://github.com/aakashrajawat75)
