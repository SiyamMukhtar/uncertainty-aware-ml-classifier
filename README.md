# Uncertainty-Aware Machine Learning Classifier

## Overview

This project explores uncertainty-aware classification using machine learning. Instead of relying only on predicted labels, the system evaluates prediction confidence and identifies low-confidence predictions for potential human review.

The project demonstrates a basic selective prediction workflow using the Iris dataset and Logistic Regression.

## Objectives

- Train a machine learning classification model.
- Measure predictive performance using standard evaluation metrics.
- Estimate prediction confidence using class probabilities.
- Apply a confidence threshold to flag uncertain predictions.
- Evaluate coverage, review rate, and accuracy on accepted predictions.
- Examine confidence calibration and the Brier score.

## Dataset

The project uses the Iris dataset provided by Scikit-learn.

The dataset contains measurements of iris flowers across three classes.

## Methodology

1. Load the Iris dataset.
2. Split the data into training, validation, and test sets.
3. Build a pipeline with StandardScaler and Logistic Regression.
4. Evaluate predictions using accuracy, macro precision, macro recall, and macro F1-score.
5. Select a confidence threshold using the validation set.
6. Flag predictions below the threshold for review.
7. Evaluate selective prediction performance on the test set.
8. Visualize confidence distribution and calibration.

## Results

Results from the current experiment:

| Metric | Result |
|---|---:|
| Training samples | 90 |
| Validation samples | 30 |
| Test samples | 30 |
| Baseline accuracy | 93.3% |
| Macro precision | 94.4% |
| Macro recall | 93.3% |
| Macro F1-score | 93.3% |
| Confidence threshold | 0.737 |
| Coverage | 80% |
| Review rate | 20% |
| Accepted prediction accuracy | 100% |
| Binary correctness Brier score | 0.0517 |

These results are based on one train-validation-test split. The test set is small, so the metrics should be interpreted cautiously. The accepted prediction accuracy does not guarantee perfect accuracy on new data.

## Project Structure

```text
uncertainty_ml_project/
├── uncertainty_ml.ipynb
├── requirements.txt
└── results/
    ├── evaluation_metrics.csv
    ├── test_predictions.csv
    └── figures/
        ├── confidence_distribution.png
        └── confidence_calibration.png
```

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `uncertainty_ml.ipynb` and run the cells.

## Limitations

- The Iris dataset is small and intended for educational use.
- Results may change with a different data split.
- Confidence is not automatically equivalent to the probability that a prediction is correct.
- Calibration estimates from a small test set are noisy.
- The current project is a learning prototype, not a validated high-stakes decision system.

## Future Improvements

- Evaluate performance across multiple stratified splits or cross-validation folds.
- Compare Logistic Regression with additional classifiers.
- Investigate probability calibration methods.
- Study uncertainty estimation on more challenging datasets.
- Explore human-review strategies for uncertain predictions.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook