# Model Card: Census Income Classification Model

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project implements a supervised binary classification model to predict whether an individual earns more than $50K per year based on U.S. Census demographic and employment data. The model was trained using a `RandomForestClassifier` from scikit-learn.

The dataset used is the publicly available Census Income dataset (Adult dataset), which contains demographic attributes such as age, education, occupation, work class, marital status, and hours worked per week.

The model is trained using an 80/20 train-test split.

## Intended Use
The intended use of this model is educational. It demonstrates:

- End-to-end machine learning pipeline construction

- Data preprocessing with one-hot encoding

- Model training and evaluation

- Slice-based fairness/performance evaluation

- Deployment using FastAPI

- CI/CD integration using GitHub Actions

This model is not intended for real-world decision-making or financial qualification systems.

## Training Data
The model was trained on the Census Income dataset (census.csv) provided in the project starter repository.

The dataset includes both categorical and continuous features. Categorical variables are one-hot encoded, and the label (`salary`) is binarized to:

- `>50K`

- `<=50K`

The dataset was split into:

- 80% training data

- 20% testing data

## Evaluation Data
Evaluation was performed on the 20% held-out test set.

Additionally, slice-based evaluation was performed across all categorical features to analyze performance consistency across demographic subgroups.

The slice evaluation results are stored in `slice_output.txt`.

## Metrics
The following evaluation metrics were computed on the test set:

- Precision: 0.7419

- Recall: 0.6384

- F1 Score: 0.6863

Precision indicates that when the model predicts income greater than $50K, it is correct approximately 74% of the time.

Recall indicates that the model identifies approximately 64% of all individuals who actually earn more than $50K.

The F1 score balances precision and recall.

## Ethical Considerations
The dataset contains sensitive demographic attributes such as race, sex, and marital status. Using such attributes in predictive systems can introduce bias and fairness concerns.

This model is for academic purposes only and should not be used in any real hiring, lending, or income qualification decisions.

Any production use would require:

- Fairness auditing

- Bias mitigation strategies

- Legal and ethical review

## Caveats and Recommendations
- The model is trained on historical census data, which may contain societal biases.

- The RandomForest model was used with default hyperparameters; further tuning may improve performance.

- The model does not include fairness-aware learning techniques.

- Performance may degrade if applied to data distributions different from the training data.

Future improvements could include:

- Hyperparameter tuning

- Cross-validation

- Feature importance analysis

- Fairness metric reporting

- Bias mitigation techniques