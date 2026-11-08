# Machine Learning Assignment 2 — Classification Models

## a. Problem statement
Build and evaluate five classification algorithms on the same dataset, save the trained models, and deploy an interactive Streamlit application that evaluates uploaded labelled test data without retraining.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC).
- **Source:** Public dataset distributed with scikit-learn; originally associated with the UCI Machine Learning Repository.
- **Instances:** 569.
- **Input features:** 30 numeric features derived from digitized images of fine needle aspirates of breast masses.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Split:** 75% training and 25% test, stratified, `random_state=42`.
- **Included files:** `dataset.csv` (full dataset) and `test_data.csv` (held-out test subset with target column).

## c. GitHub Repository Link
https://github.com/Aneetarose/ML_Assignment_2_Project.git

## d. Models used and evaluation

All five models are trained on the same training split. The saved pipelines are loaded by the Streamlit app; uploaded test data is not used to retrain them. AUC uses predicted probability for class 1 (benign).

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.986  | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.97   |
| Decision Tree          |     0.9231 | 0.9234 |      0.954  |   0.9222 | 0.9379 | 0.8378 |
| KNN                    |     0.979  | 0.9845 |      0.9677 |   1      | 0.9836 | 0.9555 |
| Naive Bayes            |     0.9371 | 0.9893 |      0.9263 |   0.9778 | 0.9514 | 0.865  |
| Support Vector Machine |     0.979  | 0.9969 |      0.9888 |   0.9778 | 0.9832 | 0.9553 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall balance on this held-out test split; strongest combined ranking of F1, AUC and accuracy. |
| Decision Tree | Easy to interpret, but a single tree may be less stable and can overfit. |
| KNN | Distance-based model; scaling is applied. Performance depends on the choice of neighbours. |
| Naive Bayes | Fast probabilistic baseline; its feature-independence assumption may limit performance. |
| Support Vector Machine | Nonlinear-capable margin classifier; scaling and probability estimates are enabled. |
| **Overall winner** | **Logistic Regression** on this split, ranked by F1, then AUC, then accuracy. Recheck on your own uploaded test CSV. |

**Important:** Metrics above are for the included fixed held-out test split. The ranking may change on other test data. The “winner” is chosen by F1, then AUC, then accuracy, and should be interpreted alongside all metrics.

## Project structure

```text
project-folder/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── dataset.csv
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally / on BITS Virtual Lab

1. Install dependencies: `pip install -r requirements.txt`
2. If needed, regenerate models and CSVs: `python train_models.py`
3. Start app: `streamlit run app.py`
4. Upload `test_data.csv` to evaluate the models.
5. Take your own screenshot of the running app on **BITS Virtual Lab** as required by the assignment.

## Deploy on Streamlit Community Cloud

1. Push all project files to a GitHub repository.
2. Open Streamlit Community Cloud and sign in with GitHub.
3. Choose **New app**, select the repository and branch, and set the main file path to `app.py`.
4. Deploy and wait for the build to finish.
5. Test the live URL by uploading `test_data.csv`.
6. Replace the GitHub and live app placeholders in the final submission PDF with your real links.

## Notes for submission

- Do not submit placeholder links. Add your actual GitHub repository URL and deployed Streamlit URL.
- Include the screenshot taken from your own BITS Virtual Lab session.
- Include this README content in the single PDF submission, in the order required by the assignment.
- The model files were created with the scikit-learn version available in the build environment. If deployment reports a model-version compatibility warning, run `python train_models.py` in the target environment and commit the regenerated model files together with the dependency versions.
