# Diabetes Prediction with a Linear SVM

Predicting the onset of diabetes from routine clinical measurements, with a focus on **clean preprocessing, honest evaluation and clinically meaningful metrics** rather than on chasing accuracy.

## Results at a glance

| Model | Accuracy | Precision (Diabetes) | Recall (Diabetes) | F1 (Diabetes) |
|---|---|---|---|---|
| Baseline: linear SVM, no class weighting | 0.75 | 0.74 | 0.46 | 0.57 |
| **Final: linear SVM, balanced class weights, 6 features** | **0.81** | **0.74** | **0.69** | **0.71** |

*Evaluated once on a held-out, stratified test set (154 patients).*

The baseline looked acceptable by accuracy but **missed more than half of the diabetic patients**. The final model finds 69 % of them while keeping precision unchanged.

![Confusion matrix](images/confusion_matrix.png)

## Dataset

Pima Indians Diabetes dataset: 768 female patients, 8 clinical features (e.g. glucose, BMI, blood pressure, number of pregnancies) and a binary outcome. The classes are imbalanced (500 without, 268 with diabetes).

## Approach

### 1. Data cleaning
Glucose, blood pressure, skin thickness, insulin and BMI contained values of **0, which are physiologically impossible**. These were treated as missing values and imputed with the median. Insulin was missing for almost half of the patients.

### 2. Preventing data leakage
In the first version the scaler was fitted on the full dataset before the train/test split, so information from the test set leaked into training. All preprocessing (imputation, scaling) now lives in a scikit-learn `Pipeline` that is fitted on training data only.

### 3. Choosing the right metric
In a screening context, overlooking a diabetic patient is worse than a false alarm. Accuracy hides this, so the evaluation focuses on **recall for the diabetes class**, together with precision and F1.

### 4. Handling class imbalance
Using `class_weight='balanced'` raised recall from 0.46 to 0.69 without lowering precision.

### 5. Model comparison (5-fold stratified cross-validation on training data)

| Model | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|
| **Linear SVM** | 0.72 | 0.64 | **0.68** | 0.76 |
| SVM (RBF kernel) | 0.76 | 0.60 | 0.67 | 0.74 |
| Logistic Regression | 0.72 | 0.61 | 0.66 | 0.74 |
| Gradient Boosting | 0.61 | 0.66 | 0.63 | 0.75 |
| Random Forest | 0.57 | 0.64 | 0.60 | 0.74 |

All models perform within a few percentage points of each other, which suggests the **limiting factor is the data** (small sample, many missing values), not the model. The linear SVM was kept because it scored best on F1 and is the easiest to interpret.

### 6. Feature selection

| Feature set | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|
| All 8 features | 0.72 | 0.64 | 0.68 | 0.76 |
| **Without insulin and skin thickness** | 0.71 | 0.64 | 0.67 | 0.76 |
| Without insulin, skin thickness and blood pressure | 0.70 | 0.64 | 0.67 | 0.76 |

Dropping insulin and skin thickness made no meaningful difference. Since these are the two features with the most missing values and the most effort to measure, the final model uses the remaining six.

## Feature importance

![Feature importance](images/feature_importance.png)

The weights of the linear SVM are clinically plausible:

- **Glucose** has by far the strongest influence, as expected for a diabetes test.
- **BMI** is a well-known risk factor.
- **Number of pregnancies** also contributes, consistent with gestational diabetes raising later risk.

Small or negative weights (e.g. insulin, blood pressure) should **not** be read as protective effects. They reflect noisy, heavily imputed features and correlations between features (for example, age and number of pregnancies share much of the same information).

## Limitations

- Small dataset (768 patients) from a specific population (female, Pima heritage), so results do not generalise directly.
- Many imputed values, especially for insulin.
- Not a medical device and not intended for clinical use.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib

## How to run

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook DiabetesPrediction.ipynb
```

Place `diabetes.csv` in the same folder as the notebook.
