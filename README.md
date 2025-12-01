# 🧹 Data Preprocessing: Advanced Missing Value Imputation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange) ![Topic](https://img.shields.io/badge/Topic-Data%20Cleaning-yellow)

### 🔍 Project Overview
Real-world data is rarely clean. Missing values are a common plague in datasets, and simply dropping them often leads to information loss. This project explores and benchmarks varying strategies for **Data Imputation** to recover lost signals and improve machine learning model performance.

I compared **Univariate (Simple)** methods against **Multivariate (Iterative)** methods using the **Heart Disease dataset** to determine which technique yields the highest classification accuracy.

---

### 🧠 Imputation Techniques
I implemented two distinct approaches using `sklearn.impute`:

1.  **Simple Imputer (Univariate):**
    * *Logic:* Fills missing values in a column using statistics from *that column alone*.
    * *Strategies:* Mean, Median, Most Frequent (Mode), Constant.
    * *Pros/Cons:* Fast and easy, but ignores relationships between variables.

2.  **Iterative Imputer (Multivariate):**
    * *Logic:* Models each feature with missing values as a function of other features. It uses Machine Learning (e.g., BayesianRidge) to *predict* the missing value based on correlations in the data.
    * *Also known as:* MICE (Multivariate Imputation by Chained Equations).
    * *Pros/Cons:* More accurate for correlated data, but computationally expensive.

---

### 🧪 Experimental Design
To evaluate the effectiveness of each imputation method, I built a **Machine Learning Pipeline**:

1.  **Preprocessing:** Applied an Imputer (Simple vs. Iterative).
2.  **Classification:** Fed the imputed data into various models:
    * Logistic Regression
    * K-Nearest Neighbors (KNN)
    * Random Forest
    * Support Vector Machine (SVM)
3.  **Evaluation:** Used **Repeated Stratified K-Fold Cross-Validation** to calculate the Mean Accuracy and Standard Deviation for every Imputer-Model combination.

---

### 🏆 Key Findings
* **Simple Imputation:** Works well for features with low variance but struggles to capture complex patterns. 'Median' strategy was generally more robust to outliers than 'Mean'.
* **Iterative Imputation:** Consistently outperformed Simple Imputation when features were highly correlated (e.g., Age and Blood Pressure), as it could infer the missing data points from the context of the patient's other health metrics.

---

### 🛠️ Setup & Usage
1.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
2.  **Run the Notebooks:**
    * `Preprocessing_05_Simple_Imputer.ipynb` (For basic statistics)
    * `Preprocessing_06_Iterative_Imputer_Solution.ipynb` (For advanced modeling)
3.  **Data:**
    The project uses `heart_disease.csv`.

---

### 👨‍💻 About the Author
**Karthik Kunnamkumarath**
*Aerospace Engineer | Project Management Professional (PMP) | AI Solutions Developer*

I combine engineering precision with data science to solve complex problems.
* 📍 Toronto, ON
* 💼 [LinkedIn Profile](https://linkedin.com/in/4karthik95)
* 📧 Aero13027@gmail.com

---

### 💻 Code Snippet: Robust Pipeline
Here is how I automated the testing of different imputation strategies within a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

# Define the strategy
imputer = IterativeImputer(max_iter=10, random_state=0)
model = LogisticRegression()

# Build the pipeline
pipeline = Pipeline([
    ('impute', imputer),
    ('model', model)
])

# Evaluate using Cross-Validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print(f"Mean Accuracy: {scores.mean():.3f}")
