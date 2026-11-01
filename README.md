# Machine Learning Laboratory Manual & Practical Reference Guide

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)](#)

---

## Table of Contents
1. [Laboratory Context & Overview](#laboratory-context--overview)
2. [Experiment Architecture & Index](#experiment-architecture--index)
3. [Practical Lab Implementations (pt1.ipynb – pt7.ipynb: Detailed Reference)](#practical-lab-implementations-pt1ipynb--pt7ipynb-detailed-reference)
   - [Practical 01: Basic Feature Engineering Techniques (`pt1.ipynb`)](#practical-01-basic-feature-engineering-techniques-pt1ipynb)
   - [Practical 02: Dimensionality Reduction using PCA and LDA (`pt2.ipynb`)](#practical-02-dimensionality-reduction-using-pca-and-lda-pt2ipynb)
   - [Practical 03: Multiple Linear Regression (`pt3.ipynb`)](#practical-03-multiple-linear-regression-pt3ipynb)
   - [Practical 04: Polynomial Regression & Bias-Variance Trade-off (`pt4.ipynb`)](#practical-04-polynomial-regression--bias-variance-trade-off-pt4ipynb)
   - [Practical 05: Ridge and Lasso Regularization (`pt5.ipynb`)](#practical-05-ridge-and-lasso-regularization-pt5ipynb)
   - [Practical 06: Logistic Regression for Binary Classification (`pt6.ipynb`)](#practical-06-logistic-regression-for-binary-classification-pt6ipynb)
   - [Practical 07: Support Vector Machines (SVM) with Kernels (`pt7.ipynb`)](#practical-07-support-vector-machines-svm-with-kernels-pt7ipynb)
   - [Practicals 08 – 11 Overview](#practicals-08--11-overview)
4. [Master Rapid-Fire & Viva Q&A Reference Guide (from `untitled.txt`)](#master-rapid-fire--viva-qa-reference-guide)
   - [1. Python & Scientific Computing Stack Rapid Questions](#1-python--scientific-computing-stack-rapid-questions)
   - [2. Experiment 06: Logistic Regression Rapid Viva Questions](#2-experiment-06-logistic-regression-rapid-viva-questions)
   - [3. Experiment 09: DBSCAN Density-Based Clustering Rapid Viva Questions](#3-experiment-09-dbscan-density-based-clustering-rapid-viva-questions)
   - [4. Foundational & Basic Machine Learning Rapid Questions](#4-foundational--basic-machine-learning-rapid-questions)
   - [5. Learning Paradigms & Machine Learning Types](#5-learning-paradigms--machine-learning-types)
   - [6. Model Evaluation & Error Metrics Rapid Questions](#6-model-evaluation--error-metrics-rapid-questions)
   - [7. Machine Learning Algorithms Rapid Questions](#7-machine-learning-algorithms-rapid-questions)
5. [Repository Structure](#repository-structure)
6. [Getting Started & Running Notebooks](#getting-started--running-notebooks)

---

## Laboratory Context & Overview

This repository constitutes a comprehensive, hands-on **Machine Learning Laboratory Manual and Practical Reference Suite**. Designed for academic rigor and practical industry relevance, it bridges theoretical statistical learning principles with end-to-end Python implementations using modern scientific computing stacks (`scikit-learn`, `numpy`, `pandas`, `matplotlib`, and `seaborn`).

### Core Learning Objectives
* **Data Preprocessing & Feature Engineering**: Address data impurities via median/mode imputation, categorical encodings (Label and One-Hot), and continuous variable scaling (Z-score normalization).
* **Dimensionality Reduction**: Contrast unsupervised variance maximization (Principal Component Analysis - PCA) with supervised class-separation maximization (Linear Discriminant Analysis - LDA).
* **Regression & Regularization**: Formulate Ordinary Least Squares (OLS) solutions, diagnose bias-variance tradeoffs through polynomial complexity sweeps, and apply $L_1$ (Lasso) and $L_2$ (Ridge) penalties to combat multicollinearity and prevent overfitting.
* **Supervised Classification & Decision Diagnostics**: Implement discriminative classifiers (Logistic Regression, Kernel SVMs) and diagnostic evaluation metrics beyond raw accuracy (Precision, Recall, $F_1$, Confusion Matrices).
* **Unsupervised Clustering & Density Mining**: Implement centroid partitioning ($K$-Means with Elbow and Silhouette criteria) and density reachability algorithms (DBSCAN) capable of isolating arbitrary shapes and noise points.
* **Ensemble Methods**: Analyze variance reduction via Bootstrap Aggregating (Random Forests) versus iterative bias reduction (Gradient Tree Boosting).
* **Reinforcement Learning**: Formulate Markov Decision Processes (MDPs), the Bellman Optimality Equation, and value-based tabular temporal-difference learning (Q-learning with $\varepsilon$-greedy exploration).

---

## Experiment Architecture & Index

| No. | Notebook / Code Link | Experiment Title | Primary Dataset | Core Method / Deliverable | Key Benchmark Result |
|:---:|:---|:---|:---|:---|:---|
| **01** | [`pt1.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt1.ipynb) | Basic Feature Engineering Techniques | Titanic (`sns.load_dataset`) | Imputation, Label & One-Hot Encoding, StandardScaler | 0 missing, shape (891, 10), $\mu=0, \sigma=1$ |
| **02** | [`pt2.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt2.ipynb) | Dimensionality Reduction (PCA & LDA) | Breast Cancer Diagnostic | Unsupervised (PCA) vs Supervised (LDA) feature compression | PCA 10 PCs = 95.60% var; Acc = 98.63% |
| **03** | [`pt3.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt3.ipynb) | Multiple Linear Regression | California Housing | Multivariate OLS Normal Equations, Residual Diagnostics | MSE: 0.5559, RMSE: 0.7456, $R^2$: 0.5758 |
| **04** | [`pt4.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt4.ipynb) | Polynomial Regression & Bias-Variance | California Housing (`MedInc`) | Degree sweep ($d=1, 2, 3, 5, 8$), U-curve error analysis | Optimal $d^*=3$ (Min Test MSE: 0.6983) |
| **05** | [`pt5.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt5.ipynb) | Ridge and Lasso Regularization | Diabetes Dataset | $L_2$ (Ridge) vs $L_1$ (Lasso) penalty, coefficient path | Lasso Test MSE: 2884.62, $s_3$ shrunk to 0.55 |
| **06** | [`pt6.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt6.ipynb) | Logistic Regression Binary Classification | Breast Cancer Diagnostic | Sigmoid logit link, Log-Loss, Confusion Matrix, Report | Accuracy: 97.37%, $F_1$: 0.98, TN: 41, TP: 70 |
| **07** | [`pt7.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt7.ipynb) | SVM with Linear & Nonlinear Kernels | Wine Recognition Dataset | Linear, Polynomial ($d=3$), RBF Mercer Kernels | RBF Accuracy: **100.0%** (36/36 correct) |
| **08** | [`pt8.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt8.ipynb) | K-Means Clustering & Optimal Clusters | Iris Dataset | Centroid partitioning, WCSS minimization, 2D Scatter | $k=3$ cluster separation |
| **09** | [`pt9.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt9.ipynb) | Density-Based Spatial Clustering (DBSCAN) | Wine Dataset | Density reachability, core/border/noise isolation | Core/border/noise point labeling |
| **10** | [`pt10.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt10.ipynb) | Ensemble Learning: Forest & Boosting | Breast Cancer Diagnostic | Bagging (Random Forest) vs Boosting (Gradient Tree) | Variance reduction vs Bias reduction |
| **11** | [`pt11.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt11.ipynb) | Reinforcement Learning: Q-Learning Grid | 4×4 Grid World | Tabular Q-learning, Bellman update, $\varepsilon$-greedy | Converged optimal path policy |

> [!NOTE]
> All implementations are fully reproducible in their respective standalone Jupyter notebooks ([`pt1.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt1.ipynb) – [`pt11.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt11.ipynb)) and unified in [`allcodes.py`](file:///C:/Users/elite/OneDrive/Documents/mllab/allcodes.py).

---

## Practical Lab Implementations (pt1.ipynb – pt7.ipynb: Detailed Reference)

This section provides in-depth technical references, mathematical foundations, code walkthroughs, empirical outputs, and dedicated 1-2 line rapid viva questions for each practical executed up to **pt7**.

---

### Practical 01: Basic Feature Engineering Techniques (`pt1.ipynb`)
* **Notebook File**: [`pt1.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt1.ipynb)
* **Objective**: Transform raw, incomplete, and heterogeneous tabular records into a clean, normalized numeric matrix ready for downstream estimators.
* **Dataset Used**: Titanic Dataset loaded via Seaborn (`sns.load_dataset("titanic")`), containing 891 records and 15 original features.
* **Selected Features**: `["survived", "sex", "age", "parch", "sibsp", "fare", "embarked"]`.

#### Pipeline & Implementation Steps:
1. **Handling Missing Values (Imputation)**:
   - Numerical (`age`): Imputed with the column median ($28.0$ years) to maintain robustness against positive skewness:
     ```python
     data["age"] = data["age"].fillna(data["age"].median())
     ```
   - Categorical (`embarked`): Imputed with the empirical mode (`'S'`, Southampton) representing the most frequent port of embarkation:
     ```python
     data["embarked"] = data["embarked"].fillna(data["embarked"].mode()[0])
     ```
2. **Categorical Encoding**:
   - Binary encoding: `sex` mapped using `LabelEncoder()` (`female -> 0, male -> 1`).
   - Nominal encoding: `embarked` converted to binary indicator variables using One-Hot Encoding (`pd.get_dummies(data, columns=["embarked"], dtype=int)`), generating `embarked_C`, `embarked_Q`, and `embarked_S`.
3. **Feature Scaling**:
   - Standardized continuous numerical columns `["age", "sibsp", "parch", "fare"]` using `StandardScaler()`:
     $$z = \frac{x - \mu}{\sigma}$$
     ensuring zero mean ($\mu \approx 0$) and unit variance ($\sigma = 1$).

#### Observed Execution Results:
* Missing values before: `Age` (177 missing), `Embarked` (2 missing).
* Missing values after preprocessing: **0 missing values**.
* Final dataset dimensions: **(891, 10)** model-ready matrix.
* Diagnostic Plot: Subplot comparing Age distribution before imputation (showing missing gaps) and after median imputation + scaling (centered at 0).

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: Why is median imputation preferred over mean imputation for skewed numerical data?**
  - **Answer**: The median is robust to extreme outliers and heavy-tailed distributions, whereas the mean gets distorted toward extreme values.
* **Q2: What is the difference between Label Encoding and One-Hot Encoding?**
  - **Answer**: Label Encoding assigns sequential integers ($0, 1, 2$) which imposes an artificial order, while One-Hot Encoding creates separate binary indicator columns for non-ordinal categories.
* **Q3: What does `StandardScaler()` achieve mathematically?**
  - **Answer**: It transforms continuous features by subtracting the sample mean and dividing by the standard deviation ($z = \frac{x-\mu}{\sigma}$), giving $\mu=0$ and $\sigma=1$.

---

### Practical 02: Dimensionality Reduction using PCA and LDA (`pt2.ipynb`)
* **Notebook File**: [`pt2.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt2.ipynb)
* **Objective**: Mitigate the curse of dimensionality while maximizing retained informational variance and class separability on high-dimensional clinical diagnostic data.
* **Dataset Used**: Breast Cancer Wisconsin Diagnostic (`load_breast_cancer(return_X_y=True)`), consisting of 569 patient samples and 30 continuous cytologic features.

#### Pipeline & Implementation Steps:
1. **Preprocessing**: Stratified 80/20 train/test split (`random_state=42, stratify=y`) followed by `StandardScaler()` normalization.
2. **Baseline Evaluation**: Train `LogisticRegression(max_iter=5000)` on all 30 original features.
3. **Principal Component Analysis (PCA - Unsupervised)**:
   - Compute covariance matrix $\Sigma = \frac{1}{n} X^T X$ and extract orthogonal eigenvectors.
   - Project features onto top 10 principal components (`n_components=10`).
   - Retains **95.60%** cumulative explained variance ratio:
     ```python
     pca = PCA(n_components=10)
     Xp = pca.fit_transform(X)
     ```
4. **Linear Discriminant Analysis (LDA - Supervised)**:
   - Maximize Fisher's criterion: between-class scatter relative to within-class scatter ($\frac{S_B}{S_W}$).
   - Projects 30 features onto a single discriminant component ($C - 1 = 2 - 1 = 1$).
5. **Execution Benchmarking**: Measure model accuracy and average training execution latency across 50 repeated runs.

#### Observed Execution Results:
| Case | Feature Count | Accuracy | Average Execution Time | Variance Retained |
|:---|:---:|:---:|:---:|:---:|
| **Original** | 30 | **0.9863** (98.63%) | 6.817 ms | 100.0% |
| **PCA** | 10 | **0.9863** (98.63%) | 7.033 ms | **95.60%** |
| **LDA** | 1 | **0.9863** (98.63%) | **5.460 ms** | Supervised projection |

* Key Insight: LDA compresses the 30-dimensional space down to **1 single feature** with zero loss in test classification accuracy while cutting execution latency.
* Diagnostic Plots: 2D scatter of PC1 vs PC2 showing clean class clusters, and 1D LDA histogram demonstrating distinct bimodal separation between malignant and benign distributions.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: What is the fundamental difference between PCA and LDA?**
  - **Answer**: PCA is an unsupervised technique that maximizes total feature variance without class labels, whereas LDA is supervised and maximizes the separation between labeled classes.
* **Q2: What is the maximum number of components LDA can extract?**
  - **Answer**: At most $C - 1$ components, where $C$ is the number of target classes (for binary classification, exactly 1 component).
* **Q3: What does `pca.explained_variance_ratio_` tell us?**
  - **Answer**: It indicates the proportion of total dataset variance accounted for by each individual principal component.

---

### Practical 03: Multiple Linear Regression (`pt3.ipynb`)
* **Notebook File**: [`pt3.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt3.ipynb)
* **Objective**: Model continuous real estate valuations using multivariate Ordinary Least Squares (OLS) regression and validate residual normality assumptions.
* **Dataset Used**: California Housing Dataset (`fetch_california_housing(as_frame=True)`), 20,640 census block groups, 8 continuous predictor features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`), target `MedHouseVal` (median house value in $100k).

#### Pipeline & Implementation Steps:
1. **Splitting**: 80/20 train/test split (`random_state=42`), producing 16,512 training instances and 4,128 test instances.
2. **Model Training**: Fit standard multivariate `LinearRegression()` estimating parameter vector $\hat{\beta}$:
   $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_8 x_8$$
3. **Inference & Metric Computation**:
   ```python
   yp = model.predict(xt)
   mse = mean_squared_error(yt, yp)
   rmse = np.sqrt(mse)
   r2 = r2_score(yt, yp)
   ```
4. **Diagnostic Residual Analysis**: Calculate prediction residuals $e_i = y_i - \hat{y}_i$.

#### Observed Execution Results:
* **Mean Squared Error (MSE)**: **0.5559**
* **Root Mean Squared Error (RMSE)**: **0.7456** (representing an average error of ~$74,560)
* **Coefficient of Determination ($R^2$)**: **0.5758** (explains 57.58% of target price variance)
* Diagnostic Plots:
  1. Actual vs Predicted scatter plot aligning tightly along the red dashed ideal 45° reference line ($y = \hat{y}$).
  2. Residual distribution histogram: symmetrical bell curve centered at $\mu = 0$, confirming the Gauss-Markov normality assumption.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: What is the closed-form normal equation for OLS Linear Regression?**
  - **Answer**: $\hat{\beta} = (X^T X)^{-1} X^T y$, which computes the optimal coefficient vector minimizing the sum of squared residuals.
* **Q2: What does an $R^2$ score of 0.5758 signify?**
  - **Answer**: It means that 57.58% of the total variation in the dependent target variable is explained by the linear combination of predictors.
* **Q3: What do the Gauss-Markov assumptions require for OLS to be BLUE (Best Linear Unbiased Estimator)?**
  - **Answer**: Linearity, exogeneity ($E[\varepsilon|X]=0$), homoscedasticity (constant error variance), absence of severe multicollinearity, and zero serial correlation among residuals.

---

### Practical 04: Polynomial Regression & Bias-Variance Trade-off (`pt4.ipynb`)
* **Notebook File**: [`pt4.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt4.ipynb)
* **Objective**: Empirically evaluate the Bias-Variance tradeoff by sweeping hypothesis complexity across polynomial degrees and locating the optimal degree $d^*$.
* **Dataset Used**: California Housing Dataset (`fetch_california_housing(as_frame=True)`), isolating `MedInc` (Median Income) as the primary non-linear predictor for `MedHouseVal`.

#### Pipeline & Implementation Steps:
1. **Splitting**: 80/20 train/test split (`random_state=42`).
2. **Polynomial Feature Transformation**: For degrees $d \in \{1, 2, 3, 5, 8\}$, transform scalar feature via `PolynomialFeatures(degree=d)`:
   $$\phi(x) = [1, x, x^2, \dots, x^d]$$
3. **Model Sweep**: Fit standard `LinearRegression()` on each transformed space, recording Train MSE and Test MSE.
4. **Optimal Selection**: Locate degree minimizing out-of-sample Test MSE:
   ```python
   best = ress.loc[ress["Test MSE"].idxmin()]
   ```

#### Observed Execution Results:
| Polynomial Degree ($d$) | Train MSE | Test MSE | Complexity Assessment |
|:---:|:---:|:---:|:---|
| **1** (Linear) | 0.6991 | 0.7091 | High Bias / Underfitting |
| **2** (Quadratic) | 0.6930 | 0.7033 | Capturing modest curvature |
| **3** (Cubic) | **0.6807** | **0.6983** | **Optimal Complexity ($d^*=3$) (Lowest Test MSE)** |
| **5** (Quintic) | 0.6804 | 0.6987 | Slight excess parameters |
| **8** (Octic) | 0.7120 | 0.7407 | High Variance / Overfitting (Runge's phenomenon) |

* Best Model: **Degree 3** achieved the lowest test error (**0.698296**).
* Diagnostic Plots:
  1. Non-linear fit curves for Degrees 1, 2, and 5 overlaid on scatter data.
  2. Bias-Variance curve plotting Degree vs Train/Test MSE, demonstrating the classic U-shaped validation curve.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: What is the Bias-Variance Trade-off?**
  - **Answer**: Simple models suffer from high bias (underfitting), while overly complex models suffer from high variance (overfitting); generalization error is minimized at an optimal intermediate complexity.
* **Q2: Why did Test MSE increase at Degree 8 despite higher model flexibility?**
  - **Answer**: The high-degree polynomial overfitted to random training noise, causing runaway boundary fluctuations and degraded out-of-sample generalization.
* **Q3: What was the optimal polynomial degree identified in practical 4?**
  - **Answer**: Degree 3 yielded the lowest generalization error with a Test MSE of 0.6983.

---

### Practical 05: Ridge and Lasso Regularization (`pt5.ipynb`)
* **Notebook File**: [`pt5.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt5.ipynb)
* **Objective**: Prevent overfitting and stabilize collinear regression coefficients using $L_2$ (Ridge) and $L_1$ (Lasso) penalized loss functions.
* **Dataset Used**: Diabetes Dataset (`load_diabetes(as_frame=True)`), 442 patients, 10 standardized physiological predictors (`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`), continuous target measure of 1-year disease progression.

#### Pipeline & Implementation Steps:
1. **Preprocessing**: 80/20 train/test split (`random_state=42`) followed by `StandardScaler()`.
2. **Model Training & Comparison**:
   - Ordinary Least Squares: `LinearRegression()`
   - Ridge Regression: `Ridge(alpha=1.0)` with $L_2$ penalty: $\mathcal{L} = \text{MSE} + \alpha \sum_{j=1}^p w_j^2$
   - Lasso Regression: `Lasso(alpha=0.1)` with $L_1$ penalty: $\mathcal{L} = \text{MSE} + \alpha \sum_{j=1}^p |w_j|$
3. **Coefficient Shrinkage Path Analysis**: Trace Lasso parameter paths across 50 log-spaced $\alpha$ values from $10^{-2}$ to $10^2$:
   ```python
   alphas = np.logspace(-2, 2, 50)
   ```

#### Observed Execution Results:
| Estimator | MSE | RMSE | $R^2$ | Penalty Mechanism |
|:---|:---:|:---:|:---:|:---|
| **Linear Regression** | 2900.19 | 53.85 | 0.4526 | None (unconstrained OLS) |
| **Ridge ($\alpha=1.0$)** | 2892.01 | 53.78 | 0.4541 | Smooth shrinkage ($L_2$) |
| **Lasso ($\alpha=0.1$)** | **2884.62** | **53.71** | **0.4555** | Feature sparsity ($L_1$) |

* Coefficient Behavior: In standard Linear Regression, collinear blood serum features (`s1` and `s2`) explode to $-44.45$ and $+24.64$. Ridge shrinks them to $-34.67$ and $+17.05$. Lasso aggressively shrinks weak feature `s3` from $7.68$ down to $0.55$, demonstrating automated sparsity selection.
* Diagnostic Plots:
  1. Grouped bar chart comparing feature coefficient values across Linear, Ridge, and Lasso.
  2. Lasso Coefficient Path showing weights monotonically collapsing to zero as $\alpha$ increases.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: What is the main difference between Ridge ($L_2$) and Lasso ($L_1$) regression?**
  - **Answer**: Ridge shrinks coefficients smoothly toward zero but never makes them exactly zero, while Lasso forces less important coefficients to exactly zero, performing automated feature selection.
* **Q2: Why is feature standardization essential before applying Ridge or Lasso?**
  - **Answer**: Regularization penalties treat all weights equally; without scaling, features with larger numerical scales would be unfairly penalized regardless of true importance.
* **Q3: What happens when the regularization parameter $\alpha = 0$?**
  - **Answer**: The regularization penalty vanishes, reducing Ridge and Lasso to standard Ordinary Least Squares (OLS) Linear Regression.

---

### Practical 06: Logistic Regression for Binary Classification (`pt6.ipynb`)
* **Notebook File**: [`pt6.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt6.ipynb)
* **Objective**: Classify breast cancer cytology as malignant or benign using the logistic sigmoid link function and evaluate multi-metric diagnostic performance.
* **Dataset Used**: Breast Cancer Wisconsin Diagnostic (`load_breast_cancer(as_frame=True)`), 569 patient samples, 30 continuous cytologic features, binary target ($0 = \text{Malignant}, 1 = \text{Benign}$).

#### Pipeline & Implementation Steps:
1. **Preprocessing**: 80/20 train/test split (`random_state=42`) followed by `StandardScaler()`.
2. **Estimator Optimization**: Fit `LogisticRegression(max_iter=5000)` minimizing Binary Cross-Entropy (Log-Loss):
   $$P(Y=1|X) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$
3. **Model Evaluation**: Compute classification Accuracy, Confusion Matrix, and full Classification Report (Precision, Recall, $F_1$-score, Support).
4. **Visualization**: Generate annotated Seaborn confusion matrix heatmap.

#### Observed Execution Results:
* **Test Accuracy**: **0.9737** (97.37% - 111 out of 114 test samples correctly classified).
* **Confusion Matrix ($N=114$)**:
  - True Negative ($TN$, Malignant correctly diagnosed): **41**
  - False Positive ($FP$, Malignant misdiagnosed as Benign): **2** (Type I Error)
  - False Negative ($FN$, Benign misdiagnosed as Malignant): **1** (Type II Error)
  - True Positive ($TP$, Benign correctly diagnosed): **70**
* **Classification Report Breakdown**:
  | Class | Precision | Recall | $F_1$-Score | Support |
  |:---|:---:|:---:|:---:|:---:|
  | **0 (Malignant)** | 0.98 | 0.95 | 0.96 | 43 |
  | **1 (Benign)** | 0.97 | 0.99 | 0.98 | 71 |
  | **Macro Average** | **0.98** | **0.97** | **0.97** | 114 |
  | **Weighted Average** | **0.97** | **0.97** | **0.97** | 114 |
* Diagnostic Plot: Blue-scaled confusion matrix heatmap clearly annotating $TN=41, FP=2, FN=1, TP=70$.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: Why is Logistic Regression called "regression" if it is used for classification?**
  - **Answer**: It models a continuous linear regression function ($z = w^T x + b$) internally, then maps the score through the non-linear logistic sigmoid link function to produce bounded probabilities.
* **Q2: What is the default decision threshold in Logistic Regression?**
  - **Answer**: The default threshold is $0.5$; samples with predicted probability $\ge 0.5$ are assigned to class 1, otherwise class 0.
* **Q3: What loss function is optimized in Logistic Regression?**
  - **Answer**: Binary Cross-Entropy (Log-Loss): $\mathcal{L} = -\frac{1}{n} \sum [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$.

---

### Practical 07: Support Vector Machines (SVM) with Kernels (`pt7.ipynb`)
* **Notebook File**: [`pt7.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt7.ipynb)
* **Objective**: Evaluate maximum-margin hyperplanes across linear, polynomial, and radial basis function (RBF) kernels on multi-class chemical profiling data.
* **Dataset Used**: Wine Recognition Dataset (`load_wine(as_frame=True)`), 178 chemical profiling samples, 13 features (Alcohol, Malic acid, Flavanoids, Color intensity, Proline, etc.), 3 cultivar target classes (0, 1, 2).

#### Pipeline & Implementation Steps:
1. **Preprocessing**: 80/20 train/test split (`random_state=42`) followed by `StandardScaler()`.
2. **Kernel Comparison Sweep**: Iterate over candidate Mercer kernel functions using `SVC(kernel=k)`:
   - Linear Kernel: $K(x, x') = x^T x'$
   - Polynomial Kernel: $K(x, x') = (\gamma x^T x' + r)^d$ ($d=3$)
   - Radial Basis Function (RBF): $K(x, x') = \exp(-\gamma ||x - x'||^2)$
3. **Optimal Selection & Evaluation**: Select the kernel achieving highest validation accuracy, fit final estimator, and generate confusion matrix.

#### Observed Execution Results:
| SVM Kernel | Mathematical Formulation | Test Accuracy | Correct / Total |
|:---|:---|:---:|:---:|
| **Linear** | $K(x, x') = x^T x'$ | 0.9722 (97.22%) | 35 / 36 |
| **Polynomial ($d=3$)** | $K(x, x') = (\gamma x^T x' + r)^3$ | 0.9722 (97.22%) | 35 / 36 |
| **RBF (Gaussian)** | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ | **1.0000 (100.0%)** | **36 / 36** |

* Best Kernel: **RBF achieved perfect 100% test accuracy** (36/36 correct).
* Confusion Matrix (RBF):
  - Class 0: 14/14 correct
  - Class 1: 14/14 correct
  - Class 2: 8/8 correct (zero misclassifications across all 3 classes).
* Diagnostic Plots:
  1. Bar chart comparing test accuracies across Linear, Poly, and RBF kernels.
  2. 3×3 Confusion Matrix heatmap showing a purely diagonal, error-free matrix.

#### Rapid Viva Questions (1-2 Line Answers):
* **Q1: What are Support Vectors in an SVM?**
  - **Answer**: Support vectors are the critical data points closest to the separating hyperplane that define the margin boundaries; removing other points leaves the boundary unchanged.
* **Q2: What is the "Kernel Trick" in SVM?**
  - **Answer**: It computes inner products in high-dimensional feature spaces implicitly using kernel functions without ever needing to calculate the high-dimensional coordinates explicitly.
* **Q3: Why did the RBF kernel achieve 100% accuracy in practical 7?**
  - **Answer**: The RBF kernel maps features into an infinite-dimensional Hilbert space, providing smooth non-linear boundaries that easily separate overlapping chemical distributions.

---

### Practicals 08 – 11 Overview
The remaining experiments are structured as follows:
* **Practical 08 (`pt8.ipynb`)**: K-Means Clustering on the Iris dataset; partitions unlabeled samples into $k=3$ clusters by minimizing Within-Cluster Sum of Squares (Inertia) with PCA visualization.
* **Practical 09 (`pt9.ipynb`)**: DBSCAN Density-Based Clustering on the Wine dataset; discovers arbitrary shapes and isolates noise points using density reachability ($\varepsilon, \text{min\_samples}$).
* **Practical 10 (`pt10.ipynb`)**: Ensemble Learning on Breast Cancer Diagnostic; contrasts parallel variance reduction (Random Forest) against sequential bias reduction (Gradient Tree Boosting).
* **Practical 11 (`pt11.ipynb`)**: Tabular Reinforcement Learning in a 4×4 Grid World; solves optimal path policies using the Bellman Optimality Equation and $\varepsilon$-greedy exploration.

---

## Master Rapid-Fire & Viva Q&A Reference Guide

> [!IMPORTANT]
> The following questions and answers are compiled directly from the laboratory syllabus and revision materials (`untitled.txt`). Every question is answered concisely in 1–2 crisp, precise lines tailored for viva-voce and rapid oral evaluations.

---

### 1. Python & Scientific Computing Stack Rapid Questions

#### Q1. Which programming language is commonly used in Machine Learning?
**Answer**: Python is the primary programming language for Machine Learning due to its clear syntax, extensive scientific library ecosystem, and massive community support.

#### Q2. What is Python?
**Answer**: Python is a high-level, interpreted, dynamically-typed programming language widely adopted for data science, artificial intelligence, and scientific computing.

#### Q3. What is NumPy?
**Answer**: NumPy (Numerical Python) is the foundational Python library for numerical computation, providing high-performance multidimensional arrays (`ndarray`) and vectorized mathematical operations.

#### Q4. What is Pandas?
**Answer**: Pandas is an open-source Python library providing fast, flexible data structures—primarily Series (1D) and DataFrame (2D)—designed for tabular data analysis and manipulation.

#### Q5. What is Matplotlib?
**Answer**: Matplotlib is a comprehensive 2D data visualization library in Python used to generate static, interactive, and publication-quality plots (line charts, histograms, scatter plots).

#### Q6. What is Scikit-learn?
**Answer**: Scikit-learn (`sklearn`) is the premier Python machine learning library built on NumPy, SciPy, and Matplotlib, providing standardized algorithms for preprocessing, classification, regression, clustering, and model evaluation.

#### Q7. What is a DataFrame?
**Answer**: A DataFrame is a 2-dimensional, size-mutable tabular data structure in Pandas with labeled rows and columns, capable of storing heterogeneous data types.

#### Q8. What is an array?
**Answer**: An array (such as a NumPy `ndarray`) is a contiguous memory collection of elements of the same data type indexed by integers, optimized for fast vectorized mathematical computation.

#### Q9. What is the purpose of `train_test_split()`?
**Answer**: It partitions a dataset into separate training and testing subsets to train the model on one portion and evaluate its out-of-sample generalization on unseen data to detect overfitting.

#### Q10. What is the purpose of `fit()`?
**Answer**: The `fit()` method calculates internal model parameters (e.g., weights in regression or mean/variance in `StandardScaler`) directly from the provided training data.

#### Q11. What is the purpose of `predict()`?
**Answer**: The `predict()` method uses a trained model to generate estimated target labels or continuous output values for new, unseen input feature vectors.

#### Q12. What is the purpose of `transform()`?
**Answer**: The `transform()` method applies previously learned parameters (computed during `fit()`) to scale or encode new data without recalculating parameters, preventing test data leakage.

#### Q13. What is the purpose of `fit_transform()`?
**Answer**: The `fit_transform()` method combines `fit()` and `transform()` into a single efficient step, computing parameters and transforming the training data simultaneously.

#### Q14. What is the use of `accuracy_score()`?
**Answer**: It computes the classification accuracy fraction: $\frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$.

#### Q15. What is the use of `confusion_matrix()`?
**Answer**: It generates an $N \times N$ contingency table contrasting true actual classes against predicted classes, detailing True Positives, True Negatives, False Positives, and False Negatives.

---

### 2. Experiment 06: Logistic Regression Rapid Viva Questions

#### Q1. What is Logistic Regression?
**Answer**: Logistic Regression is a supervised machine learning algorithm used for classification problems that estimates the probability of a categorical outcome using the logistic (sigmoid) function.

#### Q2. Why is Logistic Regression used for binary classification?
**Answer**: Because its sigmoid function maps any real-valued linear input into a bounded range between 0 and 1, which represents a valid probability that can be split into two classes.

#### Q3. Is Logistic Regression supervised or unsupervised?
**Answer**: Logistic Regression is a supervised learning algorithm because it requires labeled training data ($x_i, y_i$) to optimize model weights.

#### Q4. Why is it called Logistic Regression?
**Answer**: It is called "regression" because it fits a continuous linear combination of inputs ($w^T x + b$), and "logistic" because it passes that continuous score through the logistic (sigmoid) link function.

#### Q5. What is the output of Logistic Regression?
**Answer**: It outputs a continuous posterior probability value between 0 and 1 ($P(Y=1|X) = \frac{1}{1 + e^{-z}}$), which is thresholded to produce a discrete class label (0 or 1).

#### Q6. What is the threshold value in Logistic Regression?
**Answer**: The default decision threshold is 0.5. If the predicted probability is $\ge 0.5$, the output is Class 1; otherwise, it is assigned to Class 0.

---

### 3. Experiment 09: DBSCAN Density-Based Clustering Rapid Viva Questions

#### Q1. What is the full form of DBSCAN?
**Answer**: Density-Based Spatial Clustering of Applications with Noise.

#### Q2. Is DBSCAN supervised or unsupervised?
**Answer**: DBSCAN is an unsupervised clustering algorithm because it groups data points based on spatial density without requiring class labels.

#### Q3. What are the main parameters of DBSCAN?
**Answer**:
1. **Epsilon ($\varepsilon$ / `eps`)**: The maximum distance radius between neighboring points.
2. **MinPts (`min_samples`)**: The minimum number of points required within an $\varepsilon$-neighborhood to form a dense region.

#### Q4. What are the three types of points in DBSCAN?
**Answer**: Core Points, Border Points, and Noise Points (outliers).

#### Q5. What is a Core Point?
**Answer**: A point that contains at least `min_samples` (MinPts) points within its $\varepsilon$-neighborhood (including itself).

#### Q6. What is a Border Point?
**Answer**: A point that has fewer than `min_samples` points in its $\varepsilon$-neighborhood, but lies within the $\varepsilon$-neighborhood of a Core Point.

#### Q7. What is a Noise Point?
**Answer**: A point that is neither a Core Point nor a Border Point; it does not belong to any cluster and is tagged as an outlier (labeled as $-1$).

#### Q8. Does DBSCAN require the number of clusters in advance?
**Answer**: No, DBSCAN automatically discovers the natural number of clusters based on spatial density reachability without requiring $k$ to be predefined.

#### Q9. What is the main advantage of DBSCAN?
**Answer**: It can detect clusters of arbitrary geometric shapes (non-spherical) and effectively isolates noise and outliers.

#### Q10. What is the disadvantage of DBSCAN?
**Answer**: It struggles with datasets having varying density regions and is sensitive to the selection of the $\varepsilon$ and `min_samples` hyperparameters.

#### Q11. Where is DBSCAN used?
**Answer**: It is widely used in geographical data analysis, GPS trajectory mining, satellite image segmentation, financial fraud detection, and telemetry anomaly detection.

#### Q12. What is the time complexity of DBSCAN?
**Answer**: With spatial indexing structures (like $k$-d tree), its average time complexity is $O(n \log n)$; without an index (brute force pairwise distances), it is $O(n^2)$.

---

### 4. Foundational & Basic Machine Learning Rapid Questions

#### Q1. What is Machine Learning?
**Answer**: Machine Learning is a field of artificial intelligence that creates mathematical models and algorithms allowing computational systems to learn patterns from empirical data and make decisions without explicit programming.

#### Q2. What are the applications of Machine Learning?
**Answer**: Common applications include computer vision (facial recognition, medical diagnostics), NLP (translation, chatbots), recommendation systems, fraud detection, and autonomous vehicles.

#### Q3. What are the different types of Machine Learning?
**Answer**: Supervised Learning, Unsupervised Learning, Semi-Supervised Learning, and Reinforcement Learning.

#### Q4. What is the difference between AI, ML, and Deep Learning?
**Answer**: AI is the broad umbrella of machines mimicking human intelligence; ML is a subset of AI that learns patterns from data; Deep Learning is a specialized subset of ML using multi-layered artificial neural networks.

#### Q5. What is supervised learning?
**Answer**: A learning paradigm where algorithms train on labeled input-output pairs $(x_i, y_i)$ to learn a mapping function $f(x)$ that accurately predicts labels for unseen inputs.

#### Q6. What is unsupervised learning?
**Answer**: A learning paradigm where models uncover hidden patterns, cluster groupings, or latent distributions from unlabeled input features $\{x_i\}$ without teacher supervision.

#### Q7. What is reinforcement learning?
**Answer**: An agent-based learning framework where an autonomous agent learns optimal actions through sequential trial and error by interacting with an environment to maximize cumulative reward signals.

#### Q8. What is a dataset?
**Answer**: A structured collection of empirical records, instances, or observations organized into features (columns) and samples (rows) used for statistical analysis and model training.

#### Q9. What are features and labels?
**Answer**: Features ($X$) are the independent measurable variables or input attributes; Labels ($y$) are the ground-truth target outcomes or dependent variables being predicted.

#### Q10. What is the difference between training data and testing data?
**Answer**: Training data ($\approx 80\%$) is used to fit and optimize model parameters; Testing data ($\approx 20\%$) is strictly held out to evaluate unbiased out-of-sample generalization performance.

#### Q11. What is a machine learning model?
**Answer**: A model is the trained mathematical artifact (e.g., weights and decision rules) produced after an algorithm completes learning on a dataset, capable of generating predictions on new inputs.

#### Q12. What is an algorithm in Machine Learning?
**Answer**: An algorithm is the systematic optimization procedure or mathematical recipe (e.g., Gradient Descent, Backpropagation, OLS) used to train a model from data.

#### Q13. What is prediction?
**Answer**: Prediction ($\hat{y}$) is the output value (discrete class label or continuous number) generated by a trained model when evaluated on an unseen feature vector $x$.

#### Q14. What is classification?
**Answer**: Classification is a supervised learning task where the target output variable $y$ is a discrete categorical class (e.g., binary: spam vs. ham; multiclass: digit 0–9).

#### Q15. What is regression?
**Answer**: Regression is a supervised learning task where the target variable $y$ is a continuous real-valued quantitative number (e.g., predicting house prices, temperature, stock values).

---

### 5. Learning Paradigms & Machine Learning Types

#### Q1. Explain supervised learning with an example.
**Answer**: Supervised learning trains on labeled data $(X, y)$ to map inputs to targets; an example is email spam classification, where models learn from thousands of pre-labeled spam and legitimate emails.

#### Q2. Explain unsupervised learning with an example.
**Answer**: Unsupervised learning discovers latent structure in unlabeled data without guidance; an example is customer market segmentation, where customers are grouped by purchasing behavior without predefined segment tags.

#### Q3. Explain reinforcement learning with an example.
**Answer**: Reinforcement learning optimizes sequential decision-making via environmental rewards; an example is an autonomous grid navigation robot that receives $+10$ for reaching the goal and $-10$ for hitting hazard pits.

#### Q4. What is semi-supervised learning?
**Answer**: A paradigm that combines a small set of labeled data with a large volume of inexpensive unlabeled data to build stronger models at lower labeling costs.

#### Q5. What is the difference between classification and regression?
**Answer**: Classification predicts discrete categorical class labels ($y \in \{C_1, C_2\}$), whereas regression predicts continuous numerical values ($y \in \mathbb{R}$).

#### Q6. Give examples of supervised learning algorithms.
**Answer**: Linear Regression, Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, Naive Bayes, and Gradient Boosting.

#### Q7. Give examples of unsupervised learning algorithms.
**Answer**: $K$-Means, DBSCAN, Hierarchical Clustering, Principal Component Analysis (PCA), Gaussian Mixture Models (GMM), and Apriori.

#### Q8. What is clustering?
**Answer**: Clustering is the unsupervised task of grouping similar data points into clusters such that intra-cluster similarity is maximized and inter-cluster similarity is minimized.

#### Q9. What is dimensionality reduction?
**Answer**: It is the transformation of high-dimensional data into a lower-dimensional representation while preserving essential variance, topological structure, or class separability.

#### Q10. What is the difference between batch learning and online learning?
**Answer**: Batch (offline) learning trains on the complete dataset simultaneously and requires full retraining for updates; Online (incremental) learning updates model weights continuously sample-by-sample as data streams in.

---

### 6. Model Evaluation & Error Metrics Rapid Questions

#### Q1. What is accuracy?
**Answer**: Accuracy is the proportion of total correct predictions over all instances: $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$.

#### Q2. What is precision?
**Answer**: Precision is the ratio of true positive predictions to total predicted positives: $\text{Precision} = \frac{TP}{TP + FP}$ (crucial when false positives are expensive).

#### Q3. What is recall?
**Answer**: Recall (Sensitivity) is the ratio of true positive predictions to all actual positive cases: $\text{Recall} = \frac{TP}{TP + FN}$ (crucial when false negatives are dangerous).

#### Q4. What is F1-score?
**Answer**: The $F_1$-score is the harmonic mean of precision and recall: $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$, penalizing extreme imbalances between them.

#### Q5. What is a confusion matrix?
**Answer**: A cross-tabulation table contrasting actual target labels against predicted labels that details counts of True Positives, True Negatives, False Positives, and False Negatives.

#### Q6. What are True Positive and True Negative?
**Answer**: A True Positive ($TP$) is an actual positive instance correctly predicted as positive; a True Negative ($TN$) is an actual negative instance correctly predicted as negative.

#### Q7. What are False Positive and False Negative?
**Answer**: A False Positive ($FP$, Type I Error) is an actual negative misclassified as positive; a False Negative ($FN$, Type II Error) is an actual positive misclassified as negative.

#### Q8. What is Mean Squared Error (MSE)?
**Answer**: MSE measures the average squared difference between actual and predicted values: $\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2$, heavily penalizing large outliers.

#### Q9. What is Mean Absolute Error (MAE)?
**Answer**: MAE measures the average absolute difference between actual and predicted values: $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$, offering robustness against extreme outliers.

#### Q10. What is Root Mean Squared Error (RMSE)?
**Answer**: RMSE is the square root of MSE ($\sqrt{\text{MSE}}$); it penalizes large errors while expressing error in the same physical units as the target variable.

#### Q11. What is R-squared?
**Answer**: $R^2$ (Coefficient of Determination) quantifies the proportion of target variance explained by predictor features: $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$.

#### Q12. Why do we evaluate a machine learning model?
**Answer**: To assess generalization on out-of-sample data, diagnose underfitting/overfitting, compare competing models objectively, and ensure safety before deployment.

---

### 7. Machine Learning Algorithms Rapid Questions

#### Q1. What is Linear Regression?
**Answer**: A supervised regression algorithm that models a linear relationship between features $X$ and a continuous target $y$ by minimizing squared residuals via Ordinary Least Squares.

#### Q2. What is Logistic Regression?
**Answer**: A supervised classification algorithm that passes a linear combination of inputs through the logistic sigmoid function to predict class probabilities bounded between 0 and 1.

#### Q3. What is Decision Tree?
**Answer**: A non-parametric model that recursively partitions the feature space using orthogonal axis-aligned splits based on Gini Impurity or Information Gain to make leaf predictions.

#### Q4. What is Random Forest?
**Answer**: An ensemble algorithm that trains multiple de-correlated decision trees in parallel using Bootstrap Aggregating (Bagging) and random feature subsets, aggregating predictions by voting or averaging.

#### Q5. What is K-Nearest Neighbors (KNN)?
**Answer**: A lazy, instance-based non-parametric algorithm that classifies a query point based on the majority vote of its $k$ closest neighbors in feature space using Euclidean distance.

#### Q6. What is K-Means Clustering?
**Answer**: An unsupervised iterative partitioning algorithm that alternates between assigning points to the nearest cluster centroid and recomputing centroids as cluster means to minimize Within-Cluster Sum of Squares.

#### Q7. What is Support Vector Machine (SVM)?
**Answer**: A supervised algorithm that constructs an optimal separating hyperplane that maximizes the geometric margin between distinct classes, utilizing the kernel trick for non-linear data.

#### Q8. What is Naive Bayes?
**Answer**: A probabilistic classifier based on Bayes' Theorem that assumes all features are conditionally independent given the class label ($P(X|Y) = \prod P(x_i|Y)$).

#### Q9. What is the Apriori algorithm?
**Answer**: An association rule mining algorithm that identifies frequent itemsets in transactional databases using the anti-monotonicity principle (all subsets of a frequent itemset must also be frequent).

#### Q10. What is Principal Component Analysis (PCA)?
**Answer**: An unsupervised linear transformation that projects data onto orthogonal eigenvectors of the sample covariance matrix to reduce dimensions while preserving maximal variance.

#### Q11. What is Gradient Descent?
**Answer**: A first-order iterative optimization algorithm that minimizes a differentiable loss function $\mathcal{L}(\theta)$ by updating parameters in the opposite direction of the gradient: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$.

#### Q12. What is an Artificial Neural Network?
**Answer**: A computational network composed of interconnected artificial neurons across input, hidden, and output layers that learns complex non-linear mappings via backpropagation.

#### Q13. What is Ensemble Learning?
**Answer**: A meta-learning paradigm that combines predictions from multiple individual base models (e.g., via Bagging or Boosting) to achieve superior accuracy and lower generalization error.

#### Q14. What is the difference between Bagging and Boosting?
**Answer**: Bagging trains independent base models in parallel on bootstrap samples to reduce variance (e.g., Random Forest); Boosting trains weak learners sequentially to correct previous errors and reduce bias (e.g., Gradient Boosting).

#### Q15. What is XGBoost?
**Answer**: Extreme Gradient Boosting; an optimized, highly scalable gradient boosted decision tree framework featuring second-order Taylor expansions, built-in $L_1/L_2$ regularization, and cache-aware hardware speedups.

---

## Repository Structure

```
.
├── README.md               # Comprehensive Laboratory Manual, practical guide & viva Q&A suite
├── allcodes.py             # Complete, executable Python scripts for Experiments 01 - 11
├── untitled.txt            # Lab viva-voce reference questions and revision material
├── dataset/                # Dataset store (titanic.csv, housing.csv, auto_mpg.csv, etc.)
├── pt1.ipynb               # Practical 01: Basic Feature Engineering Techniques
├── pt2.ipynb               # Practical 02: Dimensionality Reduction using PCA and LDA
├── pt3.ipynb               # Practical 03: Multiple Linear Regression
├── pt4.ipynb               # Practical 04: Polynomial Regression & Bias-Variance Trade-off
├── pt5.ipynb               # Practical 05: Ridge and Lasso Regularization
├── pt6.ipynb               # Practical 06: Logistic Regression Binary Classification
├── pt7.ipynb               # Practical 07: Support Vector Machines (SVM) with Kernels
├── pt8.ipynb               # Practical 08: K-Means Clustering & Optimal Clusters
├── pt9.ipynb               # Practical 09: Density-Based Spatial Clustering (DBSCAN)
├── pt10.ipynb              # Practical 10: Ensemble Learning: Random Forest & Boosting
└── pt11.ipynb              # Practical 11: Reinforcement Learning: Q-Learning Grid World
```

---

## Getting Started & Running Notebooks

### Prerequisites
Ensure Python 3.9+ is installed along with the required scientific computing libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Running Practical Implementations
Execute the complete unified experiment script:
```bash
python allcodes.py
```

To run individual interactive laboratory notebooks:
```bash
jupyter notebook
```
Navigate to any notebook ([`pt1.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt1.ipynb) through [`pt7.ipynb`](file:///C:/Users/elite/OneDrive/Documents/mllab/pt7.ipynb)) to view step-by-step code execution, diagnostic charts, and tabular outputs.
