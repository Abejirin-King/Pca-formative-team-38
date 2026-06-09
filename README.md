# PCA Formative Assignment Team 38

## Overview

This project implements **Principal Component Analysis (PCA)** from scratch as part of an Advanced Linear Algebra formative assignment.

The objective is to demonstrate a clear understanding of the mathematical foundations of PCA by manually performing each step of the algorithm using NumPy, rather than relying on machine learning libraries such as Scikit-learn.

The dataset contains socio-economic and demographic indicators from African countries, including GDP per capita, population, life expectancy, fertility rate, urban population percentage, unemployment rate, infant mortality, and school enrollment rates.

---

## Team Members

**King Obafemi Abejirin**

**Janviere Munezero**

Repository: https://github.com/Abejirin-King/Pca-formative-team-38

---

## Learning Objectives

This project demonstrates:

- Data standardization
- Covariance matrix computation
- Eigendecomposition
- Principal component selection
- Dimensionality reduction
- Data visualization before and after PCA
- Interpretation of explained variance

---

## Dataset Features

The dataset contains the following variables:

- Country
- GDP_per_Capita_USD
- Population_Millions
- Life_Expectancy
- Fertility_Rate
- Urban_Population_Pct
- Unemployment_Pct
- Infant_Mortality
- School_Enrollment_Pct

Missing values are replaced using the mean of each feature before analysis.

---

## PCA Workflow

### 1. Data Loading

The dataset is loaded from:

```python
african_pca_dataset.csv

## Data Standardization

Before applying PCA, all numerical features were standardized to ensure that each variable contributed equally to the analysis. Since the dataset contains measurements on different scales, standardization prevents features with larger values from dominating the results. Each feature was transformed using z-score normalization so that it has a mean of 0 and a standard deviation of 1.

### Standardization Formula

\[
z = \frac{x - \mu}{\sigma}
\]

Where:

- **x** = original value
- **μ** = mean of the feature
- **σ** = standard deviation of the feature

---

## Covariance Matrix Computation

After standardization, a covariance matrix was computed to analyze the relationships between variables in the dataset.

```python
cov_matrix = np.cov(standardized_data, rowvar=False)
```

The covariance matrix is important because it:

- Identifies correlations between features.
- Reveals how variables change relative to one another.
- Provides the foundation for determining the principal components used in PCA.

---

## Eigendecomposition

The covariance matrix was decomposed into eigenvalues and eigenvectors.

```python
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
```

- **Eigenvalues** indicate how much variance is explained by each principal component.
- **Eigenvectors** define the direction of the principal components.

Together, they determine the most informative directions in the dataset.

---

## Principal Component Selection

The eigenvalues and eigenvectors were sorted in descending order based on the magnitude of the eigenvalues.

```python
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]
```

This ensures that the principal components explaining the most variance are selected first.

---

## Dimensionality Reduction

The first four principal components were selected for the final transformation.

```python
num_components = 4
principal_components = sorted_eigenvectors[:, :num_components]
```

The standardized dataset was then projected onto the selected principal components.

```python
reduced_data = np.dot(standardized_data, principal_components)
```

This reduced the dataset from eight dimensions to four while preserving most of the important information.

---

## Results

The PCA transformation produced a reduced dataset with the following shape:

```text
Reduced Data Shape: (20, 4)
```

The original dataset dimensions:

```text
20 × 8
```

were reduced to:

```text
20 × 4
```

while retaining the majority of the dataset's variance.

---

## Visualization

The notebook includes visualizations comparing the data before and after PCA.

The plots help illustrate:

- The structure of the original standardized data.
- The reduced representation after PCA.
- How dimensionality reduction preserves important patterns while simplifying the dataset.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

---

## Key Concepts Demonstrated

- Data Standardization
- Covariance Analysis
- Eigenvalues and Eigenvectors
- Principal Component Analysis (PCA)
- Dimensionality Reduction
- Feature Transformation
- Variance Preservation
- Data Visualization

---

## Conclusion

This project demonstrates the implementation of Principal Component Analysis (PCA) from first principles using linear algebra concepts. Through data standardization, covariance matrix computation, eigendecomposition, principal component selection, and projection into a lower-dimensional space, the project showcases how PCA can effectively reduce dimensionality while preserving the most significant information contained within a dataset.

