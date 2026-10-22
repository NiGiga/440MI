# Class 15 Activity: Encoding and Embedding with Titanic

**Data-driven Systems Engineering (440MI / 305SM)**  
**Duration:** 90–120 minutes  
**Prerequisite:** Class 14: data representation, categorical encoding, embeddings, and PCA  
**Dataset:** [Prepared Titanic train/test fixture](../notebooks/Class15_Titanic_Imputed.csv) (`Class15_Titanic_Imputed.csv`)

## Purpose

Choose and inspect representations for categorical features, then distinguish evidence about representation quality from evidence that an encoding improves a predictive model. This activity does not train a predictive model. Testing predictive contribution requires a controlled model comparison, which follows after modeling is introduced.

The CSV is an imputed fixture prepared for this activity. Its `partition` column identifies `train` and `test` rows. Use the training partition for the encoding diagnostics, survival summaries, and PCA profiles. The test partition is included to make the split explicit; do not use it to fit encoders or calculate the exercise summaries. Class 16 discusses the original missing values and imputation choices.

## Learning outcomes

By the end of the activity, you should be able to:

- distinguish nominal categories from ordered categories and justify an encoding choice;
- inspect category coverage, unseen-value behavior, dimensionality, and sparsity;
- use chi-squared and Spearman as exploratory feature–target association diagnostics;
- explain why association with `survived` does not establish predictive contribution;
- build and interpret a PCA projection of embarkation-port profiles without calling it a supervised embedding;
- state what additional experiment is needed to test predictive contribution.

## Data and scope

Use `sex`, `pclass`, and `embarked` as categorical features. Use `age`, `fare`, `sibsp`, and `parch` only to build aggregate embarkation profiles. The target is `survived`.

Do not use `alive`, which restates the target, or identifiers and post-outcome fields. Keep `survived` out of every feature transformation and PCA profile.

## Python libraries and functions

The prepared CSV is all you need; no separate Titanic download is required. The exercises use these libraries:

| Library | Main functions used | Purpose |
|---|---|---|
| `pandas` | `read_csv`, `groupby`, `agg` | Load and summarize tabular data |
| `scikit-learn` | `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `PCA` | Encode categories and project profiles |
| `scipy.stats` | `chi2_contingency`, `spearmanr` | Test categorical association and measure rank association |
| `numpy` | `log1p`, `count_nonzero` | Safe log transform and matrix density |
| `matplotlib` | `subplots`, `scatter`, `annotate`, `axhline`, `axvline`, `show` | Plot and label PCA profiles |
| `seaborn` | `heatmap` | Visualize Spearman correlations and chi-squared p-values |

Start with this setup cell. Adjust the CSV path if you placed the file somewhere else.

\newpage

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# Download the prepared CSV from the course materials and set this path.
df = pd.read_csv("Class15_Titanic_Imputed.csv")
train = df.loc[df["partition"] == "train"].copy()
test = df.loc[df["partition"] == "test"].copy()

print(train.shape, test.shape)
print(train[["sex", "pclass", "embarked", "survived"]].head())
```

## Activity

### 1. Choose a representation

For each feature, record its type, a candidate representation, and the assumption that representation makes.

| Feature | Structure | Candidate to inspect |
|---|---|---|
| `sex` | Nominal, two categories | One-hot |
| `embarked` | Nominal, three ports | One-hot |
| `pclass` | Ordered categories | Compare one-hot with ordinal codes 1, 2, 3 |

Ask what the ordinal codes imply about the distance between passenger classes. A numeric label is not automatically a meaningful distance.

\newpage

### 2. Inspect the encoded matrix

Fit encoders using training data only. `OneHotEncoder` learns category columns from `fit`; `handle_unknown="ignore"` lets `transform` accept a category not seen during fitting. `get_feature_names_out` reports the resulting column names.

```python
nominal = ["sex", "embarked"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_nominal_train = ohe.fit_transform(train[nominal])

print(ohe.categories_)
print(ohe.get_feature_names_out(nominal))
print("shape:", X_nominal_train.shape)
print("density:", np.count_nonzero(X_nominal_train) / X_nominal_train.size)

# A synthetic unseen port is ignored safely; known columns remain unchanged.
unseen = pd.DataFrame({"sex": ["female"], "embarked": ["NEW_PORT"]})
print(ohe.transform(unseen))
```

Compare one-hot and ordinal representations for `pclass`. In this fixture, the class labels are 1, 2, and 3.

```python
pclass_onehot = OneHotEncoder(sparse_output=False)
X_class_onehot = pclass_onehot.fit_transform(train[["pclass"]])

pclass_ordinal = OrdinalEncoder(categories=[[1, 2, 3]])
X_class_ordinal = pclass_ordinal.fit_transform(train[["pclass"]])

print("one-hot columns:", pclass_onehot.get_feature_names_out(["pclass"]))
print("one-hot shape:", X_class_onehot.shape)
print("ordinal shape:", X_class_ordinal.shape)
```

#### Visualize pairwise association

Build two heatmaps from the training partition. The Spearman panel shows signed rank correlations for numeric features and the binary target. The chi-squared panel shows p-values for categorical-variable pairs; its color represents evidence against independence, not association strength.

```python
numeric = train[["age", "fare", "sibsp", "parch", "survived"]]
spearman_matrix = numeric.corr(method="spearman")

categorical = ["sex", "pclass", "embarked", "survived"]
p_values = np.full((len(categorical), len(categorical)), np.nan)
p_labels = np.full((len(categorical), len(categorical)), "", dtype=object)
for i, left in enumerate(categorical):
    for j in range(i + 1, len(categorical)):
        table = pd.crosstab(train[left], train[categorical[j]])
        _, p, _, _ = chi2_contingency(table, correction=False)
        p_values[i, j] = p_values[j, i] = p
        p_labels[i, j] = p_labels[j, i] = "p<.001" if p < .001 else f"p={p:.2f}"

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.heatmap(spearman_matrix, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, square=True, ax=axes[0])
axes[0].set_title("Spearman rho")
sns.heatmap(-np.log10(np.maximum(p_values, 1e-300)), annot=p_labels, fmt="",
            cmap="YlOrBr", vmin=0, vmax=10, square=True,
            mask=np.triu(np.ones_like(p_values, dtype=bool)), ax=axes[1])
axes[1].set_title("Chi-squared: p-values")
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.45)
plt.show()
```

Read the Spearman panel as direction and strength of monotonic association. Read the chi-squared panel as evidence against independence; a small p-value is not an effect size. Do not compare colors across the panels as if they shared one scale.
Record learned categories, output names, number of columns, matrix density, and unseen-category behavior. These checks assess validity, robustness, and complexity; they do not measure model performance.

\newpage

### 3. Explore association with survival

Start with category counts and survival rates. Then compare two feature–target association checks: chi-squared for two categorical variables, and Spearman's rank correlation for a numeric feature and the binary target (`survived` coded 0/1).

```python
port_summary = train.groupby("embarked")["survived"].agg(
    passengers="count",
    survival_rate="mean",
)
class_summary = train.groupby("pclass")["survived"].agg(
    passengers="count",
    survival_rate="mean",
)
print(port_summary)
print(class_summary)

# Categorical feature vs categorical target: test independence.
port_table = pd.crosstab(train["embarked"], train["survived"])
chi2, p_chi2, dof, expected = chi2_contingency(port_table, correction=False)
print(f"chi-square={chi2:.2f}, df={dof}, p-value={p_chi2:.4g}")

# Numeric feature vs binary target: rank-based monotonic association.
rho, p_spearman = spearmanr(train["fare"], train["survived"])
print(f"Spearman rho={rho:.2f}, p-value={p_spearman:.4g}")
```

The chi-squared p-value tests whether the observed categorical distributions differ from independence; it does not measure how strong the association is. Spearman's rho gives direction and strength of a monotonic rank association. Both are exploratory diagnostics on the training sample. Neither establishes causality or proves that the feature or its encoding improves held-out predictions. Do not calculate Pearson correlation on arbitrary numeric port codes such as C=0, Q=1, S=2.

\newpage

### 4. Build a PCA projection of port profiles

For each port, summarize training-set predictors: mean age, mean log fare, mean passenger class, mean family size, and female share. Standardize the profile columns before PCA because they use different scales. `PCA.fit_transform` learns the two principal directions and returns the profile coordinates.

```python
train["log_fare"] = np.log1p(train["fare"])
train["family_size"] = train["sibsp"] + train["parch"] + 1
train["is_female"] = (train["sex"] == "female").astype(int)

profile = train.groupby("embarked").agg(
    mean_age=("age", "mean"),
    mean_log_fare=("log_fare", "mean"),
    mean_pclass=("pclass", "mean"),
    mean_family_size=("family_size", "mean"),
    female_share=("is_female", "mean"),
)

features = profile.columns.tolist()
scaled = StandardScaler().fit_transform(profile[features])
pca = PCA(n_components=2)
coordinates = pca.fit_transform(scaled)

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(coordinates[:, 0], coordinates[:, 1], s=100)
for port, (x, y) in zip(profile.index, coordinates):
    ax.annotate(port, (x, y), xytext=(6, 5), textcoords="offset points")
ax.axhline(0, color="0.8", linewidth=0.8)
ax.axvline(0, color="0.8", linewidth=0.8)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_title("PCA of passenger profiles by embarkation port")
plt.tight_layout()
plt.show()

print("Variance explained:", pca.explained_variance_ratio_.sum())
```

The points show differences among these aggregate profiles. The solution notebook also projects the original features as arrows and reports their correlations with PC1 and PC2. The plot is not a supervised embedding, does not use `survived`, and does not show that the coordinates improve a classifier.

With only three port profiles, two centered principal components can explain all profile variation by construction. This is an exploratory visualization, not strong validation. Similarly, applying PCA directly to three one-hot port columns is not useful compression: after centering, the matrix has at most two independent dimensions.

\newpage

### 5. State what the evidence supports

Complete both statements:

1. “The representation is suitable for this exercise because …” Support this with category coverage, unknown-value handling, output size, and/or the profile visualization.
2. “We cannot yet claim that it improves the model because …” Explain that predictive contribution requires a downstream model comparison.

In Class 18, test contribution by keeping the train/test split, estimator, and metric fixed; compare a baseline without the feature, one-hot encoding, and any candidate embedding. Fit learned transformations on training data only and evaluate on held-out data. Compare metric changes and added complexity rather than treating one score as conclusive.

## Student deliverable

Submit a short notebook or worksheet containing:

1. an encoding decision table for `sex`, `embarked`, and `pclass`, plus the association heatmaps;
2. learned categories, feature names, dimensionality, density, and unseen-category result;
3. a port-by-survival table, chi-squared result, Spearman result, and cautious interpretation;
4. a labeled PCA profile plot and a note about its limitations;
5. one claim supported by today's evidence and one claim that must wait until modeling.

## Assessment guide

| Criterion | Meets expectations when… |
|---|---|
| Encoding choice | Nominal and ordered features are distinguished and assumptions are stated. |
| Representation checks | Categories, dimensions, density, and unknown-value behavior are inspected correctly. |
| Association | Chi-squared and Spearman are matched to suitable variable types and described as association, not model gain or causation. |
| PCA interpretation | The plot is described as an exploratory projection, not a supervised embedding. |
| Evidence boundary | Predictive contribution is reserved for a controlled model comparison. |

## Reference materials

- [Prepared Titanic CSV](../notebooks/Class15_Titanic_Imputed.csv)
- [Possible solution notebook](../notebooks/Class15_Encoding_Embedding_Solution.ipynb)
