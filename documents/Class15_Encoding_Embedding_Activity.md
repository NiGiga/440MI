# Class 15 Activity: Encoding and Embedding with Titanic

**Data-driven Systems Engineering (440MI / 305SM)**  
**Duration:** 90–120 minutes  
**Prerequisite:** Class 14: data representation, categorical encoding, embeddings, and PCA  
**Dataset:** [https://raw.githubusercontent.com/sbarbonjr/440MI/main/notebooks/Class15_Titanic_Imputed.csv](https://raw.githubusercontent.com/sbarbonjr/440MI/main/notebooks/Class15_Titanic_Imputed.csv) (`Class15_Titanic_Imputed.csv`), 891 rows

## Purpose

In this activity you will represent the categorical features of the Titanic passengers (`sex`, `pclass`, `embarked`) in numeric form and judge how good each representation is. You will:

- encode the categories with one-hot and ordinal encoders and compare the resulting feature matrices;
- check each representation for validity, unseen-value behavior, size, and sparsity;
- explore how the categories relate to survival with group rates, chi-squared, and Spearman diagnostics;
- build a PCA projection of embarkation-port profiles and interpret its axes.

The goal is to learn which checks tell you a representation is sound and what each check can and cannot support. Whether an encoding improves a predictive model is a separate question, tested in a later class with a controlled model comparison.

The CSV is an imputed fixture prepared for this activity and contains all 891 passengers, so the analysis uses every row and needs no train/test split. Class 16 discusses the original missing values and imputation choices. Class 18 introduces the split before fitting preprocessing; use raw data for that evaluation, not this fixture.

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

## Libraries

The prepared CSV is all you need; no separate Titanic download is required. Load it with `pandas`; `numpy` and `matplotlib` cover the basic arithmetic and plots. Three libraries are likely to be new, so check their documentation before you start:

| Library | Use in this activity |
|---|---|
| `scikit-learn` | `OneHotEncoder` and `OrdinalEncoder` to encode categories, `StandardScaler` and `PCA` to project profiles |
| `scipy.stats` | `chi2_contingency` for categorical association and `spearmanr` for rank association |
| `seaborn` | `heatmap` to display Spearman correlations and chi-squared p-values |

## Activity

### 1. Choose a representation

Load the CSV and confirm it has 891 rows. For each feature, record its type, a candidate representation, and the assumption that representation makes.

| Feature | Structure | Candidate to inspect |
|---|---|---|
| `sex` | Nominal, two categories | One-hot |
| `embarked` | Nominal, three ports | One-hot |
| `pclass` | Ordered categories | Compare one-hot with ordinal codes 1, 2, 3 |

Ask what the ordinal codes imply about the distance between passenger classes. A numeric label is not automatically a meaningful distance.

### 2. Inspect the encoded matrix

1. Fit a `OneHotEncoder` on `sex` and `embarked`, configured to ignore unknown categories. In a later predictive workflow, split first and fit the encoder on training data only.
2. Print the learned categories, the output column names, and the matrix shape.
3. Compute the matrix density: the share of non-zero entries.
4. Transform one synthetic row with an unseen port, such as `NEW_PORT`, and check that it is ignored safely while the known columns stay unchanged.
5. Encode `pclass` twice, with `OneHotEncoder` and with `OrdinalEncoder` using the order 1, 2, 3, and compare the number of columns each produces.

Record learned categories, output names, number of columns, matrix density, and unseen-category behavior. These checks assess validity, robustness, and complexity; they do not measure model performance.

### 3. Explore association with survival

1. Build count and survival-rate tables by `embarked` and by `pclass`, keeping the group sizes beside the rates.
2. Run a chi-squared independence test between `embarked` and `survived`, and report the statistic, degrees of freedom, and p-value.
3. Compute Spearman's rank correlation between `fare` and `survived` (coded 0/1).
4. Build two heatmaps from all rows: Spearman correlations among `age`, `fare`, `sibsp`, `parch`, and `survived`, and chi-squared p-values among `sex`, `pclass`, `embarked`, and `survived`.

The chi-squared p-value tests whether the observed distributions differ from independence; it does not measure how strong the association is. Spearman's rho gives direction and strength of a monotonic rank association. Read the two heatmaps on their own scales and do not compare colors across them. Both are exploratory diagnostics on these 891 passengers; neither establishes causality or proves that the feature or its encoding improves held-out predictions. Do not calculate Pearson correlation on arbitrary numeric port codes such as C=0, Q=1, S=2.

### 4. Build a PCA projection of port profiles

1. Create `log_fare` with `log1p`, `family_size` as `sibsp + parch + 1`, and a 0/1 female indicator.
2. For each port, compute the profile: mean age, mean log fare, mean passenger class, mean family size, and female share. Keep `survived` out of the profile.
3. Standardize the profile columns, because they use different scales.
4. Fit a two-component PCA on the three standardized profiles.
5. Plot the profiles on PC1 and PC2, label each port, and put the explained variance on the axes.

The points show differences among these aggregate profiles. The solution notebook also projects the original features as arrows and reports their correlations with PC1 and PC2. The plot is not a supervised embedding, does not use `survived`, and does not show that the coordinates improve a classifier.

With only three port profiles, two centered principal components can explain all profile variation by construction. This is an exploratory visualization, not strong validation. Similarly, applying PCA directly to three one-hot port columns is not useful compression: after centering, the matrix has at most two independent dimensions.

### 5. State what the evidence supports

Complete both statements:

1. “The representation is suitable for this exercise because …” Support this with category coverage, unknown-value handling, output size, and/or the profile visualization.
2. “We cannot yet claim that it improves the model because …” Explain that predictive contribution requires a downstream model comparison.

In Class 18, test contribution by creating a train/test split and keeping it, the estimator, and metric fixed across comparisons. Compare a baseline without the feature, one-hot encoding, and any candidate embedding. Fit learned transformations on training data only and evaluate on held-out data. Start from raw data, not this all-rows imputed activity fixture. Compare metric changes and added complexity rather than treating one score as conclusive.

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
