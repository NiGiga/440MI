# Beamer Presentations

This directory contains the 36-class lecture sequence for the course `440MI / 305SM`, written in Beamer and organized in three macro-blocks: **Python** (instrumental programming, Classes 1-12) → **Data-Driven** (data types, representation, and modeling, Classes 13-20) → **MLOps** (software engineering and operations, Classes 21-36). It draws on:

- notebooks for EDA, feature engineering, model training, Neptune tuning, and incremental learning
- Flask and Streamlit demos for fraud detection and streaming ML
- the synthetic pasteurization case study
- tutorials for Docker and GitHub Actions

## Files

- `common.tex`: shared Beamer styling and macros
<<<<<<< HEAD
- `class01_intro_course.tex` to `class36_presentation_proposals_2.tex`: lecture decks, numbered in delivery order

Three sessions — `class21_intervention_cybertec.tex`, `class24_intervention_yesalps.tex`, and `class33_intervention_sbroiavacca.tex` — are minimal placeholders for external guest speakers. Their material is provided by the guests, not authored here.
=======
- `class01_intro_course.tex` to `class36_presentation_proposals.tex`: lecture decks, numbered in delivery order

Three sessions — `class21_intervention_cybertec.tex`, `class24_intervention_yesalps.tex`, and `class34_intervention_sbroiavacca.tex` — are minimal placeholders for external guest speakers. Their material is provided by the guests, not authored here.
>>>>>>> upstream/HEAD

## Suggested workflow

Compile each deck from this directory, for example:

```bash
cd presentations
pdflatex class01_intro_course.tex
```

If your TeX setup needs multiple runs for links or tables of contents, run `pdflatex` twice.

## Design choices

- The Python block (1-12) restores dedicated programming fundamentals — basic syntax, control flow, functions, files, OOP — that had previously been compressed into fewer sessions, drawing on the legacy `delete/2024/lessons/DataDriven_L1`-`L7` material.
<<<<<<< HEAD
- The Data-Driven block (13-20) is new: it opens with two theory-heavy sessions on data types (tabular, time series, spectral/signal, image) and on structured/semi-structured/unstructured data with encoding and embeddings, before feature engineering, modeling, online learning, and serving.
- The MLOps block (21-36) merges what used to be split or duplicated sessions (software engineering intro + activities; Agile theory + toolkit; MLOps practice + continuous training; monitoring + scalability; the two "practical project" entries) to make room for the Data-Driven block without growing the total session count.
- Class 17 (`class17_version_control.tex`) is a dedicated session on version control (Git fundamentals, branching, merge conflicts, pull requests) -- introduced right after modeling and evaluation (Class 16) so students carry the habit into their growing codebase and, later, their project repositories. Evaluation and decision-making, previously a separate session, is folded into Class 16 (`class16_ml_modeling1_experiment_tracking.tex`) alongside modeling and experiment tracking.
- Classes 29 (MLOps practice / continuous training) and 31 (continuous monitoring / scalability) each include a "Practical Component" frame flagging that no runnable exercise exists yet in this repository for that topic — a candidate for a follow-up project or submodule.
- Real examples are drawn from this repository instead of relying only on textbook slides.
- Each deck ends with a curated reading slide to support deeper study.
=======
- The Data-Driven block (13-20) opens with data types and representation, then includes an encoding lab (15), feature engineering (16), Git (17), modeling (18), online learning and concept drift (19), and a streaming/drift lab (20).
- Model deployment and prediction serving is Class 30, ahead of Docker and CI/CD (31) and monitoring/scalability (32). The two proposal-presentation sessions merge into one final Class 36.
- Class 17 (`class17_version_control.tex`) covers Git fundamentals, branching, merge conflicts, and pull requests. It now precedes modeling (Class 18), so students can version the experiments they build in that session. Evaluation and decision-making remain folded into Class 18 (`class18_ml_modeling1_experiment_tracking.tex`) alongside modeling and experiment tracking.
- The encoding lab (Class 15) gives students practice selecting and applying encoders with a preprocessing pipeline, building on Class 14's representation concepts.
- Class 15's activity brief is in `documents/Class15_Encoding_Embedding_Activity.md`, with a worked notebook in `notebooks/Class15_Encoding_Embedding_Solution.ipynb` and its prepared all-rows exploratory fixture in `notebooks/Class15_Titanic_Imputed.csv`. The train/test split is introduced in Class 18 before model evaluation.
- Classes 29 (MLOps practice / continuous training) and 32 (continuous monitoring / scalability) each include a "Practical Component" frame flagging that no runnable exercise exists yet in this repository for that topic — a candidate for a follow-up project or submodule.
- Real examples are drawn from this repository instead of relying only on textbook slides.
- Lecture decks end with curated reading; practice decks close with a deliverable or reflection prompt.
>>>>>>> upstream/HEAD

