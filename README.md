# M25DE1011 CSL7110 Assignment 3

This repository contains the complete submission for the CSL7110 recommender-systems assignment built on the MovieLens dataset. The project combines classical recommenders, matrix factorization, a hybrid meta-model, a neural recommender, a simple reinforcement-learning setup, and multiple explainability methods in one executed notebook.

## Submission Contents

- `recommender_systems_assignment_solution.ipynb`: fully executed notebook with outputs, tables, plots, and discussion for all assignment tasks.
- `full_dataset_solution.py`: reusable implementation for data preparation, model training, evaluation, recommendation, and explanation helpers.
- `run_notebook.py`: script to execute the notebook in place without depending on a `jupyter` command being available on `PATH`.
- `requirements.txt`: Python dependencies needed to reproduce the notebook and helper module.
- `M25DE1011_CSL7110_Assignment3.pdf`: submitted assignment PDF.

## Project Goal

The assignment studies how different recommendation strategies behave on the same movie-rating dataset. The notebook starts with content-based ideas, moves into collaborative filtering and factorization methods, then expands into hybrid, neural, reinforcement-learning, and explainability-oriented approaches. The goal is not only to compare prediction quality, but also to discuss interpretability and practical tradeoffs.

## Dataset Overview

The workflow uses the full MovieLens `ml-latest` dataset and then extracts a denser modeling subset so heavier algorithms can be executed end to end in a practical amount of time.

- Full dataset profile from the executed notebook: 330,975 users, 86,537 movies, 33,832,162 ratings, average rating 3.5425, sparsity 99.8819%.
- Modeling core used for training and comparison: 500 users, 800 movies, 308,654 ratings.
- Temporal split used for evaluation: 215,632 train rows, 31,091 validation rows, and 61,931 test rows.

The raw downloaded MovieLens files are intentionally not committed because they are too large for a normal GitHub repository. The code downloads and prepares them locally when needed.

## Summarized Concepts

The notebook covers the following ideas and explains how each one works:

- TF-IDF content-based recommendation: converts movie genres into text-like features and recommends movies with similar content.
- User-profile content recommendation: builds a preference profile for each user from previously liked genres and scores new movies against that profile.
- User-based collaborative filtering: recommends items from users with similar rating behavior.
- Item-based collaborative filtering: recommends movies similar to those a user already rated highly.
- Manual SVD: factorizes the sparse user-item rating matrix into latent user and item factors.
- Surprise SVD: uses the `surprise` library to train a tuned matrix-factorization recommender more efficiently.
- Hybrid meta-model: combines signals from content-based and collaborative methods with movie popularity and user preference statistics.
- Neural recommender: learns user and item representations through a two-tower neural network.
- Reinforcement-learning recommender: explores how repeated interaction and reward feedback can shape recommendation policy.
- Explainability modules: uses direct feature overlap, neighborhood-based explanations, SHAP, and LIME to justify recommendations.

## Key Results From The Executed Notebook

The final notebook comparison shows that different models excel on different metrics:

- Best RMSE: `Surprise SVD` with `0.8115`
- Best precision@10: `Manual SVD` with `0.4200`
- Best recall@10: `Manual SVD` with `0.1703`
- Best overall balance in the comparison table: `Item-CF (k=40)` with RMSE `0.8178`, precision@10 `0.3933`, and recall@10 `0.1564`

This result fits the usual recommender-system tradeoff: matrix factorization gives strong rating prediction accuracy, while neighborhood methods can remain highly competitive for top-N recommendation quality and interpretability.

## Repository Structure

```text
.
|-- M25DE1011_CSL7110_Assignment3.pdf
|-- README.md
|-- full_dataset_solution.py
|-- recommender_systems_assignment_solution.ipynb
|-- requirements.txt
`-- run_notebook.py
```

## How To Reproduce

1. Create and activate a Python 3.12 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Execute the notebook in one of these ways:

```bash
python run_notebook.py
```

or open `recommender_systems_assignment_solution.ipynb` in JupyterLab and run it top to bottom.

## Notes

- The notebook is already committed in executed form for easy review.
- Large MovieLens data files are ignored by Git and recreated locally by the code.
- The helper module sets `LOKY_MAX_CPU_COUNT=1` to avoid multiprocessing issues on some Windows setups.

## Repository Purpose

This repository is prepared as the GitHub submission bundle for `M25DE1011_CSL7110_Assignment3`, containing the executed notebook, supporting code, improved documentation, dependency list, and the assignment PDF.
