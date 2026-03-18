# M25DE1011 CSL7110 Assignment 3

This repository contains the executed submission notebook for the recommender-systems assignment based on the MovieLens dataset.

## Repository Contents

- `recommender_systems_assignment_solution.ipynb`: fully executed notebook with outputs, tables, plots, and analysis for every task.
- `full_dataset_solution.py`: support module used by the notebook for the full-dataset workflow.
- `README.md`: project overview and run instructions.

## What The Notebook Covers

- Task 1: TF-IDF content-based movie recommendation
- Task 2: user-profile-based content recommendation
- Task 3: user-based collaborative filtering
- Task 4: item-based collaborative filtering
- Task 5: manual SVD
- Task 6: Surprise SVD
- Task 7: hybrid recommendation model
- Task 8: neural recommender
- Task 9: reinforcement-learning recommender
- Task 10: feature-based explanations
- Task 11: neighborhood-based explanations
- Task 12: LIME-based neural explanation
- Task 13: explainability evaluation and final comparison

## Dataset Note

The notebook loads the full `ml-latest` MovieLens dataset and then builds a dense evaluation core from that full dataset for the heavier recommendation models so the notebook can execute end to end in a practical amount of time.

The raw downloaded dataset is intentionally **not committed** to Git because it is too large for a normal GitHub repository. The notebook/download code will recreate it locally when needed.

## How To Run

1. Open `recommender_systems_assignment_solution.ipynb` in Jupyter.
2. Run the notebook from top to bottom.
3. Ensure the required Python packages are installed:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn torch shap lime scikit-surprise nbformat nbclient
```

## GitHub Repo

This repo was prepared for pushing to a dedicated GitHub repository named `M25DE1011_CSL7110_Assignment3`.
