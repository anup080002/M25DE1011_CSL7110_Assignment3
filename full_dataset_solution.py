from __future__ import annotations

import math
import os
import random
import re
import urllib.request
import warnings
import zipfile
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


class TwoTower(torch.nn.Module):
    def __init__(self, user_dim: int, item_dim: int, emb_dim: int = 32) -> None:
        super().__init__()
        self.user_branch = torch.nn.Sequential(
            torch.nn.Linear(user_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, emb_dim),
            torch.nn.ReLU(),
        )
        self.item_branch = torch.nn.Sequential(
            torch.nn.Linear(item_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, emb_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 3, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, user_x, item_x):
        user_vec = self.user_branch(user_x)
        item_vec = self.item_branch(item_x)
        merged = torch.cat([user_vec, item_vec, user_vec * item_vec], dim=1)
        return self.head(merged).squeeze(1)


class FullDatasetSolution:
    urls = {"latest": "https://files.grouplens.org/datasets/movielens/ml-latest.zip"}

    def __init__(self, data_dir: str | Path = "data", seed: int = 42) -> None:
        self.data_dir = Path(data_dir)
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.movies = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.ratings_model = pd.DataFrame()
        self.train = pd.DataFrame()
        self.val = pd.DataFrame()
        self.test = pd.DataFrame()

        self.global_mean = np.nan
        self.user_mean = {}
        self.movie_mean = {}
        self.train_seen = {}
        self.ground_truth = {}
        self.eval_users = []
        self.popular_items = []

        self.movie_lookup = None
        self.movie_id_to_row = {}
        self.tfidf = None
        self.tfidf_matrix = None
        self.user_profiles = {}

        self.rating_matrix = None
        self.user_means_series = None
        self.item_means_series = None
        self.user_similarity = None
        self.item_similarity = None
        self.svd_predictions = None
        self.surprise_model = None
        self.hybrid_model = None
        self.hybrid_features_train = None

        self.genre_cols = []
        self.movie_genre_features = None
        self.movie_feature_df = None
        self.movie_feature_scaled = None
        self.user_feature_df = None
        self.user_feature_scaled = None
        self.neural_model = None
        self.training_history = []

    def download_and_load(self) -> "FullDatasetSolution":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        archive = self.data_dir / "ml-latest.zip"
        folder = self.data_dir / "ml-latest"
        if not folder.exists():
            if not archive.exists():
                urllib.request.urlretrieve(self.urls["latest"], archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(self.data_dir)
        self.movies = pd.read_csv(folder / "movies.csv")
        self.ratings = pd.read_csv(folder / "ratings.csv")
        self.movies["genres"] = self.movies["genres"].replace("(no genres listed)", "Unknown")
        self.movies["genres_text"] = self.movies["genres"].str.replace("|", " ", regex=False)
        self.movies["genre_list"] = self.movies["genres"].str.split("|")
        self.movies["year"] = pd.to_numeric(self.movies["title"].str.extract(r"\((\d{4})\)$")[0], errors="coerce")
        self.movies["year"] = self.movies["year"].fillna(self.movies["year"].median())
        self.movie_lookup = self.movies.set_index("movieId")
        self.movie_id_to_row = pd.Series(self.movies.index, index=self.movies["movieId"]).to_dict()
        return self

    def full_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "users": self.ratings["userId"].nunique(),
                    "movies": self.movies["movieId"].nunique(),
                    "ratings": len(self.ratings),
                    "avg_rating": round(self.ratings["rating"].mean(), 4),
                    "sparsity_pct": round(
                        100
                        * (
                            1
                            - len(self.ratings)
                            / (self.ratings["userId"].nunique() * self.movies["movieId"].nunique())
                        ),
                        4,
                    ),
                }
            ]
        )

    def prepare_model_core(
        self,
        min_user_ratings: int = 80,
        min_item_ratings: int = 80,
        max_users: int = 2500,
        max_items: int = 3500,
    ) -> "FullDatasetSolution":
        df = self.ratings.copy()
        for _ in range(4):
            item_counts = df["movieId"].value_counts()
            df = df[df["movieId"].isin(item_counts[item_counts >= min_item_ratings].index)]
            user_counts = df["userId"].value_counts()
            df = df[df["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]
        top_users = df["userId"].value_counts().head(max_users).index
        df = df[df["userId"].isin(top_users)]
        top_items = df["movieId"].value_counts().head(max_items).index
        df = df[df["movieId"].isin(top_items)].copy()
        self.ratings_model = df
        return self

    def core_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "core_users": self.ratings_model["userId"].nunique(),
                    "core_movies": self.ratings_model["movieId"].nunique(),
                    "core_ratings": len(self.ratings_model),
                    "avg_rating": round(self.ratings_model["rating"].mean(), 4),
                }
            ]
        )

    def temporal_split(self, val_ratio: float = 0.1, test_ratio: float = 0.2, eval_user_limit: int = 150) -> "FullDatasetSolution":
        train_parts, val_parts, test_parts = [], [], []
        for _, group in self.ratings_model.sort_values("timestamp").groupby("userId"):
            n = len(group)
            test_n = max(1, int(math.ceil(n * test_ratio)))
            val_n = max(1, int(math.ceil(n * val_ratio)))
            if n - test_n - val_n < 1:
                val_n = max(0, n - test_n - 1)
            a, b = n - test_n - val_n, n - test_n
            train_parts.append(group.iloc[:a])
            if val_n > 0:
                val_parts.append(group.iloc[a:b])
            test_parts.append(group.iloc[b:])
        self.train = pd.concat(train_parts).reset_index(drop=True)
        self.val = pd.concat(val_parts).reset_index(drop=True)
        self.test = pd.concat(test_parts).reset_index(drop=True)
        self.global_mean = float(self.train["rating"].mean())
        self.user_mean = self.train.groupby("userId")["rating"].mean().to_dict()
        self.movie_mean = self.train.groupby("movieId")["rating"].mean().to_dict()
        self.train_seen = self.train.groupby("userId")["movieId"].apply(set).to_dict()
        self.ground_truth = {
            int(uid): set(grp.loc[grp["rating"] >= 4.0, "movieId"])
            for uid, grp in self.test.groupby("userId")
            if (grp["rating"] >= 4.0).any()
        }
        self.eval_users = sorted(self.ground_truth)[:eval_user_limit]
        self.popular_items = self.train["movieId"].value_counts().index.tolist()
        return self

    def clip(self, value: float) -> float:
        return float(np.clip(value, 0.5, 5.0))

    def fallback(self, user_id=None, movie_id=None) -> float:
        return self.clip(
            0.5 * self.user_mean.get(user_id, self.global_mean)
            + 0.5 * self.movie_mean.get(movie_id, self.global_mean)
        )

    def candidate_pool(self, user_id: int, limit: int = 500) -> list[int]:
        seen = self.train_seen.get(user_id, set())
        return [int(mid) for mid in self.popular_items if mid not in seen][:limit]

    def evaluate_rmse(self, predictor) -> float:
        preds = [predictor(row.userId, row.movieId) for row in self.test.itertuples(index=False)]
        return float(mean_squared_error(self.test["rating"], preds, squared=False))

    def evaluate_rmse_sample(self, predictor, max_rows: int = 300) -> float:
        sample = self.test.sample(min(max_rows, len(self.test)), random_state=self.seed)
        preds = [predictor(row.userId, row.movieId) for row in sample.itertuples(index=False)]
        return float(mean_squared_error(sample["rating"], preds, squared=False))

    def evaluate_ranking(self, recommender, k: int = 10, users: list[int] | None = None) -> dict:
        users = self.eval_users if users is None else users
        prec, rec, used = [], [], 0
        for user_id in users:
            truth = self.ground_truth.get(user_id, set())
            if not truth:
                continue
            recs = recommender(user_id, top_n=k)
            items = [row[0] for row in recs[:k]]
            if not items:
                continue
            hits = len(set(items) & truth)
            prec.append(hits / k)
            rec.append(hits / len(truth))
            used += 1
        return {
            "precision@10": float(np.mean(prec)) if prec else np.nan,
            "recall@10": float(np.mean(rec)) if rec else np.nan,
            "evaluated_users": used,
        }

    def resolve_title(self, title: str) -> int:
        exact = self.movies[self.movies["title"].str.lower() == title.lower()]
        if not exact.empty:
            return int(exact.index[0])
        fuzzy = self.movies[self.movies["title"].str.lower().str.contains(re.escape(title.lower()), regex=True)]
        if not fuzzy.empty:
            return int(fuzzy.index[0])
        raise KeyError(title)

    def build_content(self) -> "FullDatasetSolution":
        self.tfidf = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies["genres_text"])
        self.user_profiles = {int(uid): self.user_profile(int(uid)) for uid in self.train["userId"].unique()}
        return self

    def movie_to_movie(self, title: str, top_n: int = 5) -> pd.DataFrame:
        idx = self.resolve_title(title)
        sims = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        rows = []
        for j in np.argsort(sims)[::-1]:
            if j == idx:
                continue
            rows.append(
                {
                    "movieId": int(self.movies.at[j, "movieId"]),
                    "title": self.movies.at[j, "title"],
                    "genres": self.movies.at[j, "genres"],
                    "cosine_similarity": float(sims[j]),
                }
            )
            if len(rows) == top_n:
                break
        return pd.DataFrame(rows)

    def user_profile(self, user_id: int):
        history = self.train[self.train["userId"] == user_id]
        history = history[history["movieId"].isin(self.movie_id_to_row)].reset_index(drop=True)
        if history.empty:
            return None
        rows = history["movieId"].map(self.movie_id_to_row).to_numpy()
        weights = history["rating"].to_numpy()
        profile = np.asarray(self.tfidf_matrix[rows].multiply(weights[:, None]).sum(axis=0)).ravel()
        profile /= weights.sum() if weights.sum() else 1.0
        norm = np.linalg.norm(profile)
        if norm > 0:
            profile /= norm
        return csr_matrix(profile)

    def recommend_content(self, user_id: int, top_n: int = 10):
        profile = self.user_profiles.get(user_id)
        if profile is None:
            return []
        scores = linear_kernel(profile, self.tfidf_matrix).flatten()
        recs = []
        for idx in np.argsort(scores)[::-1]:
            movie_id = int(self.movies.at[idx, "movieId"])
            if movie_id in self.train_seen.get(user_id, set()):
                continue
            recs.append((movie_id, self.movies.at[idx, "title"], float(scores[idx])))
            if len(recs) == top_n:
                break
        return recs

    def predict_content(self, user_id: int, movie_id: int, top_k: int = 20) -> float:
        candidate = self.movie_id_to_row.get(movie_id)
        if candidate is None:
            return self.fallback(user_id, movie_id)
        history = self.train[(self.train["userId"] == user_id) & (self.train["movieId"] != movie_id)].copy()
        history = history[history["movieId"].isin(self.movie_id_to_row)].reset_index(drop=True)
        if history.empty:
            return self.fallback(user_id, movie_id)
        hist_idx = history["movieId"].map(self.movie_id_to_row).to_numpy()
        sims = linear_kernel(self.tfidf_matrix[candidate], self.tfidf_matrix[hist_idx]).flatten()
        mask = sims > 0
        if mask.sum() == 0:
            return self.fallback(user_id, movie_id)
        history = history.iloc[np.where(mask)[0]].reset_index(drop=True)
        sims = sims[mask]
        order = np.argsort(sims)[::-1][:top_k]
        return self.clip(np.average(history.loc[order, "rating"], weights=sims[order]))

    def build_cf(self) -> "FullDatasetSolution":
        self.rating_matrix = self.train.pivot_table(index="userId", columns="movieId", values="rating")
        self.user_means_series = self.rating_matrix.mean(axis=1)
        self.item_means_series = self.rating_matrix.mean(axis=0)
        user_centered = self.rating_matrix.sub(self.user_means_series, axis=0).fillna(0.0)
        item_centered = self.rating_matrix.sub(self.item_means_series, axis=1).fillna(0.0)
        self.user_similarity = pd.DataFrame(cosine_similarity(user_centered), index=self.rating_matrix.index, columns=self.rating_matrix.index)
        self.item_similarity = pd.DataFrame(cosine_similarity(item_centered.T), index=self.rating_matrix.columns, columns=self.rating_matrix.columns)
        return self

    def predict_user_cf(self, user_id: int, movie_id: int, k: int = 30) -> float:
        if user_id not in self.rating_matrix.index or movie_id not in self.rating_matrix.columns:
            return self.fallback(user_id, movie_id)
        movie_ratings = self.rating_matrix[movie_id].dropna()
        sims = self.user_similarity.loc[user_id].drop(user_id, errors="ignore").loc[movie_ratings.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        if sims.empty:
            return self.fallback(user_id, movie_id)
        denom = np.abs(sims.values).sum()
        if denom == 0:
            return self.fallback(user_id, movie_id)
        ratings = movie_ratings.loc[sims.index]
        means = self.user_means_series.loc[sims.index]
        pred = self.user_means_series.loc[user_id] + np.dot(sims.values, (ratings - means).values) / denom
        return self.clip(pred)

    def recommend_user_cf(self, user_id: int, top_n: int = 10, k: int = 30):
        if user_id not in self.rating_matrix.index:
            return []
        sims = self.user_similarity.loc[user_id].drop(user_id, errors="ignore")
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        neighbor_ratings = self.rating_matrix.loc[sims.index]
        adjusted = neighbor_ratings.sub(self.user_means_series.loc[sims.index], axis=0)
        weighted = adjusted.mul(sims, axis=0).sum(axis=0, skipna=True)
        denom = adjusted.notna().mul(np.abs(sims), axis=0).sum(axis=0).replace(0, np.nan)
        preds = (self.user_means_series.loc[user_id] + weighted / denom).dropna()
        preds = preds.drop(index=list(self.train_seen.get(user_id, set())), errors="ignore").sort_values(ascending=False).head(top_n)
        return [(int(mid), self.movie_lookup.loc[mid, "title"], float(self.clip(score))) for mid, score in preds.items()]

    def predict_item_cf(self, user_id: int, movie_id: int, k: int = 30) -> float:
        if user_id not in self.rating_matrix.index or movie_id not in self.rating_matrix.columns:
            return self.fallback(user_id, movie_id)
        user_ratings = self.rating_matrix.loc[user_id].dropna()
        sims = self.item_similarity.loc[movie_id, user_ratings.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        if sims.empty:
            return self.fallback(user_id, movie_id)
        denom = np.abs(sims.values).sum()
        if denom == 0:
            return self.fallback(user_id, movie_id)
        adjusted = user_ratings.loc[sims.index] - self.item_means_series.loc[sims.index]
        pred = self.item_means_series.loc[movie_id] + np.dot(sims.values, adjusted.values) / denom
        return self.clip(pred)

    def recommend_item_cf(self, user_id: int, top_n: int = 10, k: int = 30):
        if user_id not in self.rating_matrix.index:
            return []
        user_ratings = self.rating_matrix.loc[user_id].dropna()
        rows = []
        for movie_id in self.candidate_pool(user_id, limit=500):
            sims = self.item_similarity.loc[movie_id, user_ratings.index]
            sims = sims[sims > 0].sort_values(ascending=False).head(k)
            if sims.empty:
                continue
            denom = np.abs(sims.values).sum()
            if denom == 0:
                continue
            adjusted = user_ratings.loc[sims.index] - self.item_means_series.loc[sims.index]
            score = self.item_means_series.loc[movie_id] + np.dot(sims.values, adjusted.values) / denom
            rows.append((int(movie_id), self.movie_lookup.loc[movie_id, "title"], float(self.clip(score))))
        return sorted(rows, key=lambda x: x[2], reverse=True)[:top_n]

    def build_svd(self, n_factors: int = 50) -> "FullDatasetSolution":
        n_factors = min(n_factors, min(self.rating_matrix.shape) - 1)
        matrix = self.rating_matrix.to_numpy()
        means = np.nanmean(matrix, axis=1)
        demeaned = np.where(np.isnan(matrix), 0, matrix - means.reshape(-1, 1))
        U, sigma, Vt = svds(csr_matrix(demeaned), k=n_factors)
        order = np.argsort(sigma)[::-1]
        reconstructed = U[:, order] @ np.diag(sigma[order]) @ Vt[order, :] + means.reshape(-1, 1)
        self.svd_predictions = pd.DataFrame(reconstructed, index=self.rating_matrix.index, columns=self.rating_matrix.columns)
        return self

    def predict_svd(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.svd_predictions.index or movie_id not in self.svd_predictions.columns:
            return self.fallback(user_id, movie_id)
        return self.clip(self.svd_predictions.loc[user_id, movie_id])

    def recommend_svd(self, user_id: int, top_n: int = 10):
        if user_id not in self.svd_predictions.index:
            return []
        scores = self.svd_predictions.loc[user_id].drop(index=list(self.train_seen.get(user_id, set())), errors="ignore")
        scores = scores.sort_values(ascending=False).head(top_n)
        return [(int(mid), self.movie_lookup.loc[mid, "title"], float(score)) for mid, score in scores.items()]

    def build_surprise(self) -> "FullDatasetSolution":
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(self.train[["userId", "movieId", "rating"]], reader)
        grid = {"n_factors": [40, 60], "n_epochs": [15], "lr_all": [0.005], "reg_all": [0.02, 0.05]}
        search = GridSearchCV(SVD, grid, measures=["rmse"], cv=2, n_jobs=1)
        search.fit(data)
        self.surprise_model = SVD(random_state=self.seed, **search.best_params["rmse"])
        self.surprise_model.fit(data.build_full_trainset())
        return self

    def predict_surprise(self, user_id: int, movie_id: int) -> float:
        return self.clip(self.surprise_model.predict(user_id, movie_id).est)

    def recommend_surprise(self, user_id: int, top_n: int = 10):
        rows = [(mid, self.predict_surprise(user_id, mid)) for mid in self.candidate_pool(user_id, limit=500)]
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
        return [(int(mid), self.movie_lookup.loc[mid, "title"], float(score)) for mid, score in rows]

    def build_hybrid(self, max_rows: int = 3000) -> "FullDatasetSolution":
        source = self.val.sample(min(max_rows, len(self.val)), random_state=self.seed)
        rows = [self.hybrid_row(r.userId, r.movieId, r.rating) for r in source.itertuples(index=False)]
        self.hybrid_features_train = pd.DataFrame(rows)
        feats = ["content_score", "cf_score", "movie_popularity", "user_avg_rating"]
        self.hybrid_model = HistGradientBoostingRegressor(random_state=self.seed)
        self.hybrid_model.fit(self.hybrid_features_train[feats], self.hybrid_features_train["rating"])
        return self

    def hybrid_row(self, user_id: int, movie_id: int, rating=None) -> dict:
        return {
            "userId": user_id,
            "movieId": movie_id,
            "content_score": self.predict_content(user_id, movie_id),
            "cf_score": self.predict_item_cf(user_id, movie_id),
            "movie_popularity": self.movie_mean.get(movie_id, self.global_mean),
            "user_avg_rating": self.user_mean.get(user_id, self.global_mean),
            "rating": rating,
        }

    def predict_hybrid(self, user_id: int, movie_id: int) -> float:
        row = pd.DataFrame([self.hybrid_row(user_id, movie_id)])
        feats = ["content_score", "cf_score", "movie_popularity", "user_avg_rating"]
        return self.clip(self.hybrid_model.predict(row[feats])[0])

    def recommend_hybrid(self, user_id: int, top_n: int = 10):
        rows = [(mid, self.predict_hybrid(user_id, mid)) for mid in self.candidate_pool(user_id, limit=120)]
        rows = sorted(rows, key=lambda x: x[1], reverse=True)[:top_n]
        return [(int(mid), self.movie_lookup.loc[mid, "title"], float(score)) for mid, score in rows]

    def build_neural_features(self) -> "FullDatasetSolution":
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(self.movies["genre_list"])
        self.genre_cols = [f"genre_{g}" for g in mlb.classes_]
        self.movie_genre_features = pd.DataFrame(genre_matrix, index=self.movies["movieId"], columns=self.genre_cols)
        self.movie_feature_df = self.movie_genre_features.copy()
        self.movie_feature_df["year"] = self.movies.set_index("movieId")["year"]
        self.movie_feature_df["movie_avg_rating"] = self.movie_feature_df.index.map(self.movie_mean).fillna(self.global_mean)
        merged = self.train.merge(self.movie_genre_features, left_on="movieId", right_index=True, how="left")
        weighted = merged[self.genre_cols].mul(merged["rating"], axis=0)
        numerator = weighted.groupby(merged["userId"]).sum()
        denominator = merged[self.genre_cols].groupby(merged["userId"]).sum().replace(0, np.nan)
        self.user_feature_df = numerator.div(denominator).fillna(self.global_mean)
        self.user_feature_df["user_avg_rating"] = self.user_feature_df.index.map(self.user_mean).fillna(self.global_mean)
        self.user_feature_df["user_rating_count"] = self.user_feature_df.index.map(self.train["userId"].value_counts()).fillna(0)
        self.movie_feature_scaled = pd.DataFrame(StandardScaler().fit_transform(self.movie_feature_df), index=self.movie_feature_df.index, columns=self.movie_feature_df.columns)
        self.user_feature_scaled = pd.DataFrame(StandardScaler().fit_transform(self.user_feature_df), index=self.user_feature_df.index, columns=self.user_feature_df.columns)
        return self

    def _neural_xy(self, df: pd.DataFrame):
        valid = df[df["userId"].isin(self.user_feature_scaled.index) & df["movieId"].isin(self.movie_feature_scaled.index)].copy()
        xu = self.user_feature_scaled.loc[valid["userId"]].to_numpy(dtype=np.float32)
        xm = self.movie_feature_scaled.loc[valid["movieId"]].to_numpy(dtype=np.float32)
        y = valid["rating"].to_numpy(dtype=np.float32)
        return valid, xu, xm, y

    def build_neural(self, max_rows: int = 100_000, epochs: int = 5) -> "FullDatasetSolution":
        train_df = self.train.sample(min(max_rows, len(self.train)), random_state=self.seed)
        val_df = self.val.sample(min(20_000, len(self.val)), random_state=self.seed)
        _, xu, xm, y = self._neural_xy(train_df)
        _, xuv, xmv, yv = self._neural_xy(val_df)
        self.neural_model = TwoTower(xu.shape[1], xm.shape[1], emb_dim=32)
        opt = torch.optim.Adam(self.neural_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        self.training_history = []
        for epoch in range(epochs):
            self.neural_model.train()
            idx = np.random.permutation(len(y))
            train_loss = []
            for start in range(0, len(y), 512):
                batch = idx[start : start + 512]
                ub = torch.from_numpy(xu[batch])
                ib = torch.from_numpy(xm[batch])
                yb = torch.from_numpy(y[batch])
                opt.zero_grad()
                pred = self.neural_model(ub, ib)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                train_loss.append(float(loss.item()))
            self.neural_model.eval()
            with torch.no_grad():
                val_pred = self.neural_model(torch.from_numpy(xuv), torch.from_numpy(xmv))
                val_loss = float(loss_fn(val_pred, torch.from_numpy(yv)).item())
            self.training_history.append({"epoch": epoch + 1, "train_mse": np.mean(train_loss), "val_mse": val_loss})
        return self

    def predict_neural(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_feature_scaled.index or movie_id not in self.movie_feature_scaled.index:
            return self.fallback(user_id, movie_id)
        self.neural_model.eval()
        with torch.no_grad():
            xu = torch.from_numpy(self.user_feature_scaled.loc[[user_id]].to_numpy(dtype=np.float32))
            xm = torch.from_numpy(self.movie_feature_scaled.loc[[movie_id]].to_numpy(dtype=np.float32))
            return self.clip(float(self.neural_model(xu, xm).cpu().numpy()[0]))

    def recommend_neural(self, user_id: int, top_n: int = 10):
        candidates = self.candidate_pool(user_id, limit=150)
        self.neural_model.eval()
        with torch.no_grad():
            xu = np.repeat(self.user_feature_scaled.loc[[user_id]].to_numpy(dtype=np.float32), len(candidates), axis=0)
            xm = self.movie_feature_scaled.loc[candidates].to_numpy(dtype=np.float32)
            preds = self.neural_model(torch.from_numpy(xu), torch.from_numpy(xm)).cpu().numpy()
        order = np.argsort(preds)[::-1][:top_n]
        return [(int(candidates[i]), self.movie_lookup.loc[candidates[i], "title"], float(preds[i])) for i in order]

    def run_rl(self, steps: int = 3000) -> dict:
        rating_lookup = {(int(r.userId), int(r.movieId)): float(r.rating) for r in self.test.itertuples(index=False)}
        user_state = {int(uid): self.user_feature_df.loc[uid, self.genre_cols].idxmax().replace("genre_", "") for uid in self.user_feature_df.index}
        movie_state = {int(mid): next((g for g, v in self.movie_genre_features.loc[mid].items() if v == 1), "Unknown").replace("genre_", "") for mid in self.movie_genre_features.index}
        pool = self.popular_items[:100]

        def reward(uid, mid):
            val = rating_lookup.get((int(uid), int(mid)))
            return 0 if val is None else (1 if val >= 4 else -1)

        q = pd.Series(0.0, index=pool)
        counts = pd.Series(0, index=pool, dtype=int)
        history = []
        for step in range(steps):
            uid = int(np.random.choice(self.eval_users))
            choices = [m for m in pool if m not in self.train_seen.get(uid, set())]
            if not choices:
                continue
            explore = np.random.rand() < 0.1 or (counts.loc[choices] == 0).all()
            mid = int(np.random.choice(choices)) if explore else int(q.loc[choices].idxmax())
            r = reward(uid, mid)
            counts.loc[mid] += 1
            q.loc[mid] += (r - q.loc[mid]) / counts.loc[mid]
            history.append({"step": step, "reward": r, "mode": "explore" if explore else "exploit"})

        q_table = pd.DataFrame(0.0, index=sorted(set(user_state.values()) | set(movie_state.values())), columns=pool)
        for step in range(steps):
            uid = int(np.random.choice(self.eval_users))
            state = user_state.get(uid, "Unknown")
            choices = [m for m in pool if m not in self.train_seen.get(uid, set())]
            if not choices:
                continue
            mid = int(np.random.choice(choices)) if np.random.rand() < 0.1 else int(q_table.loc[state, choices].idxmax())
            r = reward(uid, mid)
            next_state = movie_state.get(mid, state)
            old = q_table.loc[state, mid]
            q_table.loc[state, mid] = old + 0.1 * (r + 0.9 * q_table.loc[next_state].max() - old)
        hist = pd.DataFrame(history)
        return {
            "summary": pd.DataFrame([{"avg_reward": hist["reward"].mean(), "exploration_rate": (hist["mode"] == "explore").mean()}]),
            "top_movies": q.sort_values(ascending=False).head(10).rename_axis("movieId").reset_index(name="estimated_reward").merge(self.movies[["movieId", "title", "genres"]], on="movieId", how="left"),
            "q_table": q_table,
        }

    def explain_content(self, user_id: int, movie_id: int, top_n: int = 5) -> pd.DataFrame:
        contrib = (self.user_feature_df.loc[user_id, self.genre_cols] * self.movie_genre_features.loc[movie_id, self.genre_cols]).sort_values(ascending=False)
        contrib = contrib[contrib > 0].head(top_n).reset_index()
        contrib.columns = ["feature", "contribution"]
        contrib["feature"] = contrib["feature"].str.replace("genre_", "", regex=False)
        return contrib

    def shap_content(self, user_id: int, movie_id: int):
        sample = self.train.sample(min(3000, len(self.train)), random_state=self.seed)
        X = pd.DataFrame(
            {
                "content_score": [self.predict_content(r.userId, r.movieId) for r in sample.itertuples(index=False)],
                "movie_avg_rating": sample["movieId"].map(self.movie_mean).fillna(self.global_mean).to_numpy(),
                "user_avg_rating": sample["userId"].map(self.user_mean).fillna(self.global_mean).to_numpy(),
            }
        )
        model = HistGradientBoostingRegressor(random_state=self.seed).fit(X, sample["rating"])
        explainer = shap.Explainer(model, X.sample(min(200, len(X)), random_state=self.seed))
        row = pd.DataFrame([{"content_score": self.predict_content(user_id, movie_id), "movie_avg_rating": self.movie_mean.get(movie_id, self.global_mean), "user_avg_rating": self.user_mean.get(user_id, self.global_mean)}])
        return explainer(row)

    def explain_user_neighbors(self, user_id: int, movie_id: int, k: int = 5) -> pd.DataFrame:
        ratings = self.rating_matrix[movie_id].dropna()
        sims = self.user_similarity.loc[user_id].drop(user_id, errors="ignore").loc[ratings.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        return pd.DataFrame({"neighbor_user": sims.index, "similarity": sims.values, "neighbor_rating": ratings.loc[sims.index].values})

    def explain_item_neighbors(self, user_id: int, movie_id: int, k: int = 5) -> pd.DataFrame:
        user_ratings = self.rating_matrix.loc[user_id].dropna()
        sims = self.item_similarity.loc[movie_id, user_ratings.index]
        sims = sims[sims > 0].sort_values(ascending=False).head(k)
        return pd.DataFrame({"movieId": sims.index, "title": [self.movie_lookup.loc[mid, "title"] for mid in sims.index], "similarity": sims.values, "user_rating": user_ratings.loc[sims.index].values})

    def lime_neural(self, user_id: int, movie_id: int):
        sample = self.train.sample(min(2000, len(self.train)), random_state=self.seed)
        _, xu, xm, _ = self._neural_xy(sample)
        combo = np.hstack([xu, xm])
        names = [f"user_{c}" for c in self.user_feature_scaled.columns] + [f"movie_{c}" for c in self.movie_feature_scaled.columns]
        explainer = LimeTabularExplainer(combo, feature_names=names, mode="regression", random_state=self.seed)
        row = np.hstack([self.user_feature_scaled.loc[user_id].to_numpy(dtype=np.float32), self.movie_feature_scaled.loc[movie_id].to_numpy(dtype=np.float32)])
        cut = self.user_feature_scaled.shape[1]

        def pred(arr):
            arr = np.asarray(arr, dtype=np.float32)
            with torch.no_grad():
                out = self.neural_model(torch.from_numpy(arr[:, :cut]), torch.from_numpy(arr[:, cut:])).cpu().numpy()
            return out.reshape(-1)

        return explainer.explain_instance(row, pred, num_features=10)

    def comparison_table(self) -> pd.DataFrame:
        rows = [
            {"model": "Content-Based", "RMSE": self.evaluate_rmse(self.predict_content), **self.evaluate_ranking(self.recommend_content)},
            {"model": "User-CF", "RMSE": self.evaluate_rmse(lambda u, m: self.predict_user_cf(u, m, 30)), **self.evaluate_ranking(lambda u, top_n=10: self.recommend_user_cf(u, top_n, 30))},
            {"model": "Item-CF", "RMSE": self.evaluate_rmse(lambda u, m: self.predict_item_cf(u, m, 30)), **self.evaluate_ranking(lambda u, top_n=10: self.recommend_item_cf(u, top_n, 30))},
            {"model": "Manual SVD", "RMSE": self.evaluate_rmse(self.predict_svd), **self.evaluate_ranking(self.recommend_svd)},
            {"model": "Surprise SVD", "RMSE": self.evaluate_rmse(self.predict_surprise), **self.evaluate_ranking(self.recommend_surprise)},
            {"model": "Hybrid", "RMSE": self.evaluate_rmse(self.predict_hybrid), **self.evaluate_ranking(self.recommend_hybrid)},
            {"model": "Neural", "RMSE": self.evaluate_rmse(self.predict_neural), **self.evaluate_ranking(self.recommend_neural)},
        ]
        return pd.DataFrame(rows).sort_values("RMSE")


def build_ready_solution() -> FullDatasetSolution:
    sol = FullDatasetSolution()
    sol.download_and_load().prepare_model_core().temporal_split()
    sol.build_content().build_cf().build_svd().build_surprise().build_hybrid().build_neural_features().build_neural()
    return sol
