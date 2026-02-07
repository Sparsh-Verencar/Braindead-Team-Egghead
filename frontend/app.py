import os
import pickle
import pandas as pd
import streamlit as st
import sys

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilityfunctions.modeladapter import SVDAdapter, HybridAdapter, RecommenderWrapper, PopularityAdapter

# -----------------------------
# Minimal PureSVDRecommender class (for unpickling)
# -----------------------------
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import numpy as np

class PureSVDRecommender:
    def __init__(self, rating_scale=(0.5, 5.0), random_state=30):
        self.reader = Reader(rating_scale=rating_scale)
        self.random_state = random_state
        self.model = None
        self.ratings_df = None
        self.trainset = None
        self.metrics_ = {}

    def fit(self, ratings_df, test_size=0.2):
        self.ratings_df = ratings_df.copy()
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], self.reader)
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        self.model = SVD(n_factors=200, n_epochs=200, lr_all=0.005, reg_all=0.05, random_state=self.random_state)
        self.model.fit(trainset)
        self.trainset = trainset
        predictions = self.model.test(testset)
        self.metrics_["rmse"] = accuracy.rmse(predictions, verbose=False)
        self.metrics_["mae"] = accuracy.mae(predictions, verbose=False)
        return self

    def _get_watched_movies(self, user_id):
        return set(self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"])

    def _get_liked_movies(self, user_id, min_rating=4.0):
        return set(self.ratings_df[(self.ratings_df["userId"] == user_id) & (self.ratings_df["rating"] >= min_rating)]["movieId"])

    def _get_similar_liked_movies(self, user_id, rec_movie_id, movies_df, top_k=2):
        liked_movies = self._get_liked_movies(user_id)
        try:
            rec_inner_id = self.trainset.to_inner_iid(rec_movie_id)
        except ValueError:
            return []
        rec_vec = self.model.qi[rec_inner_id]
        similarities = []
        for m in liked_movies:
            try:
                m_inner = self.trainset.to_inner_iid(m)
            except ValueError:
                continue
            m_vec = self.model.qi[m_inner]
            sim = np.dot(rec_vec, m_vec) / (np.linalg.norm(rec_vec) * np.linalg.norm(m_vec) + 1e-9)
            similarities.append((m, sim))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        similar_titles = []
        for m, _ in similarities:
            title = movies_df.loc[movies_df["movieId"] == m, "title"].values
            if len(title) > 0:
                similar_titles.append(title[0])
        return similar_titles

    def recommend(self, user_id, movies_df, n=10):
        watched = self._get_watched_movies(user_id)
        candidates = list(set(movies_df["movieId"]) - watched)
        preds = [(m, self.model.predict(user_id, m).est) for m in candidates]
        top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:n]
        results = []
        for movie_id, score in top_n:
            title = movies_df.loc[movies_df["movieId"] == movie_id, "title"].values[0]
            similar_liked = self._get_similar_liked_movies(user_id, movie_id, movies_df, top_k=2)
            if similar_liked:
                explanation = f"Recommended because it is similar to movies you liked such as {' and '.join(similar_liked)}."
            else:
                explanation = "Recommended based on patterns in your past movie ratings."
            results.append((movie_id, title, round(score, 3), explanation))
        return results


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")
st.title("🎬 Movie Recommender")

# -----------------------------
# Load CSVs
# -----------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

movies_file = load_csv("../data/movies.csv")
ratings_file = load_csv("../data/ratings.csv")

# -----------------------------
# Sidebar: Model selector
# -----------------------------
# -----------------------------
# Sidebar: Model Selector
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Select Recommender Model",
    ["Pure SVD", "Hybrid", "Popularity"]
)

# -----------------------------
# Initialize selected model
# -----------------------------
if model_choice == "Pure SVD":
    svd_pickle_path = st.sidebar.text_input("SVD pickle path", "../models/pure_svd_recommender.pkl")
    if not os.path.exists(svd_pickle_path):
        st.error(f"SVD pickle not found at {svd_pickle_path}")
        st.stop()
    with open(svd_pickle_path, "rb") as f:
        svd_model = pickle.load(f)
    adapter = SVDAdapter(svd_model=svd_model, movies_df=movies_file)
elif model_choice == "Hybrid":
    hybrid_pickle_path = st.sidebar.text_input("Hybrid pickle path", "../models/hybrid_bundle.pkl")
    if not os.path.exists(hybrid_pickle_path):
        st.error(f"Hybrid bundle not found at {hybrid_pickle_path}")
        st.stop()
    with open(hybrid_pickle_path, "rb") as f:
        hybrid_bundle = pickle.load(f)
    alpha = st.sidebar.slider("SVD vs Content Weight (alpha)", 0.0, 1.0, 0.7, 0.05)
    from utilityfunctions.modeladapter import HybridAdapter
    adapter = HybridAdapter(bundle=hybrid_bundle, alpha=alpha)
elif model_choice == "Popularity":
    from utilityfunctions.modeladapter import PopularityAdapter
    adapter = PopularityAdapter(ratings_df=ratings_file, movies_df=movies_file)

model_wrapper = RecommenderWrapper(adapter=adapter)


# -----------------------------
# Sidebar: User input
# -----------------------------
user_id = st.sidebar.number_input(
    "User ID:",
    min_value=1,
    max_value=int(ratings_file["userId"].max()),
    value=1
)
top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)

# -----------------------------
# Recommendations
# -----------------------------
st.header(f"🎯 Top {top_n} Recommendations for User {user_id}")

if st.button("🚀 Generate Recommendations"):
    try:
        df = model_wrapper.recommend(user_id=user_id, top_n=top_n)
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    if df.empty:
        st.warning("No recommendations found!")
    else:
        for idx, row in enumerate(df.itertuples(), 1):
            st.markdown(f"### {idx}. 🎬 {row.title}")
            st.metric("Predicted Score", f"{row.svd_score:.2f}")
            if hasattr(row, "genre") and pd.notna(row.genre):
                st.markdown(f"**Genres:** {row.genre}")
            st.info(row.explanation)
            st.markdown("---")
