import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# -----------------------------
# SVD Adapter
# -----------------------------
class SVDAdapter:
    """
    Wraps the trained SVD pickle object into a uniform interface
    """
    def __init__(self, svd_model, movies_df):
        self.model = svd_model  # the loaded pickle
        self.movies = movies_df

        # Detect movieId column
        if 'movieId' in movies_df.columns:
            self.movie_id_col = 'movieId'
        elif 'movieid' in movies_df.columns:
            self.movie_id_col = 'movieid'
        else:
            raise ValueError("No movieId column found in movies dataframe")

        # Detect genre column if exists
        if 'genres' in movies_df.columns:
            self.genre_col = 'genres'
        elif 'genre' in movies_df.columns:
            self.genre_col = 'genre'
        else:
            self.genre_col = None

    def recommend_user(self, user_id, top_n=10):
        """
        Generate recommendations using the loaded SVD pickle.
        Returns a DataFrame with columns: [movieId, title, svd_score, explanation, (optional genre)]
        """
        if not hasattr(self.model, "recommend"):
            raise ValueError(
                "The loaded SVD model pickle has no 'recommend' method."
            )

        recs = self.model.recommend(user_id, self.movies, n=top_n)

        df = pd.DataFrame(
            recs,
            columns=[self.movie_id_col, "title", "svd_score", "explanation"]
        )

        # Merge genres if available
        if self.genre_col:
            df = df.merge(
                self.movies[[self.movie_id_col, self.genre_col]],
                on=self.movie_id_col,
                how="left"
            )

        return df


# -----------------------------
# Hybrid Adapter
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HybridAdapter:
    """
    Fast Hybrid Recommender: combines SVD + TF-IDF content-based similarity
    Precomputes movie-to-movie similarity for speed.
    """
    def __init__(self, bundle, alpha=0.7, use_precomputed=True):
        self.bundle = bundle
        self.movies = bundle["movies"]
        self.ratings = bundle["ratings"]
        self.svd = bundle["svd"]
        self.tfidf_matrix = bundle["tfidf_matrix"]
        self.movie_indices = bundle["movie_indices"]
        self.alpha = alpha  # SVD weight

        # Column detection
        self.movie_col = "movieid" if "movieid" in self.movies.columns else "movieId"
        self.user_col = "userid" if "userid" in self.ratings.columns else "userId"
        self.genre_col = "genre" if "genre" in self.movies.columns else None

        # Precompute similarity matrix if desired
        self.use_precomputed = use_precomputed
        self.similarity_matrix = None
        if self.use_precomputed:
            print("⚡ Precomputing movie-to-movie similarity matrix...")
            self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend_user(self, user_id, top_n=10):
        # Movies already watched
        watched = set(self.ratings[self.ratings[self.user_col] == user_id][self.movie_col])
        candidates = list(set(self.movies[self.movie_col]) - watched)

        # Movies the user liked
        liked_movies = self.ratings[
            (self.ratings[self.user_col] == user_id) &
            (self.ratings["rating"] >= 4.0)
        ][self.movie_col].values
        liked_indices = [self.movie_indices[m] for m in liked_movies if m in self.movie_indices]

        hybrid_scores = []

        for movie_id in candidates:
            # ----- SVD prediction -----
            svd_pred = self.svd.predict(user_id, movie_id).est

            # ----- Content-based score -----
            if liked_indices and movie_id in self.movie_indices:
                movie_idx = self.movie_indices[movie_id]

                if self.use_precomputed and self.similarity_matrix is not None:
                    sims = self.similarity_matrix[movie_idx, liked_indices]
                    content_score = np.mean(sims)
                else:
                    # fallback: compute on the fly
                    sims = cosine_similarity(
                        self.tfidf_matrix[movie_idx],
                        self.tfidf_matrix[liked_indices]
                    )[0]
                    content_score = np.mean(sims)

                # scale content similarity to rating scale
                content_score = 0.5 + content_score * (5 - 0.5)
            else:
                content_score = 0

            # ----- Combine SVD + content -----
            final_score = self.alpha * svd_pred + (1 - self.alpha) * content_score
            hybrid_scores.append((movie_id, final_score))

        # ----- Top-N movies -----
        top_n_preds = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_n]

        # ----- Build DataFrame -----
        results = []
        for movie_id, score in top_n_preds:
            title = self.movies.loc[self.movies[self.movie_col] == movie_id, "title"].values[0]
            genre = None
            if self.genre_col:
                genre = self.movies.loc[self.movies[self.movie_col] == movie_id, self.genre_col].values[0]
            explanation = "Recommended using hybrid SVD + content similarity"
            results.append((movie_id, title, round(score, 3), explanation, genre))

        df = pd.DataFrame(results, columns=[self.movie_col, "title", "svd_score", "explanation", "genre"])
        return df


class PopularityAdapter:
    """
    Popularity-based recommender: returns top trending movies
    """
    def __init__(self, ratings_df, movies_df):
        self.ratings = ratings_df
        self.movies = movies_df

    def recommend_user(self, user_id=None, top_n=10):
        # Compute IMDB weighted rating
        movie_stats = self.ratings.groupby('movieId').agg({'rating': ['count', 'mean']})
        movie_stats.columns = ['vote_count', 'vote_average']

        C = movie_stats['vote_average'].mean()
        m = movie_stats['vote_count'].quantile(0.90)

        qualified = movie_stats[movie_stats['vote_count'] >= m].copy()

        def weighted_rating(x, m=m, C=C):
            v = x['vote_count']
            R = x['vote_average']
            return (v / (v + m) * R) + (m / (v + m) * C)

        qualified['score'] = qualified.apply(weighted_rating, axis=1)
        qualified = qualified.merge(self.movies[['movieId', 'title']], on='movieId', how='left')
        top_movies = qualified.sort_values('score', ascending=False).head(top_n)

        df = pd.DataFrame(top_movies[['movieId', 'title', 'score']])
        df = df.rename(columns={'score': 'svd_score'})
        df['explanation'] = "Recommended because it is trending/popular globally."
        if 'genre' in self.movies.columns:
            df = df.merge(self.movies[['movieId', 'genre']], on='movieId', how='left')
        return df


# -----------------------------
# Recommender Wrapper
# -----------------------------
class RecommenderWrapper:
    """
    Unified interface for SVD and Hybrid models
    """
    def __init__(self, adapter=None):
        if adapter is None:
            raise ValueError("Adapter must be provided")
        self.model = adapter

    def recommend(self, user_id=None, top_n=10):
        if user_id is None:
            raise ValueError("User ID is required for recommendations")
        return self.model.recommend_user(user_id, top_n)
