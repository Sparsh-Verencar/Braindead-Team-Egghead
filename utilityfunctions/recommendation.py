import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, bundle, alpha=0.6):
        """
        bundle: either dict containing movies, ratings, svd, tfidf, tfidf_matrix, movie_indices
                or a class object with these attributes
        """
        self.alpha = alpha
        # handle dict or object
        if isinstance(bundle, dict):
            self.movies = bundle['movies']
            self.ratings = bundle['ratings']
            self.svd = bundle['svd']
            self.tfidf = bundle['tfidf']
            self.tfidf_matrix = bundle['tfidf_matrix']
            self.movie_indices = bundle['movie_indices']
        else:
            self.movies = bundle.movies
            self.ratings = bundle.ratings
            self.svd = bundle.svd
            self.tfidf = bundle.tfidf
            self.tfidf_matrix = bundle.tfidf_matrix
            self.movie_indices = bundle.movie_indices

    def _user_profile(self, user_rated_movies):
        idxs = [self.movie_indices[mid] for mid in user_rated_movies if mid in self.movie_indices]
        if not idxs:
            return np.zeros(self.tfidf_matrix.shape[1])
        return np.asarray(self.tfidf_matrix[idxs].mean(axis=0)).flatten()

    def recommend(self, user_id, user_seen_movies=None, k=10):
        if user_seen_movies is None:
            user_seen_movies = []

        all_movies = list(set(self.movies.movieid) - set(user_seen_movies))

        # CF predictions
        cf_scores = np.array([self.svd.predict(user_id, mid).est for mid in all_movies])

        # Content similarity
        profile = self._user_profile(user_seen_movies)
        content_scores = np.zeros(len(all_movies))
        idx_map = {mid: i for i, mid in enumerate(all_movies) if mid in self.movie_indices}

        if idx_map:
            tfidf_idxs = [self.movie_indices[mid] for mid in idx_map.keys()]
            sim = cosine_similarity(self.tfidf_matrix[tfidf_idxs], profile.reshape(1, -1)).flatten()
            for i, mid in enumerate(idx_map.keys()):
                content_scores[idx_map[mid]] = sim[i]

        # Normalize
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-9)

        hybrid = self.alpha * norm(cf_scores) + (1 - self.alpha) * norm(content_scores)

        recs = pd.DataFrame({
            "movieid": all_movies,
            "hybrid_score": hybrid,
            "cf_score": cf_scores,
            "content_score": content_scores
        }).sort_values("hybrid_score", ascending=False).head(k)

        # Merge movie info
        recs = recs.merge(self.movies[['movieid', 'title', 'genre']], on='movieid', how='left')
        return recs
