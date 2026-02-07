from sklearn.metrics.pairwise import cosine_similarity

class Explainer:
    def __init__(self, bundle):
        """
        bundle: dict or object containing movies, tfidf_matrix, movie_indices, svd
        """
        if isinstance(bundle, dict):
            self.movies = bundle['movies']
            self.tfidf_matrix = bundle['tfidf_matrix']
            self.movie_indices = bundle['movie_indices']
            self.svd = bundle['svd']
        else:
            self.movies = bundle.movies
            self.tfidf_matrix = bundle.tfidf_matrix
            self.movie_indices = bundle.movie_indices
            self.svd = bundle.svd

    def _user_profile(self, user_rated_movies):
        idxs = [self.movie_indices[mid] for mid in user_rated_movies if mid in self.movie_indices]
        if not idxs:
            return np.zeros(self.tfidf_matrix.shape[1])
        return self.tfidf_matrix[idxs].mean(axis=0).A1  # flatten

    def explain(self, recs, user_id, user_seen_movies):
        profile = self._user_profile(user_seen_movies)

        print(f"\n🎯 EXPLANATIONS FOR USER {user_id}\n")
        for _, row in recs.iterrows():
            mid = row.movieid
            title = row.title
            score = row.hybrid_score
            genres = row.genre

            cf = self.svd.predict(user_id, mid).est
            content = 0
            if mid in self.movie_indices:
                idx = self.movie_indices[mid]
                content = cosine_similarity(self.tfidf_matrix[idx], profile.reshape(1, -1))[0][0]

            print("="*60)
            print(f"🎬 {title}")
            print(f"Genres: {genres}")
            print(f"Hybrid Score: {score:.3f}")
            print(f"CF Score (Users liked it): {cf:.3f}")
            print(f"Content Similarity (Matches your taste): {content:.3f}")

            reason = []
            if cf > 3.5:
                reason.append("Users similar to you rated this highly")
            if content > 0.2:
                reason.append("Movie matches your preferred genres")
            if not reason:
                reason.append("Exploration recommendation (novel content)")

            print("Reason:", " + ".join(reason))
