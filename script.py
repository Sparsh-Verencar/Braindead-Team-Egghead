import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import Dataset, Reader
from surprise import SVD as SurpriseSVD

class ReelSenseRecommender:
    def __init__(self, ratings_path='data/ratings.csv', movies_path='data/movies.csv', genre_path='data/movie_genre.csv'):
        print("📂 Loading data...")
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.movie_genres = pd.read_csv(genre_path)
        
        # 1. Robust Timestamp Parsing
        print("🕒 Processing timestamps...")
        self.ratings['ts'] = pd.to_datetime(
            self.ratings['date'].astype(str) + ' ' + self.ratings['time'].astype(str),
            errors='coerce'
        )
        self.ratings = self.ratings.dropna(subset=['ts']).sort_values(['userId', 'ts'])

        # 2. Flatten Genres
        genre_collapsed = self.movie_genres.groupby('movieId')['genre'].apply(lambda x: '|'.join(x)).reset_index()
        self.movies = self.movies.merge(genre_collapsed, on='movieId', how='left').rename(columns={'genre': 'genres'})
        self.movies['genres'] = self.movies['genres'].fillna('Unknown')
        
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.item_sim_df = None
        self.predicted_ratings_df = None
        self.movie_popularity = None

    def time_based_split(self, n=5):
        """
        Chronological split: Last N movies per user for testing.
        Increased N to 5 to give the ranking metrics more targets to find.
        """
        print(f"✂️ Splitting data (Leave-last-{n} per user)...")
        # Ensure data is sorted by time
        self.ratings = self.ratings.sort_values(['userId', 'ts'])
        self.test_data = self.ratings.groupby('userId').tail(n)
        self.train_data = self.ratings.drop(self.test_data.index)

    def train_model(self):
        print("\n⚙️  Training Hybrid Model (SVD + Item Similarity)...")
        
        # --- A. Surprise SVD (The "Quality" Signal) ---
        # Optimized Hyperparameters based on common baselines for MovieLens
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_data[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        # Tuned: Higher factors for detail, higher reg to prevent overfitting
        algo = SurpriseSVD(n_factors=150, lr_all=0.005, reg_all=0.05, random_state=42)
        algo.fit(trainset)
        
        # --- B. Pre-compute Predictions (Vectorized) ---
        print("   -> Generating Rating Predictions...")
        users = self.train_data['userId'].unique()
        items = self.train_data['movieId'].unique()
        
        # Create a placeholder matrix
        self.predicted_ratings_df = pd.DataFrame(index=users, columns=items, dtype=float)
        
        # OPTIMIZATION: Predict in bulk is hard with Surprise, so we iterate efficiently
        # This part is still O(U*I) but necessary for dense matrix
        # For a hackathon, we can speed this up by only predicting relevant items, 
        # but let's stick to full coverage for accuracy.
        
        # We'll use the user_item_matrix structure to map predictions
        self.user_item_matrix = self.train_data.pivot_table(index='userId', columns='movieId', values='rating')
        
        # Create a dict for fast lookups
        predictions = []
        for u in tqdm(users, desc="Predicting User Ratings"):
            # Predict for ALL items (even unwatched)
            u_preds = [algo.predict(u, i).est for i in items]
            predictions.append(u_preds)
            
        self.predicted_ratings_df = pd.DataFrame(predictions, index=users, columns=items)

        # --- C. Item-Item Similarity (The "Relevance" Signal) ---
        print("   -> Computing Item Similarity...")
        # We use Pearson correlation on the ratings matrix to find similar movies
        # Filling NaN with 0 is crude, so we center by row mean first
        matrix_centered = self.user_item_matrix.sub(self.user_item_matrix.mean(axis=1), axis=0).fillna(0)
        sim_matrix = cosine_similarity(matrix_centered.T)
        self.item_sim_df = pd.DataFrame(sim_matrix, index=self.user_item_matrix.columns, columns=self.user_item_matrix.columns)

        # --- D. Popularity (The "Penalty" Signal) ---
        self.movie_popularity = self.train_data['movieId'].value_counts()
        
        print("✅ Training Complete.")

    def get_hybrid_recommendations(self, user_id, top_n=10):
        if user_id not in self.predicted_ratings_df.index:
            return []

        # 1. Get Base SVD Scores (The "Quality" Signal)
        svd_scores = self.predicted_ratings_df.loc[user_id]
        
        # --- IMPROVEMENT 1: MINIMUM VOTE FILTER ---
        # Ignore movies with fewer than 25 ratings to remove "noise"
        # (This drastically improves Precision because it sticks to known quantities)
        popular_movies = self.movie_popularity[self.movie_popularity >= 25].index
        svd_scores = svd_scores.loc[svd_scores.index.intersection(popular_movies)]
        
        # Normalize
        svd_norm = (svd_scores - svd_scores.min()) / (svd_scores.max() - svd_scores.min() + 1e-9)

        # 2. Get Similarity Scores (Temporal Awareness)
        user_history = self.train_data[self.train_data['userId'] == user_id].sort_values('ts', ascending=False).head(5)
        recent_movies = user_history['movieId'].values
        
        # Valid recent movies that exist in our similarity matrix
        valid_recent = [m for m in recent_movies if m in self.item_sim_df.index]
        
        if valid_recent:
            sim_scores = self.item_sim_df[valid_recent].mean(axis=1)
            sim_scores = sim_scores.reindex(svd_scores.index).fillna(0)
            sim_norm = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-9)
        else:
            sim_norm = 0.0

        # --- IMPROVEMENT 2: GENRE BOOSTING ---
        # If the user watched 'War', boost other 'War' movies
        genre_bonus = pd.Series(0.0, index=svd_scores.index)
        if valid_recent:
            # Get genres of recent movies
            recent_genres = set()
            for rm in valid_recent:
                g = self.movies[self.movies['movieId'] == rm]['genres'].values[0].split('|')
                recent_genres.update(g)
            
            # Boost movies that share these genres
            for m_id in svd_scores.index:
                m_genres = set(self.movies[self.movies['movieId'] == m_id]['genres'].values[0].split('|'))
                share = len(recent_genres & m_genres)
                if share > 0:
                    genre_bonus[m_id] = 0.2 * (share / len(m_genres)) # 20% bonus for genre match

        # 3. HYBRID MIXING
        # 40% Quality + 40% Similarity + 20% Genre Match
        final_scores = (0.4 * svd_norm) + (0.4 * sim_norm) + genre_bonus

        # Filter out watched
        watched = self.user_item_matrix.loc[user_id].dropna().index
        final_scores = final_scores.drop(watched, errors='ignore').sort_values(ascending=False)

        results = []
        for m_id in final_scores.head(top_n).index:
            m_info = self.movies[self.movies['movieId'] == m_id].iloc[0]
            
            # Better Explanations
            reason = "Highly Rated"
            if valid_recent and (m_id in self.item_sim_df.index):
                # Check Genre overlap first
                m_genres = set(m_info['genres'].split('|'))
                common_g = list(m_genres & recent_genres)
                
                if common_g:
                    reason = f"Matches your interest in {common_g[0]}"
                else:
                    best_match_idx = np.argmax(self.item_sim_df.loc[m_id, valid_recent].values)
                    match_id = valid_recent[best_match_idx]
                    match_title = self.movies[self.movies['movieId'] == match_id]['title'].values[0]
                    reason = f"Similar to '{match_title}'"

            results.append({
                'movieId': m_id,
                'title': m_info['title'],
                'explanation': f"{reason} (Score: {final_scores[m_id]:.2f})"
            })
        
        return results

    def calculate_performance(self, k=10):
        print(f"\n🧪 Comprehensive Evaluation (K={k})...")
        
        # A. RMSE (Rating Accuracy)
        actuals, preds = [], []
        # Optimization: Iterate only over test data that exists in our prediction matrix
        test_filtered = self.test_data[
            (self.test_data['userId'].isin(self.predicted_ratings_df.index)) & 
            (self.test_data['movieId'].isin(self.predicted_ratings_df.columns))
        ]
        
        for _, row in test_filtered.iterrows():
            actuals.append(row['rating'])
            preds.append(self.predicted_ratings_df.loc[row['userId'], row['movieId']])
            
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        # B. Ranking Metrics (Precision, Recall, NDCG)
        precisions, recalls, ndcgs = [], [], []
        diversity_scores = []
        
        # Sample 100 users for speed
        sample_users = test_filtered['userId'].unique()[:100]
        
        for u in tqdm(sample_users, desc="Ranking & Diversity"):
            # Get Hybrid Recommendations
            recs = self.get_hybrid_recommendations(u, top_n=k)
            rec_ids = [r['movieId'] for r in recs]
            
            # Ground Truth (Movies user actually watched in test period)
            # We consider a "hit" only if they rated it >= 3.5 (Implicit positive)
            actual_hits = set(self.test_data[(self.test_data['userId'] == u) & (self.test_data['rating'] >= 3.5)]['movieId'])
            
            # Precision/Recall
            hits = len(set(rec_ids) & actual_hits)
            precisions.append(hits / k)
            recalls.append(hits / len(actual_hits) if len(actual_hits) > 0 else 0)
            
            # NDCG
            dcg = sum([1.0/np.log2(i+2) if rec_ids[i] in actual_hits else 0 for i in range(len(rec_ids))])
            idcg = sum([1.0/np.log2(i+2) for i in range(min(len(actual_hits), k))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
            # Diversity (ILD)
            if len(rec_ids) > 1:
                # Get similarity submatrix for recommended items
                valid_recs = [r for r in rec_ids if r in self.item_sim_df.index]
                if len(valid_recs) > 1:
                    sim_sub = self.item_sim_df.loc[valid_recs, valid_recs].values
                    # Average off-diagonal similarity
                    avg_sim = (np.sum(sim_sub) - len(valid_recs)) / (len(valid_recs)**2 - len(valid_recs))
                    diversity_scores.append(1 - avg_sim)

        print("\n" + "═"*40)
        print(f"📈  REELSENSE HYBRID PERFORMANCE (K={k})")
        print("═"*40)
        print(f"ACCURACY:   RMSE: {rmse:.4f}")
        print(f"RANKING:    Prec@{k}: {np.mean(precisions):.4f} | Recall@{k}: {np.mean(recalls):.4f}")
        print(f"QUALITY:    NDCG@{k}: {np.mean(ndcgs):.4f}")
        print(f"DIVERSITY:  ILD: {np.mean(diversity_scores):.4f}")
        print("═"*40)

if __name__ == "__main__":
    reelsense = ReelSenseRecommender()
    reelsense.time_based_split()
    reelsense.train_model()
    reelsense.calculate_performance(k=10)
    
    # Demo
    test_user = reelsense.train_data['userId'].iloc[0]
    print(f"\n🎬 HYBRID PICKS FOR USER {test_user}:")
    for r in reelsense.get_hybrid_recommendations(test_user):
        print(f"⭐ {r['title']}")
        print(f"   💡 {r['explanation']}")