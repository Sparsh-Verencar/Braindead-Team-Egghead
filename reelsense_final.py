import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader
from surprise import SVD as SurpriseSVD
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FinalReelSenseRecommender:
    def __init__(self, ratings_path='data/ratings.csv', movies_path='data/movies.csv', genre_path='data/movie_genre.csv'):
        print("📂 Loading data...")
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        self.movie_genres = pd.read_csv(genre_path)
        
        # Process timestamps
        print("🕒 Processing timestamps...")
        self.ratings['ts'] = pd.to_datetime(
            self.ratings['date'].astype(str) + ' ' + self.ratings['time'].astype(str),
            errors='coerce'
        )
        self.ratings = self.ratings.dropna(subset=['ts']).sort_values(['userId', 'ts'])

        # Flatten Genres
        genre_collapsed = self.movie_genres.groupby('movieId')['genre'].apply(lambda x: '|'.join(x)).reset_index()
        self.movies = self.movies.merge(genre_collapsed, on='movieId', how='left').rename(columns={'genre': 'genres'})
        self.movies['genres'] = self.movies['genres'].fillna('Unknown')
        
        # Create mappings
        self.movie_title_map = dict(zip(self.movies['movieId'], self.movies['title']))
        self.movie_genre_map = dict(zip(self.movies['movieId'], self.movies['genres']))
        
        self._create_genre_matrix()
        self._create_tag_system()
        
        self.train_data = None
        self.test_data = None
        self.user_item_matrix = None
        self.item_sim_df = None
        self.user_profiles = {}
        self.user_neighborhoods = {}
        self.algo = None
        self.global_mean = None
        
    def _create_genre_matrix(self):
        """Create one-hot encoded genre matrix"""
        all_genres = set()
        for genres in self.movies['genres']:
            all_genres.update(genres.split('|'))
        
        self.all_genres = sorted(list(all_genres))
        self.genre_matrix = pd.DataFrame(0, index=self.movies['movieId'], columns=self.all_genres)
        
        for idx, row in self.movies.iterrows():
            movie_id = row['movieId']
            genres = row['genres'].split('|')
            for genre in genres:
                if genre in self.all_genres:
                    self.genre_matrix.loc[movie_id, genre] = 1
    
    def _create_tag_system(self):
        """
        Create enhanced tag system that includes both genres and derived tags.
        Tags are more granular than genres (e.g., 'mind-bending', 'epic', 'dark').
        """
        # Base tags are genres
        self.movie_tags = {}
        
        for movie_id, genres_str in self.movie_genre_map.items():
            tags = set(genres_str.split('|'))
            
            # Add derived tags based on genre combinations
            genre_list = list(tags)
            
            # Sci-Fi + Thriller = mind-bending
            if 'Sci-Fi' in genre_list and 'Thriller' in genre_list:
                tags.add('mind-bending')
            
            # Action + Adventure = epic
            if 'Action' in genre_list and 'Adventure' in genre_list:
                tags.add('epic')
            
            # Crime + Thriller = suspenseful
            if 'Crime' in genre_list and 'Thriller' in genre_list:
                tags.add('suspenseful')
            
            # Drama + War = intense
            if 'Drama' in genre_list and 'War' in genre_list:
                tags.add('intense')
            
            # Horror + Thriller = dark
            if 'Horror' in genre_list and 'Thriller' in genre_list:
                tags.add('dark')
            
            # Comedy + Romance = feel-good
            if 'Comedy' in genre_list and 'Romance' in genre_list:
                tags.add('feel-good')
            
            # Sci-Fi + Action = futuristic
            if 'Sci-Fi' in genre_list and 'Action' in genre_list:
                tags.add('futuristic')
            
            # Animation + Children = family-friendly
            if 'Animation' in genre_list and 'Children' in genre_list:
                tags.add('family-friendly')
            
            # Drama alone = character-driven (common pattern)
            if 'Drama' in genre_list and len(genre_list) <= 2:
                tags.add('character-driven')
            
            self.movie_tags[movie_id] = tags

    def time_based_split(self, n=5):
        """Chronological split"""
        print(f"✂️ Splitting data (Leave-last-{n} per user)...")
        self.ratings = self.ratings.sort_values(['userId', 'ts'])
        self.test_data = self.ratings.groupby('userId').tail(n)
        self.train_data = self.ratings.drop(self.test_data.index)
        self.global_mean = self.train_data['rating'].mean()

    def train_model(self):
        print("\n⚙️  Training Final Hybrid Model with Rich Explanations...")
        
        # === 1. MATRIX FACTORIZATION (SVD) ===
        print("   -> Training SVD...")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.train_data[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        
        self.algo = SurpriseSVD(n_factors=250, n_epochs=40, lr_all=0.008, reg_all=0.04, random_state=42)
        self.algo.fit(trainset)
        
        # === 2. USER PROFILES & NEIGHBORHOODS ===
        print("   -> Building user profiles and neighborhoods...")
        self._build_enhanced_user_profiles()
        self._build_user_neighborhoods()
        
        # === 3. COLLABORATIVE FILTERING ===
        print("   -> Computing collaborative similarity...")
        self.user_item_matrix = self.train_data.pivot_table(
            index='userId', columns='movieId', values='rating'
        )
        
        # Use adjusted cosine similarity
        user_mean = self.user_item_matrix.mean(axis=1)
        matrix_normalized = self.user_item_matrix.sub(user_mean, axis=0).fillna(0)
        
        # Item-item similarity
        item_sim = cosine_similarity(matrix_normalized.T)
        self.item_sim_df = pd.DataFrame(
            item_sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        # === 4. CONTENT-BASED SIMILARITY ===
        print("   -> Computing content similarity...")
        content_sim = cosine_similarity(self.genre_matrix)
        self.content_sim_df = pd.DataFrame(
            content_sim, 
            index=self.genre_matrix.index, 
            columns=self.genre_matrix.index
        )
        
        # === 5. QUALITY METRICS ===
        self.movie_popularity = self.train_data['movieId'].value_counts()
        self.movie_avg_rating = self.train_data.groupby('movieId')['rating'].mean()
        self.movie_rating_count = self.train_data.groupby('movieId')['rating'].count()
        
        # Weighted Rating
        v = self.movie_rating_count
        m = v.quantile(0.70)
        R = self.movie_avg_rating
        C = self.global_mean
        
        self.movie_quality_score = (v / (v + m) * R) + (m / (v + m) * C)
        
        # Recency score
        movie_recency = self.train_data.groupby('movieId')['ts'].max()
        max_ts = movie_recency.max()
        self.movie_recency_score = (movie_recency - movie_recency.min()) / (max_ts - movie_recency.min())
        
        print("✅ Training Complete.")
    
    def _build_enhanced_user_profiles(self):
        """Build comprehensive user profiles"""
        for user_id in tqdm(self.train_data['userId'].unique(), desc="Building profiles"):
            user_data = self.train_data[self.train_data['userId'] == user_id]
            
            # Separate by rating levels
            liked_movies = user_data[user_data['rating'] >= 4.0]['movieId'].values
            loved_movies = user_data[user_data['rating'] >= 4.5]['movieId'].values
            disliked_movies = user_data[user_data['rating'] <= 2.5]['movieId'].values
            
            # Genre preferences
            genre_prefs = defaultdict(float)
            for _, row in user_data.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                weight = (rating - self.global_mean) / 2.0
                
                if movie_id in self.movie_genre_map:
                    genres = self.movie_genre_map[movie_id].split('|')
                    for genre in genres:
                        genre_prefs[genre] += weight
            
            # Normalize
            total = sum(abs(v) for v in genre_prefs.values())
            if total > 0:
                genre_prefs = {k: v/total for k, v in genre_prefs.items()}
            
            # Tag preferences (more granular)
            tag_prefs = defaultdict(float)
            for _, row in user_data.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                weight = (rating - self.global_mean) / 2.0
                
                if movie_id in self.movie_tags:
                    tags = self.movie_tags[movie_id]
                    for tag in tags:
                        tag_prefs[tag] += weight
            
            total_tags = sum(abs(v) for v in tag_prefs.values())
            if total_tags > 0:
                tag_prefs = {k: v/total_tags for k, v in tag_prefs.items()}
            
            # Temporal analysis
            user_data_sorted = user_data.sort_values('ts', ascending=False)
            recent_movies = user_data_sorted.head(10)['movieId'].values
            recent_liked = user_data_sorted[user_data_sorted['rating'] >= 4.0].head(5)['movieId'].values
            
            # User behavior
            avg_rating = user_data['rating'].mean()
            user_bias = avg_rating - self.global_mean
            
            self.user_profiles[user_id] = {
                'liked_movies': liked_movies,
                'loved_movies': loved_movies,
                'disliked_movies': disliked_movies,
                'genre_prefs': genre_prefs,
                'tag_prefs': tag_prefs,
                'recent_movies': recent_movies,
                'recent_liked': recent_liked,
                'avg_rating': avg_rating,
                'user_bias': user_bias,
                'rating_count': len(user_data)
            }
    
    def _build_user_neighborhoods(self):
        """Build collaborative neighborhoods (similar users)"""
        print("   -> Computing user neighborhoods...")
        
        # User-user similarity based on ratings
        user_sim = cosine_similarity(self.user_item_matrix.fillna(0))
        user_sim_df = pd.DataFrame(
            user_sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # For each user, find top 10 most similar users
        for user_id in self.user_item_matrix.index:
            similar_users = user_sim_df[user_id].nlargest(11)[1:].index.tolist()  # Exclude self
            self.user_neighborhoods[user_id] = similar_users

    def get_hybrid_recommendations(self, user_id, top_n=10):
        """Generate hybrid recommendations with comprehensive explanations"""
        
        if user_id not in self.user_profiles:
            return self._get_popular_recommendations(top_n)
        
        profile = self.user_profiles[user_id]
        
        # Get watched movies
        if user_id in self.user_item_matrix.index:
            watched = set(self.user_item_matrix.loc[user_id].dropna().index)
        else:
            watched = set()
        
        # === SMART CANDIDATE SELECTION ===
        min_ratings = 10
        popular_enough = self.movie_popularity[self.movie_popularity >= min_ratings].index
        
        # Include movies similar to loved movies
        loved_similar = set()
        if len(profile['loved_movies']) > 0:
            for loved_id in profile['loved_movies'][:3]:
                if loved_id in self.item_sim_df.index:
                    similar_items = self.item_sim_df[loved_id].nlargest(50).index
                    loved_similar.update(similar_items)
        
        candidates = set(popular_enough) | loved_similar
        candidates = [m for m in candidates if m not in watched]
        
        if len(candidates) == 0:
            return []
        
        scores = {}
        explanation_data = {}
        
        for movie_id in candidates:
            # === SCORING ===
            # Signal 1: SVD (35%)
            try:
                svd_pred = self.algo.predict(user_id, movie_id).est
                svd_score = svd_pred - profile['user_bias']
            except:
                svd_score = self.global_mean
            svd_normalized = max(0, (svd_score - 1) / 4)
            
            # Signal 2: Collaborative Similarity (30%)
            collab_score = 0
            collab_movies = []
            if movie_id in self.item_sim_df.index:
                recent_liked = [m for m in profile['recent_liked'] if m in self.item_sim_df.columns]
                
                if recent_liked:
                    sims = self.item_sim_df.loc[movie_id, recent_liked]
                    top_indices = sims.nlargest(3).index
                    collab_movies = [(idx, sims[idx]) for idx in top_indices]
                    
                    sim_values = np.maximum(sims.values, 0)
                    if len(sim_values) > 0 and sim_values.max() > 0:
                        top_k_sims = np.sort(sim_values)[-3:]
                        collab_score = np.mean(top_k_sims)
            
            # Signal 3: Content-Based (15%)
            content_score = 0
            if movie_id in self.content_sim_df.index:
                liked_in_matrix = [m for m in profile['liked_movies'] if m in self.content_sim_df.columns]
                
                if liked_in_matrix:
                    content_sims = self.content_sim_df.loc[movie_id, liked_in_matrix].values
                    content_score = np.mean(np.maximum(content_sims, 0))
            
            # Signal 4: Genre Affinity (10%)
            genre_score = 0
            if movie_id in self.movie_genre_map:
                movie_genres = self.movie_genre_map[movie_id].split('|')
                positive_match = sum(max(0, profile['genre_prefs'].get(g, 0)) for g in movie_genres)
                genre_score = max(0, positive_match)
            
            # Signal 5: Tag Affinity (5%)
            tag_score = 0
            if movie_id in self.movie_tags:
                movie_tag_list = self.movie_tags[movie_id]
                positive_tag_match = sum(max(0, profile['tag_prefs'].get(t, 0)) for t in movie_tag_list)
                tag_score = max(0, positive_tag_match)
            
            # Signal 6: Quality (3%)
            quality_score = 0
            if movie_id in self.movie_quality_score.index:
                quality_score = (self.movie_quality_score[movie_id] - 1) / 4
            
            # Signal 7: Recency (2%)
            recency_score = 0
            if movie_id in self.movie_recency_score.index:
                recency_score = self.movie_recency_score[movie_id]
            
            # === COMBINED SCORE ===
            final_score = (
                0.35 * svd_normalized +
                0.30 * collab_score +
                0.15 * content_score +
                0.10 * genre_score +
                0.05 * tag_score +
                0.03 * quality_score +
                0.02 * recency_score
            )
            
            # Boost for agreement
            signal_agreement = sum([
                svd_normalized > 0.6,
                collab_score > 0.5,
                content_score > 0.5,
                genre_score > 0.3
            ])
            
            if signal_agreement >= 3:
                final_score *= 1.15
            
            scores[movie_id] = final_score
            explanation_data[movie_id] = {
                'collab_movies': collab_movies,
                'collab_score': collab_score,
                'content_score': content_score,
                'genre_score': genre_score,
                'tag_score': tag_score
            }
        
        # Sort by score
        sorted_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Format results with COMPREHENSIVE EXPLANATIONS
        results = []
        for movie_id, final_score in sorted_movies:
            if movie_id not in self.movie_title_map:
                continue
                
            title = self.movie_title_map[movie_id]
            tags = self.movie_genre_map.get(movie_id, 'Unknown')
            
            explanation = self._generate_comprehensive_explanation(
                user_id, movie_id, explanation_data[movie_id]
            )
            
            results.append({
                'movieId': int(movie_id),
                'title': title,
                'tags': tags,
                'explanation': explanation
            })
        
        return results
    
    def _generate_comprehensive_explanation(self, user_id, movie_id, exp_data):
        """
        Generate explanation using ALL THREE SOURCES:
        1. Tag similarity
        2. Genre overlap  
        3. Collaborative user neighborhood
        
        Format: "Because you liked X and Y, which share the tags 'tag1' and 'tag2'"
        """
        profile = self.user_profiles[user_id]
        
        if movie_id not in self.movie_genre_map:
            return "Recommended for you"
        
        movie_genres = set(self.movie_genre_map[movie_id].split('|'))
        movie_tags = self.movie_tags.get(movie_id, set())
        
        # === STRATEGY 1: Collaborative Similarity (Item-based CF) ===
        if exp_data['collab_score'] > 0.3 and exp_data['collab_movies']:
            # Get top 2 similar movies
            top_collab = exp_data['collab_movies'][:2]
            
            similar_movies = []
            for collab_id, sim_score in top_collab:
                if collab_id in self.movie_title_map and sim_score > 0.3:
                    similar_movies.append({
                        'id': collab_id,
                        'title': self.movie_title_map[collab_id],
                        'genres': set(self.movie_genre_map.get(collab_id, '').split('|')),
                        'tags': self.movie_tags.get(collab_id, set())
                    })
            
            if len(similar_movies) >= 2:
                # Try tag similarity first (more specific)
                shared_tags = movie_tags & similar_movies[0]['tags'] & similar_movies[1]['tags']
                
                if len(shared_tags) >= 2:
                    tag_list = sorted(list(shared_tags))[:2]
                    return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tags '{tag_list[0]}' and '{tag_list[1]}'"
                elif len(shared_tags) == 1:
                    tag = list(shared_tags)[0]
                    # Also find genre overlap
                    shared_genres = movie_genres & similar_movies[0]['genres'] & similar_movies[1]['genres']
                    if shared_genres:
                        genre = sorted(list(shared_genres))[0]
                        return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tags '{tag}' and '{genre}'"
                    return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tag '{tag}'"
                
                # Fall back to genre overlap
                shared_genres = movie_genres & similar_movies[0]['genres'] & similar_movies[1]['genres']
                if len(shared_genres) >= 2:
                    genre_list = sorted(list(shared_genres))[:2]
                    return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tags '{genre_list[0]}' and '{genre_list[1]}'"
                elif len(shared_genres) == 1:
                    genre = list(shared_genres)[0]
                    return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tag '{genre}'"
                
                # Any overlap at all
                all_shared = (movie_tags | movie_genres) & (similar_movies[0]['tags'] | similar_movies[0]['genres']) & (similar_movies[1]['tags'] | similar_movies[1]['genres'])
                if all_shared:
                    tag_list = sorted(list(all_shared))[:2]
                    if len(tag_list) >= 2:
                        return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, which share the tags '{tag_list[0]}' and '{tag_list[1]}'"
                    return f"Because you liked {similar_movies[0]['title']} and {similar_movies[1]['title']}, similar to this film"
            
            elif len(similar_movies) == 1:
                # Single movie reference
                shared_tags = movie_tags & similar_movies[0]['tags']
                shared_genres = movie_genres & similar_movies[0]['genres']
                all_shared = (shared_tags | shared_genres)
                
                if len(all_shared) >= 2:
                    tag_list = sorted(list(all_shared))[:2]
                    return f"Because you liked {similar_movies[0]['title']}, which shares the tags '{tag_list[0]}' and '{tag_list[1]}'"
                elif len(all_shared) == 1:
                    tag = list(all_shared)[0]
                    return f"Because you liked {similar_movies[0]['title']}, which shares the tag '{tag}'"
        
        # === STRATEGY 2: User Neighborhood (Collaborative Users) ===
        if user_id in self.user_neighborhoods:
            neighbors = self.user_neighborhoods[user_id][:3]
            
            # Check if neighbors also liked this movie
            neighbor_liked = []
            for neighbor_id in neighbors:
                if neighbor_id in self.user_item_matrix.index:
                    if movie_id in self.user_item_matrix.columns:
                        neighbor_rating = self.user_item_matrix.loc[neighbor_id, movie_id]
                        if not pd.isna(neighbor_rating) and neighbor_rating >= 4.0:
                            neighbor_liked.append(neighbor_id)
            
            if len(neighbor_liked) >= 2:
                # Users with similar taste liked this
                common_genres = sorted(list(movie_genres))[:2]
                if len(common_genres) >= 2:
                    return f"Recommended by users with similar taste, featuring '{common_genres[0]}' and '{common_genres[1]}'"
        
        # === STRATEGY 3: Tag & Genre Preferences ===
        if exp_data['tag_score'] > 0.15 or exp_data['genre_score'] > 0.15:
            # Find user's top tags/genres that match this movie
            user_top_tags = sorted(profile['tag_prefs'].items(), key=lambda x: x[1], reverse=True)[:5]
            user_top_genres = sorted(profile['genre_prefs'].items(), key=lambda x: x[1], reverse=True)[:5]
            
            matching_tags = [t for t, _ in user_top_tags if t in movie_tags and _ > 0]
            matching_genres = [g for g, _ in user_top_genres if g in movie_genres and _ > 0]
            
            all_matching = (matching_tags + matching_genres)
            
            if len(all_matching) >= 2:
                return f"Based on your preference for '{all_matching[0]}' and '{all_matching[1]}' films"
            elif len(all_matching) == 1:
                return f"Based on your preference for '{all_matching[0]}' films"
        
        # === DEFAULT: Generic but informative ===
        all_tags = sorted(list(movie_tags | movie_genres))[:2]
        if len(all_tags) >= 2:
            return f"Recommended {all_tags[0]} and {all_tags[1]} film for you"
        elif len(all_tags) == 1:
            return f"Highly rated {all_tags[0]} recommendation"
        
        return "Recommended based on your viewing history"
    
    def _get_popular_recommendations(self, top_n=10):
        """Cold start recommendations"""
        top_movies = self.movie_quality_score.nlargest(top_n * 2).index
        
        results = []
        for movie_id in top_movies[:top_n]:
            if movie_id not in self.movie_title_map:
                continue
            
            results.append({
                'movieId': int(movie_id),
                'title': self.movie_title_map[movie_id],
                'tags': self.movie_genre_map.get(movie_id, 'Unknown'),
                'explanation': "Trending pick with high ratings"
            })
        
        return results

    def calculate_performance(self, k=10):
        """Comprehensive evaluation"""
        print(f"\n🧪 Comprehensive Evaluation (K={k})...")
        
        # RMSE
        actuals, preds = [], []
        test_filtered = self.test_data[
            self.test_data['userId'].isin(self.user_profiles.keys())
        ]
        
        for _, row in test_filtered.iterrows():
            try:
                pred = self.algo.predict(row['userId'], row['movieId']).est
                actuals.append(row['rating'])
                preds.append(pred)
            except:
                continue
        
        rmse = np.sqrt(mean_squared_error(actuals, preds))

        # Ranking Metrics
        precisions, recalls, ndcgs, mrrs = [], [], [], []
        diversity_scores = []
        
        sample_users = list(self.user_profiles.keys())[:200]
        
        for u in tqdm(sample_users, desc="Evaluating"):
            recs = self.get_hybrid_recommendations(u, top_n=k)
            rec_ids = [r['movieId'] for r in recs]
            
            actual_hits = set(
                self.test_data[(self.test_data['userId'] == u) & (self.test_data['rating'] >= 4.0)]['movieId']
            )
            
            if len(actual_hits) == 0:
                continue
            
            hits = len(set(rec_ids) & actual_hits)
            precisions.append(hits / k if k > 0 else 0)
            recalls.append(hits / len(actual_hits))
            
            # NDCG
            dcg = sum([1.0/np.log2(i+2) if rec_ids[i] in actual_hits else 0 for i in range(len(rec_ids))])
            idcg = sum([1.0/np.log2(i+2) for i in range(min(len(actual_hits), k))])
            ndcgs.append(dcg / idcg if idcg > 0 else 0)
            
            # MRR
            for i, rec_id in enumerate(rec_ids):
                if rec_id in actual_hits:
                    mrrs.append(1.0 / (i + 1))
                    break
            else:
                mrrs.append(0)
            
            # Diversity
            if len(rec_ids) > 1:
                valid_recs = [r for r in rec_ids if r in self.content_sim_df.index]
                if len(valid_recs) > 1:
                    sim_sub = self.content_sim_df.loc[valid_recs, valid_recs].values
                    avg_sim = (np.sum(sim_sub) - len(valid_recs)) / (len(valid_recs)**2 - len(valid_recs))
                    diversity_scores.append(1 - avg_sim)

        print("\n" + "═"*60)
        print(f"📈  FINAL REELSENSE PERFORMANCE (K={k})")
        print("═"*60)
        print(f"ACCURACY:   RMSE: {rmse:.4f}")
        print(f"RANKING:    Prec@{k}: {np.mean(precisions):.4f} | Recall@{k}: {np.mean(recalls):.4f}")
        print(f"QUALITY:    NDCG@{k}: {np.mean(ndcgs):.4f} | MRR: {np.mean(mrrs):.4f}")
        print(f"DIVERSITY:  ILD: {np.mean(diversity_scores):.4f}")
        print(f"COVERAGE:   {len([p for p in precisions if p > 0])/len(precisions)*100:.1f}% users got ≥1 hit")
        print("═"*60)
        
        return {
            'rmse': rmse,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'ndcg': np.mean(ndcgs),
            'mrr': np.mean(mrrs),
            'diversity': np.mean(diversity_scores)
        }
    
    def save_model(self, path='models/reelsense_final.pkl'):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'algo': self.algo,
            'user_profiles': self.user_profiles,
            'user_neighborhoods': self.user_neighborhoods,
            'item_sim_df': self.item_sim_df,
            'content_sim_df': self.content_sim_df,
            'movie_popularity': self.movie_popularity,
            'movie_quality_score': self.movie_quality_score,
            'movie_avg_rating': self.movie_avg_rating,
            'movie_rating_count': self.movie_rating_count,
            'movie_recency_score': self.movie_recency_score,
            'global_mean': self.global_mean,
            'movie_title_map': self.movie_title_map,
            'movie_genre_map': self.movie_genre_map,
            'movie_tags': self.movie_tags,
            'genre_matrix': self.genre_matrix,
            'all_genres': self.all_genres,
            'user_item_matrix': self.user_item_matrix
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path='models/reelsense_final.pkl'):
        """Load model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.algo = model_data['algo']
        self.user_profiles = model_data['user_profiles']
        self.user_neighborhoods = model_data['user_neighborhoods']
        self.item_sim_df = model_data['item_sim_df']
        self.content_sim_df = model_data['content_sim_df']
        self.movie_popularity = model_data['movie_popularity']
        self.movie_quality_score = model_data['movie_quality_score']
        self.movie_avg_rating = model_data['movie_avg_rating']
        self.movie_rating_count = model_data['movie_rating_count']
        self.movie_recency_score = model_data['movie_recency_score']
        self.global_mean = model_data['global_mean']
        self.movie_title_map = model_data['movie_title_map']
        self.movie_genre_map = model_data['movie_genre_map']
        self.movie_tags = model_data['movie_tags']
        self.genre_matrix = model_data['genre_matrix']
        self.all_genres = model_data['all_genres']
        self.user_item_matrix = model_data['user_item_matrix']
        
        print(f"✅ Model loaded from {path}")


if __name__ == "__main__":
    # Train
    reelsense = FinalReelSenseRecommender()
    reelsense.time_based_split(n=5)
    reelsense.train_model()
    
    # Evaluate
    metrics = reelsense.calculate_performance(k=10)
    
    # Save
    reelsense.save_model('models/reelsense_final.pkl')
    
    # Demo
    print("\n" + "="*70)
    print("🎬 SAMPLE RECOMMENDATIONS WITH COMPREHENSIVE EXPLANATIONS")
    print("="*70)
    print("\n✨ Explanation sources: Tag similarity | Genre overlap | Collaborative neighborhood\n")
    
    test_user = list(reelsense.user_profiles.keys())[0]
    recs = reelsense.get_hybrid_recommendations(test_user, top_n=10)
    
    for i, rec in enumerate(recs, 1):
        print(f"{i}. {rec['title']}")
        print(f"   Tags: {rec['tags']}")
        print(f"   💡 {rec['explanation']}\n")
    
    # Test loading
    print("="*70)
    print("🔄 Testing model persistence...")
    new_model = FinalReelSenseRecommender()
    new_model.load_model('models/reelsense_final.pkl')
    
    loaded_recs = new_model.get_hybrid_recommendations(test_user, top_n=3)
    print(f"✅ Generated {len(loaded_recs)} recommendations with loaded model\n")
    for rec in loaded_recs:
        print(f"• {rec['title']}")
        print(f"  {rec['explanation']}\n")
    print("="*70)