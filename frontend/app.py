import os
import sys
import pandas as pd
import streamlit as st
sys.path.append(os.path.abspath('..'))
from utilityfunctions.explaination import Explainer
from utilityfunctions.recommendation import Recommender

# ----------------------------
# Load trained bundle
# ----------------------------
bundle_path = "../models/hybrid_bundle.pkl"
bundle = pd.read_pickle(bundle_path)

# Convert any StringDtype columns to object to be safe
for key, df in bundle.items():
    if isinstance(df, pd.DataFrame):
        for col in df.select_dtypes(include=["string"]).columns:
            df[col] = df[col].astype("object")

recommender = Recommender(bundle, alpha=0.6)
explainer = Explainer(bundle)
movies_df = bundle['movies']

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Hybrid Movie Recommender")

st.sidebar.header("User Preferences")
user_id = st.sidebar.number_input("Your User ID:", min_value=1, value=1)

# ----------------------------
# Section 1: Rate a Movie
# ----------------------------
st.header("1️⃣ Rate a Movie")

selected_movie = st.selectbox("Select a movie to rate:", movies_df['title'])
rating = st.slider("Your rating:", 0.5, 5.0, 3.0, 0.5)

if st.button("Submit Rating"):
    movie_id = movies_df.loc[movies_df['title'] == selected_movie, 'movieid'].values[0]
    user_seen = [movie_id]

    recs = recommender.recommend(user_id, user_seen_movies=user_seen, k=10)

    st.subheader(f"Movies similar to '{selected_movie}'")
    
    # Display in columns (cards)
    for i in range(0, len(recs), 2):
        cols = st.columns(2)
        for j, row in enumerate(recs.iloc[i:i+2].itertuples()):
            with cols[j]:
                st.markdown(f"**🎬 {row.title}**")
                st.markdown(f"Genres: {row.genre}")
                st.markdown(f"Score: {row.hybrid_score:.3f}")
                
                # Explanation
                reason = []
                if row.cf_score > 3.5:
                    reason.append("Users similar to you liked this")
                if row.content_score > 0.2:
                    reason.append("Matches your preferred genres")
                if not reason:
                    reason.append("Exploration recommendation")
                st.markdown(f"**Reason:** {' + '.join(reason)}")
                st.markdown("---")

# ----------------------------
# Section 2: Search by Genre
# ----------------------------
st.header("2️⃣ Search Movies by Genre")

all_genres = sorted({g for gs in movies_df['genre'] for g in gs.split('|')})
selected_genre = st.multiselect("Select genres:", all_genres)

if st.button("Search by Genre"):
    if selected_genre:
        # Filter movies for selected genres
        genre_movies = movies_df[movies_df['genre'].apply(
            lambda x: any(g in x.split('|') for g in selected_genre)
        )]
        genre_movie_ids = set(genre_movies['movieid'].tolist())

        # Recommend personalized among these genre movies
        all_recs = recommender.recommend(user_id, user_seen_movies=[], k=100)
        filtered_recs = all_recs[all_recs['movieid'].isin(genre_movie_ids)].head(10)

        if filtered_recs.empty:
            st.warning("No recommendations found for these genres!")
        else:
            st.subheader(f"Top movies for: {', '.join(selected_genre)}")

            # Display in columns
            for i in range(0, len(filtered_recs), 2):
                cols = st.columns(2)
                for j, row in enumerate(filtered_recs.iloc[i:i+2].itertuples()):
                    with cols[j]:
                        st.markdown(f"**🎬 {row.title}**")
                        st.markdown(f"Genres: {row.genre}")
                        st.markdown(f"Score: {row.hybrid_score:.3f}")
                        
                        # Explanation
                        reason = []
                        if row.cf_score > 3.5:
                            reason.append("Popular among similar users")
                        if row.content_score > 0.2:
                            reason.append("Matches your liked genres")
                        if not reason:
                            reason.append("Hidden gem")
                        st.markdown(f"**Reason:** {' + '.join(reason)}")
                        st.markdown("---")
    else:
        st.warning("Please select at least one genre!")

st.header("3️⃣ Graphs")

st.info("Graphs will be displayed here based on user data and recommendations.")