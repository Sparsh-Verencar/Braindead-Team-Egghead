import pickle
import pandas as pd

# 1. IMPORTANT: The class definition MUST be present for pickle to load it.
# If you saved the class in a file named 'recommender_logic.py', use:
# from recommender_logic import FinalReelSenseRecommender

class FinalReelSenseRecommender:
    # (Ensure the full class code you wrote is here or imported)
    pass 

def get_movie_suggestions(user_id, model_path='models/reelsense_model.pkl', top_n=10):
    """Loads the pickled model and returns recommendations for a specific user."""
    try:
        # Load the serialized model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"--- Model Loaded Successfully ---")
        
        # Generate recommendations
        recommendations = model.get_recommendations(user_id, top_n=top_n)
        
        return recommendations

    except FileNotFoundError:
        print(f"Error: The file '{model_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    USER_ID = 1  # Replace with the ID you want to test
    
    results = get_movie_suggestions(USER_ID, top_n=5)
    
    if results is not None:
        print(f"\nTop 5 Recommendations for User {USER_ID}:")
        print(results[['title', 'genres', 'hybrid_score']])
        