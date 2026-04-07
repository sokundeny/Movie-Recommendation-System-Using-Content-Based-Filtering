

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.feature_engineering import main as run_feature_engineering
from ml.movie_vectorizer    import main as run_movie_vectorizer
from ml.user_vectorizer     import main as run_user_vector
from ml.recommender         import main as run_recommender

USER_ID          = 1      # Change to any userId in ratings.csv
RATING_THRESHOLD = 4.0    # Minimum rating to consider "liked"
TOP_N            = 10     # Number of movies to recommend

if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1: Feature Engineering (Leab)")
    print("=" * 60)
    run_feature_engineering()

    print()
    print("=" * 60)
    print("  STEP 2: Movie Vectorization (Methy)")
    print("=" * 60)
    run_movie_vectorizer()

    print()
    print("=" * 60)
    print("  STEP 3: User Vector (Sak)")
    print("=" * 60)
    run_user_vector(
        user_id=USER_ID,
        rating_threshold=RATING_THRESHOLD
    )

    print()
    print("=" * 60)
    print("  STEP 4: Comparison for Recommendation (Theara)")
    print("=" * 60)
    run_recommender(
        user_id=USER_ID,
        top_n=TOP_N,
        exclude_seen=True,
    )

    print()
    print("✓ Pipeline (Step 1 → 4) complete!")