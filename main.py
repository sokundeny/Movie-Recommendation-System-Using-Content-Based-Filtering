from ml.feature_engineering import main as run_feature_engineering
from ml.movie_vectorizer import main as run_movie_vectorizer

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
    print("✓ Pipeline complete! Ready for brro Sak and b Srey.")