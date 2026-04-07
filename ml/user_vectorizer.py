import numpy as np
import pandas as pd
from scipy.sparse import load_npz
import os
import pickle


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_artifacts(processed_dir: str):
    """Load movie matrix, movie index, and vectorizer produced by Methy's step."""
    matrix_path     = os.path.join(processed_dir, "movie_vectors.npz")
    index_path      = os.path.join(processed_dir, "movie_index.csv")
    vectorizer_path = os.path.join(processed_dir, "vectorizer.pkl")

    print(f"\nLooking for artifacts in: {processed_dir}")

    for p in [matrix_path, index_path, vectorizer_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                "Please run feature_engineering.py (Leab) and movie_vectorizer.py (Methy) first."
            )

    movie_matrix = load_npz(matrix_path)
    movie_index  = pd.read_csv(index_path)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    print(f"Loaded movie matrix  : {movie_matrix.shape}")
    print(f"Loaded movie index   : {len(movie_index)} titles")

    return movie_matrix, movie_index, vectorizer


def get_liked_movies(
    ratings_path: str,
    user_id: int,
    rating_threshold: float = 4.0,
) -> list[int]:

    print(f"\nLooking for ratings at: {ratings_path}")

    if os.path.exists(ratings_path):
        ratings = pd.read_csv(ratings_path)
        liked = ratings[
            (ratings["userId"] == user_id) &
            (ratings["rating"] >= rating_threshold)
        ]["movieId"].tolist()

        print(f"User {user_id}: found {len(liked)} liked movies (rating >= {rating_threshold})")
        return liked
    else:
        demo_ids = [1, 2, 3, 10, 32]
        print(f"[Demo mode] ratings.csv not found — using demo movie IDs: {demo_ids}")
        return demo_ids


def build_user_vector(
    liked_movie_ids: list[int],
    movie_matrix,
    movie_index: pd.DataFrame,
) -> np.ndarray:

    id_to_row = dict(zip(movie_index["movieId"], movie_index["vector_row"]))

    matched_rows = []
    skipped = []

    for mid in liked_movie_ids:
        if mid in id_to_row:
            matched_rows.append(id_to_row[mid])
        else:
            skipped.append(mid)

    if skipped:
        print(f"⚠ Skipped {len(skipped)} movies not found in index: {skipped[:5]}")

    if not matched_rows:
        raise ValueError("No liked movies found in movie index.")

    liked_vectors = movie_matrix[matched_rows].toarray()

    print(f"\nBuilding user vector from {len(matched_rows)} movies...")
    print(f"Each movie vector has {liked_vectors.shape[1]} features")

    user_vector = liked_vectors.mean(axis=0)

    print(f"User vector shape : {user_vector.shape}")
    print(f"Non-zero features : {np.count_nonzero(user_vector)}")

    return user_vector


def save_user_vector(user_vector: np.ndarray, processed_dir: str, user_id: int):
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, f"user_vector_{user_id}.npy")

    np.save(path, user_vector)

    print(f"\nSaved user vector → {path}")
    return path


def describe_user_taste(user_vector: np.ndarray, vectorizer, top_n: int = 10):
    feature_names = vectorizer.get_feature_names_out()
    top_idx = user_vector.argsort()[::-1][:top_n]

    print(f"\n── Top-{top_n} taste keywords ──")

    for rank, i in enumerate(top_idx, 1):
        if user_vector[i] > 0:
            print(f"{rank:2d}. '{feature_names[i]}' → {user_vector[i]:.4f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(user_id: int = 1, rating_threshold: float = 4.0):
    print("=" * 60)
    print("  USER VECTOR — Sak's Task")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    processed_dir = os.path.join(project_root, "data", "processed")
    ratings_path  = os.path.join(project_root, "data", "raw", "ratings.csv")

    print(f"\nProject root : {project_root}")
    print(f"Processed dir: {processed_dir}")

    # 1. Load artifacts
    movie_matrix, movie_index, vectorizer = load_artifacts(processed_dir)

    # 2. Get liked movies
    liked_ids = get_liked_movies(ratings_path, user_id, rating_threshold)

    if not liked_ids:
        print("No liked movies found. Exiting.")
        return

    # 3. Build vector
    user_vector = build_user_vector(liked_ids, movie_matrix, movie_index)

    # 4. Describe taste
    describe_user_taste(user_vector, vectorizer)

    # 5. Save
    save_user_vector(user_vector, processed_dir, user_id)

    print("\n✓ User Vector complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=4.0)

    args = parser.parse_args()

    main(user_id=args.user_id, rating_threshold=args.threshold)