import pandas as pd
import re
import os

def clean_text(text):
    """
    Cleans the input text by removing special characters and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    # Remove special characters and keep only letters/numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

def main():
    print("Starting Feature Engineering...")
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(project_dir, "data", "raw")
    
    movies_path = os.path.join(data_dir, "movies.csv")
    tags_path = os.path.join(data_dir, "tags.csv")
    
    if not os.path.exists(movies_path) or not os.path.exists(tags_path):
        print("Error: Dataset files not found in raw directory.")
        return

    # 1. Load data
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)
    print(f"Loaded {len(movies)} movies and {len(tags)} tags.")

    # 2. Aggregate user tags for each movie
    # A single movie can have multiple tags from different users
    agg_tags = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()

    # 3. Merge metadata (genres) and tags
    movies_featured = pd.merge(movies, agg_tags, on="movieId", how="left")
    movies_featured["tag"] = movies_featured["tag"].fillna("")
    
    # Replace the pipe delimiter in genres with spaces
    movies_featured["genres"] = movies_featured["genres"].str.replace("|", " ", regex=False)

    # 4. Create structured text (feature soup)
    # Combine genres and tags into a single text column 'combined_features'
    movies_featured["combined_features"] = movies_featured["genres"] + " " + movies_featured["tag"]
    
    # Apply cleaning function to the combined features
    movies_featured["combined_features"] = movies_featured["combined_features"].apply(clean_text)
    
    print("\nSample Featured Text for 'Toy Story':")
    print(movies_featured.loc[movies_featured['title'].str.contains('Toy Story', na=False), 'combined_features'].values[0])

    # 5. Save the final featured dataset
    processed_dir = os.path.join(project_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "movies_featured.csv")
    movies_featured.to_csv(output_path, index=False)
    print(f"\nFeature engineering complete! Saved structured movie data to: {output_path}")

if __name__ == "__main__":
    main()
