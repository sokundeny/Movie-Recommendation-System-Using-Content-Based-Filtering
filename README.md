# Movie Recommendation System Using Content-Based Filtering

Movie Recommendation system based on user like and historical movie

## Objective 
The main objectives of this project are:
1. To collect movie rating and metadata datasets
2. To remove incorrect or repeated data.
3. To analyst movie ratings to understand which movies are popular.
4. Implement a Content-Based Filtering system to recommend movies based on
similarity
5. Visualize top-rated movies and recommendations

---

## Dataset From MovieLens

### Before can run we should
1. Download DataSet
2. Clean data.
3. Process Data.

## Backend Setup (FastAPI)

### Requirements

- Python 3.9+
- pip packages:

```bash
pip install fastapi uvicorn pandas numpy scipy requests scikit-learn python-multipart
```
Run

```
uvicorn backend.app:app --reload
```

## Frontend Setup (Vue)

### Requirements

change directory to frontend

```bash
cd frontend
```


```bash
npm install
```
Run

```
npm run dev
```
