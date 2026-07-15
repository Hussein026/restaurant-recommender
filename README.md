# Restaurant Recommendation System

Hybrid recommender (collaborative + content-based + location) served via FastAPI.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Database (PostgreSQL)

```bash
createdb restaurant_db
psql restaurant_db -f init_db.sql
cp .env.example .env   # edit DATABASE_URL if needed
```

## Seed sample data

```bash
python -m data.seed_data
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Docs at: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | /users | Create a user |
| POST | /restaurants | Create a restaurant (retrains model) |
| POST | /ratings | Add/update a rating (retrains model) |
| POST | /retrain | Force model retrain |
| GET | /recommend/{user_id}?top_n=10&max_km=10 | Get hybrid recommendations |
| GET | /health | Health check |

## How the hybrid model works

1. **Collaborative filtering** (`app/recommender/collaborative.py`): SVD matrix
   factorization via `scikit-surprise`, trained on the user x restaurant rating
   matrix. Predicts a rating for any (user, restaurant) pair.
2. **Content-based filtering** (`app/recommender/content_based.py`): TF-IDF
   over `cuisine + tags` per restaurant (scikit-learn). A user profile vector
   is the rating-weighted average of restaurants they liked; cosine similarity
   scores all other restaurants against it. A haversine-distance score adds
   location relevance.
3. **Hybrid blend** (`app/recommender/hybrid.py`): weighted sum
   `w_cf * cf + w_cb * cb + w_loc * loc` (defaults 0.5 / 0.3 / 0.2).
   For a brand-new user with no ratings, the collaborative weight is
   automatically redistributed to content-based + location (cold start).
