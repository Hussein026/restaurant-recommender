# Restaurant Recommendation Engine — Hybrid AI Recommender

> Hybrid recommendation system combining collaborative filtering, content-based filtering, and location scoring — served via FastAPI.

---

## How the Hybrid Model Works

Three signals combined into one weighted score:

| Signal | Algorithm | Default Weight |
|---|---|---|
| Collaborative Filtering | SVD matrix factorization (scikit-surprise) | 0.5 |
| Content-Based Filtering | TF-IDF + cosine similarity (scikit-learn) | 0.3 |
| Location Scoring | Haversine distance | 0.2 |

**Final score:** `0.5 × CF + 0.3 × CB + 0.2 × location`

**Cold-start handling:** New users with no ratings automatically get collaborative weight redistributed to content-based + location — no blank recommendations.

---

## Collaborative Filtering
- SVD matrix factorization on the user × restaurant rating matrix
- Predicts a rating for any (user, restaurant) pair
- Retrains automatically when new ratings are added

## Content-Based Filtering
- TF-IDF vectorization over cuisine type + tags per restaurant
- User profile = rating-weighted average of restaurants they liked
- Cosine similarity scores all unrated restaurants against the user profile
- Haversine distance adds location relevance

---

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Database | PostgreSQL / SQLite |
| Collaborative Filtering | scikit-surprise (SVD) |
| Content-Based Filtering | scikit-learn (TF-IDF, cosine similarity) |
| Location Scoring | Haversine formula |
| Language | Python |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/users` | Create a user |
| POST | `/restaurants` | Add a restaurant (retrains model) |
| POST | `/ratings` | Add/update a rating (retrains model) |
| POST | `/retrain` | Force model retrain |
| GET | `/recommend/{user_id}?top_n=10&max_km=10` | Get hybrid recommendations |
| GET | `/health` | Health check |

Docs auto-generated at: `http://localhost:8000/docs`

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Database:**
```bash
createdb restaurant_db
psql restaurant_db -f init_db.sql
cp .env.example .env
python -m data.seed_data
```

**Run:**
```bash
uvicorn app.main:app --reload
```

---

## Project Structure

```
restaurant-recommender/
├── app/
│   ├── recommender/
│   │   ├── collaborative.py   # SVD matrix factorization
│   │   ├── content_based.py   # TF-IDF + cosine similarity
│   │   └── hybrid.py          # Weighted blend + cold-start
│   ├── main.py                # FastAPI app
│   ├── models.py
│   ├── schemas.py
│   ├── crud.py
│   └── database.py
├── data/
│   └── seed_data.py
├── requirements.txt
└── init_db.sql
```

---

## Author

**Hussein Mokhadder**  
Computer Engineering Student — IoT & AI  
Politehnica University of Bucharest  
[github.com/Hussein026](https://github.com/Hussein026)
