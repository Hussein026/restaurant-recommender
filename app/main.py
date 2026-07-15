from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import models, schemas, crud
from app.database import engine, get_db
from app.recommender.hybrid import HybridRecommender

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Restaurant Recommendation System")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

# One shared hybrid model instance, retrained on demand
hybrid_model = HybridRecommender(w_cf=0.5, w_cb=0.3, w_loc=0.2)
MODEL_STATE = {"trained": False}


def train_model(db: Session):
    ratings_df = crud.get_all_ratings_df(db)
    restaurants_df = crud.get_all_restaurants_df(db)
    if restaurants_df.empty:
        return
    hybrid_model.fit(ratings_df, restaurants_df)
    MODEL_STATE["trained"] = True


@app.on_event("startup")
def startup_event():
    db = next(get_db())
    train_model(db)


@app.post("/users", response_model=schemas.UserOut)
def add_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, user)


@app.post("/restaurants", response_model=schemas.RestaurantOut)
def add_restaurant(restaurant: schemas.RestaurantCreate, db: Session = Depends(get_db)):
    obj = crud.create_restaurant(db, restaurant)
    train_model(db)
    return obj


@app.post("/ratings")
def add_rating(rating: schemas.RatingCreate, db: Session = Depends(get_db)):
    obj = crud.create_rating(db, rating)
    train_model(db)
    return {"status": "ok", "rating_id": obj.id}


@app.post("/retrain")
def retrain(db: Session = Depends(get_db)):
    train_model(db)
    return {"status": "retrained", "trained": MODEL_STATE["trained"]}


@app.get("/recommend/{user_id}", response_model=list[schemas.RecommendationOut])
def recommend(user_id: int, top_n: int = 10, max_km: float = 10.0, db: Session = Depends(get_db)):
    if not MODEL_STATE["trained"]:
        train_model(db)
    if not MODEL_STATE["trained"]:
        raise HTTPException(status_code=400, detail="No restaurant data available yet")

    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    rated_ids, rated_values = crud.get_user_ratings(db, user_id)
    results = hybrid_model.recommend(
        user_id=user_id,
        user_lat=user.latitude,
        user_lon=user.longitude,
        rated_restaurant_ids=rated_ids,
        rated_values=rated_values,
        top_n=top_n,
        max_km=max_km,
    )

    output = []
    for rid, score in results:
        r = crud.get_restaurant(db, rid)
        dist = None
        if user.latitude is not None and r.latitude is not None:
            from app.recommender.content_based import ContentBasedRecommender
            dist = ContentBasedRecommender.haversine_km(user.latitude, user.longitude, r.latitude, r.longitude)
        output.append(schemas.RecommendationOut(
            restaurant_id=r.id, name=r.name, cuisine=r.cuisine, score=round(score, 4),
            distance_km=round(dist, 2) if dist is not None else None
        ))
    return output


@app.get("/health")
def health():
    return {"status": "ok", "model_trained": MODEL_STATE["trained"]}
