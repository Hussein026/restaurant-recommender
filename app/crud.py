import pandas as pd
from sqlalchemy.orm import Session
from app import models, schemas


def create_user(db: Session, user: schemas.UserCreate):
    obj = models.User(**user.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def create_restaurant(db: Session, restaurant: schemas.RestaurantCreate):
    obj = models.Restaurant(**restaurant.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def create_rating(db: Session, rating: schemas.RatingCreate):
    existing = db.query(models.Rating).filter_by(
        user_id=rating.user_id, restaurant_id=rating.restaurant_id
    ).first()
    if existing:
        existing.rating = rating.rating
        db.commit()
        db.refresh(existing)
        return existing
    obj = models.Rating(**rating.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_all_ratings_df(db: Session) -> pd.DataFrame:
    rows = db.query(models.Rating).all()
    data = [{"user_id": r.user_id, "restaurant_id": r.restaurant_id, "rating": r.rating} for r in rows]
    return pd.DataFrame(data, columns=["user_id", "restaurant_id", "rating"])


def get_all_restaurants_df(db: Session) -> pd.DataFrame:
    rows = db.query(models.Restaurant).all()
    data = [{
        "id": r.id, "name": r.name, "cuisine": r.cuisine, "tags": r.tags,
        "price_range": r.price_range, "latitude": r.latitude, "longitude": r.longitude
    } for r in rows]
    return pd.DataFrame(data, columns=["id", "name", "cuisine", "tags", "price_range", "latitude", "longitude"])


def get_user_ratings(db: Session, user_id: int):
    rows = db.query(models.Rating).filter(models.Rating.user_id == user_id).all()
    ids = [r.restaurant_id for r in rows]
    values = [r.rating for r in rows]
    return ids, values


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_restaurant(db: Session, restaurant_id: int):
    return db.query(models.Restaurant).filter(models.Restaurant.id == restaurant_id).first()
