"""
Loads a FREE, real-world, public dataset into the database:
  - 130 real restaurants (Mexico), with real lat/lon and cuisines
  - 138 real consumers
  - 1161 real user -> restaurant ratings

Source: UCI Machine Learning Repository "Restaurant & consumer data"
        (mirrored as CSVs on GitHub by liyenhsu/restaurant-data-with-consumer-ratings)
No API key, no billing, no credit card needed.

Run:
  python -m data.load_uci_dataset
"""
import pandas as pd
import requests

from app.database import SessionLocal, engine, Base
from app import models

BASE_URL = "https://raw.githubusercontent.com/liyenhsu/restaurant-data-with-consumer-ratings/master/data"

PRICE_MAP = {"low": 1, "medium": 2, "high": 4}


def download_csv(name: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{name}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(resp.text))


def run():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    print("Downloading dataset from GitHub...")
    places = download_csv("geoplaces2.csv")
    cuisines = download_csv("chefmozcuisine.csv")
    ratings = download_csv("rating_final.csv")
    users = download_csv("userprofile.csv")

    cuisine_map = cuisines.groupby("placeID")["Rcuisine"].apply(lambda s: " ".join(s)).to_dict()

    place_id_to_db_id = {}
    inserted_restaurants = 0
    for _, row in places.iterrows():
        place_id = row["placeID"]
        tags = cuisine_map.get(place_id, "")
        cuisine = tags.split(" ")[0] if tags else "unspecified"

        existing = db.query(models.Restaurant).filter_by(name=row["name"], latitude=row["latitude"]).first()
        if existing:
            place_id_to_db_id[place_id] = existing.id
            continue

        obj = models.Restaurant(
            name=row["name"],
            cuisine=cuisine,
            tags=tags,
            price_range=PRICE_MAP.get(row["price"], 2),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        db.add(obj)
        db.commit()
        db.refresh(obj)
        place_id_to_db_id[place_id] = obj.id
        inserted_restaurants += 1

    user_id_to_db_id = {}
    inserted_users = 0
    for _, row in users.iterrows():
        uid = row["userID"]
        existing = db.query(models.User).filter_by(name=uid).first()
        if existing:
            user_id_to_db_id[uid] = existing.id
            continue

        obj = models.User(name=uid, latitude=row["latitude"], longitude=row["longitude"])
        db.add(obj)
        db.commit()
        db.refresh(obj)
        user_id_to_db_id[uid] = obj.id
        inserted_users += 1

    inserted_ratings = 0
    for _, row in ratings.iterrows():
        db_user_id = user_id_to_db_id.get(row["userID"])
        db_place_id = place_id_to_db_id.get(row["placeID"])
        if db_user_id is None or db_place_id is None:
            continue

        rescaled = 1 + (row["rating"] * 2)  # 0->1, 1->3, 2->5

        existing = db.query(models.Rating).filter_by(user_id=db_user_id, restaurant_id=db_place_id).first()
        if existing:
            continue

        db.add(models.Rating(user_id=db_user_id, restaurant_id=db_place_id, rating=rescaled))
        inserted_ratings += 1

    db.commit()
    db.close()

    print(f"Inserted {inserted_restaurants} restaurants, {inserted_users} users, {inserted_ratings} ratings.")


if __name__ == "__main__":
    run()