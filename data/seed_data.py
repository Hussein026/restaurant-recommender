"""
Run: python -m data.seed_data
Populates the DB with sample users, restaurants, and ratings so the
recommender has something to train on immediately.
"""
import random
from app.database import SessionLocal, engine, Base
from app import models

Base.metadata.create_all(bind=engine)

USERS = [
    {"name": "Alice", "latitude": 44.4268, "longitude": 26.1025},
    {"name": "Bob", "latitude": 44.4325, "longitude": 26.1039},
    {"name": "Carla", "latitude": 44.4396, "longitude": 26.0963},
]

RESTAURANTS = [
    {"name": "Trattoria Bella", "cuisine": "italian", "tags": "pasta romantic wine cozy", "price_range": 3, "latitude": 44.4270, "longitude": 26.1030},
    {"name": "Sushi Zen", "cuisine": "japanese", "tags": "sushi fresh minimalist quiet", "price_range": 4, "latitude": 44.4300, "longitude": 26.1000},
    {"name": "Spicy Wok", "cuisine": "chinese", "tags": "spicy noodles casual quick", "price_range": 2, "latitude": 44.4350, "longitude": 26.1100},
    {"name": "Green Bowl", "cuisine": "vegan", "tags": "healthy vegan salad cheap", "price_range": 1, "latitude": 44.4200, "longitude": 26.0950},
    {"name": "Burger Yard", "cuisine": "american", "tags": "burgers casual fast cheap", "price_range": 1, "latitude": 44.4400, "longitude": 26.1050},
    {"name": "Le Petit Paris", "cuisine": "french", "tags": "romantic wine fine-dining", "price_range": 4, "latitude": 44.4310, "longitude": 26.0990},
    {"name": "Taco Fiesta", "cuisine": "mexican", "tags": "spicy tacos casual cheap", "price_range": 2, "latitude": 44.4380, "longitude": 26.1080},
    {"name": "Curry House", "cuisine": "indian", "tags": "spicy curry vegan-options cozy", "price_range": 2, "latitude": 44.4250, "longitude": 26.1010},
]


def run():
    db = SessionLocal()

    user_objs = [models.User(**u) for u in USERS]
    db.add_all(user_objs)
    db.commit()
    for u in user_objs:
        db.refresh(u)

    restaurant_objs = [models.Restaurant(**r) for r in RESTAURANTS]
    db.add_all(restaurant_objs)
    db.commit()
    for r in restaurant_objs:
        db.refresh(r)

    random.seed(42)
    ratings = []
    for u in user_objs:
        sample = random.sample(restaurant_objs, k=5)
        for r in sample:
            ratings.append(models.Rating(user_id=u.id, restaurant_id=r.id, rating=round(random.uniform(2.5, 5.0), 1)))

    db.add_all(ratings)
    db.commit()
    db.close()
    print(f"Seeded {len(user_objs)} users, {len(restaurant_objs)} restaurants, {len(ratings)} ratings.")


if __name__ == "__main__":
    run()
