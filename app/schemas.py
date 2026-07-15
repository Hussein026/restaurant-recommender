from pydantic import BaseModel
from typing import Optional


class UserCreate(BaseModel):
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class UserOut(UserCreate):
    id: int

    class Config:
        from_attributes = True


class RestaurantCreate(BaseModel):
    name: str
    cuisine: str
    tags: str
    price_range: int
    latitude: float
    longitude: float


class RestaurantOut(RestaurantCreate):
    id: int

    class Config:
        from_attributes = True


class RatingCreate(BaseModel):
    user_id: int
    restaurant_id: int
    rating: float


class RecommendationOut(BaseModel):
    restaurant_id: int
    name: str
    cuisine: str
    score: float
    distance_km: Optional[float] = None
