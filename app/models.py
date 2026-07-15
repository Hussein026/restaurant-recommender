from sqlalchemy import Column, Integer, String, Float, SmallInteger, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    latitude = Column(Float)
    longitude = Column(Float)

    ratings = relationship("Rating", back_populates="user")


class Restaurant(Base):
    __tablename__ = "restaurants"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    cuisine = Column(String(100))
    tags = Column(String)  # space separated keywords
    price_range = Column(SmallInteger)
    latitude = Column(Float)
    longitude = Column(Float)

    ratings = relationship("Rating", back_populates="restaurant")


class Rating(Base):
    __tablename__ = "ratings"
    __table_args__ = (UniqueConstraint("user_id", "restaurant_id", name="uq_user_restaurant"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    restaurant_id = Column(Integer, ForeignKey("restaurants.id"))
    rating = Column(Float, nullable=False)

    user = relationship("User", back_populates="ratings")
    restaurant = relationship("Restaurant", back_populates="ratings")
