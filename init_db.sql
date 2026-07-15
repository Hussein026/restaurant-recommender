CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    latitude FLOAT,
    longitude FLOAT
);

CREATE TABLE IF NOT EXISTS restaurants (
    id SERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    cuisine VARCHAR(100),
    tags TEXT,               -- space separated keywords e.g. "spicy vegan cheap romantic"
    price_range SMALLINT,    -- 1-4
    latitude FLOAT,
    longitude FLOAT
);

CREATE TABLE IF NOT EXISTS ratings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    restaurant_id INTEGER REFERENCES restaurants(id),
    rating FLOAT CHECK (rating >= 1 AND rating <= 5),
    UNIQUE(user_id, restaurant_id)
);

CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_restaurant ON ratings(restaurant_id);
