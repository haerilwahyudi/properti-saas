import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Data dummy properti
data = {
    "size_sqm": [120, 60, 90, 150],
    "bedrooms": [3, 2, 2, 4],
    "distance_to_city_center": [5, 20, 10, 15],
    "price": [2000000000, 800000000, 1500000000, 3000000000]
}
df = pd.DataFrame(data)

# Latih model
model = RandomForestRegressor()
model.fit(df[["size_sqm", "bedrooms", "distance_to_city_center"]], df["price"])

# Simpan model
joblib.dump(model, "model_properti.pkl")