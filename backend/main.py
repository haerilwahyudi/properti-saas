from fastapi import FastAPI, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

app = FastAPI()

# Izinkan CORS (untuk frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Koneksi ke Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Load model ML (latih dulu)
model = joblib.load("model_properti.pkl")

@app.post("/predict")
async def predict(file: UploadFile):
    # Baca data CSV
    df = pd.read_csv(file.file)
    
    # Prediksi harga
    df["predicted_price"] = model.predict(df[["size_sqm", "bedrooms", "distance_to_city_center"]])
    
    # Simpan ke Supabase
    for _, row in df.iterrows():
        data = {
            "location": row["location"],
            "price": int(row["predicted_price"]),
            "size_sqm": int(row["size_sqm"]),
            "bedrooms": int(row["bedrooms"]),
            "distance_to_city_center": int(row["distance_to_city_center"])
        }
        supabase.table("properties").insert(data).execute()
    
    return {"prediksi": df.to_dict()}

@app.get("/properties")
def get_properties():
    data = supabase.table("properties").select("*").execute()
    return data.data

@app.get("/")
def home():
    return {"message": "Properti SaaS API is running!"}