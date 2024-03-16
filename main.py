from fastapi import FastAPI
from ai_engine.vehicle_and_license_plate_recognition import VehicleLicensePlateRecognition
from data_models.data_models import Img
import os
import dotenv

dotenv.load_dotenv()

app = FastAPI()

vlpr = VehicleLicensePlateRecognition()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/vehicle-license-plate-recognition/")
async def vehicle_license_plate_recognition(img: Img):
    img = img.path
    ret = vlpr.predict(img)
    if len(ret) == 0:
        vehicle = None
        license_plate = None
    else:
        ret = ret[0]
        vehicle = ret[1]
        license_plate = ret[2]
    return {"vehicle": vehicle, "license_plate": license_plate}