from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier

# load the saved CatBoost model
model = CatBoostClassifier()
model.load_model('./Model/catboost_phone_price_prediction.cbm')

app = FastAPI()

class DeviceSpec(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.post("/predict")
def predict_price(device: DeviceSpec):
    device_data = pd.DataFrame([device.dict()])
    prediction = model.predict(device_data)
    return {"price_range": int(prediction[0])}

# you can run the app using `uvicorn` or similar ASGI server
