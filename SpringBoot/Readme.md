## FastAPI Application

The FastAPI application serves the model for making predictions. It includes an endpoint to predict the price range based on device specifications.

### app.py

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier

# load the saved CatBoost model
model = CatBoostClassifier()
model.load_model('catboost_phone_price_prediction.cbm')

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
```

## Dockerization

The application is containerized using Docker. The Dockerfile is as follows:

### Dockerfile

```dockerfile
# use an official Python runtime as a parent image
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy the current directory contents into the container at /app
COPY . .

# make port 8000 available to the world outside this container
EXPOSE 8000

# run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Running the Application

### Building the Docker Image

To build the Docker image, run the following command in the directory containing the Dockerfile:

```bash
docker build -t fastapi-catboost .
```

### Running the Docker Container

To run the Docker container, use the following command:

```bash
docker run -d -p 8000:8000 fastapi-catboost
```

### Testing the API

You can test the API using the provided `test_request.py` script or by sending a POST request to `http://127.0.0.1:8000/predict` with a JSON payload.

### test_request.py

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "battery_power": 500,
    "blue": 1,
    "clock_speed": 2.5,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 16,
    "m_dep": 0.5,
    "mobile_wt": 150,
    "n_cores": 4,
    "pc": 12,
    "px_height": 1280,
    "px_width": 720,
    "ram": 2048,
    "sc_h": 15,
    "sc_w": 7,
    "talk_time": 20,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

response = requests.post(url, json=payload)
print(response.json())
```

This script sends a POST request with device specifications and prints the predicted price range.