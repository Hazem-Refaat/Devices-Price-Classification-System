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
    "ram": 4008,
    "sc_h": 15,
    "sc_w": 7,
    "talk_time": 20,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

response = requests.post(url, json=payload)

print(response.json())
