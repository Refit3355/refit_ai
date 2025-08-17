import os
from typing import Tuple, Optional
import requests
import numpy as np
import pandas as pd
import math as _math

DEFAULT_TZ = os.getenv("DEFAULT_TZ", "Asia/Seoul")

OPENMETEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

def _nz(x, default):
    try:
        if x is None:
            return default
        v = float(x)
        if _math.isfinite(v):
            return v
    except Exception:
        pass
    return default

def geocode_city(name: str, country_code: Optional[str] = "KR") -> Tuple[float, float, dict]:
    candidates = [{"name": name, "country": country_code}, {"name": name, "country": None}]
    if name in ("서울","서울시","서울특별시"):
        candidates += [{"name":"Seoul","country":"KR"}, {"name":"Seoul","country":None}]
    last_err = None
    for c in candidates:
        try:
            params = {"name": c["name"], "count": 1, "language": "ko", "format": "json"}
            if c["country"]:
                params["country"] = c["country"]
            r = requests.get(OPENMETEO_GEOCODE_URL, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("results"):
                res = data["results"][0]
                return float(res["latitude"]), float(res["longitude"]), res
        except Exception as e:
            last_err = e
    if name in ("서울","서울시","서울특별시","Seoul"):
        return 37.5665, 126.9780, {"name":"Seoul (fallback)"}
    raise ValueError(f"지오코딩 실패: '{name}' (last_err={last_err})")

def fetch_weather_ctx(location_name: str = "서울", tz: str = DEFAULT_TZ) -> dict:
    try:
        lat, lon, meta = geocode_city(location_name, country_code="KR")
    except Exception:
        return {"uvi": 0.0, "humidity": 50.0, "temp": 20.0, "pm25": 10.0}

    forecast_params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "hourly": ["temperature_2m", "relative_humidity_2m", "uv_index"],
        "daily": ["uv_index_max"],
    }
    air_params = {"latitude": lat, "longitude": lon, "timezone": tz, "hourly": ["pm2_5"]}

    fjson, ajson = {}, {}
    try:
        fr = requests.get(OPENMETEO_FORECAST_URL, params=forecast_params, timeout=10)
        fr.raise_for_status()
        fjson = fr.json()
    except Exception:
        pass
    try:
        ar = requests.get(OPENMETEO_AIR_URL, params=air_params, timeout=10)
        ar.raise_for_status()
        ajson = ar.json()
    except Exception:
        pass

    uvi = 0.0
    try:
        uvi_list = fjson.get("daily", {}).get("uv_index_max", [])
        if uvi_list:
            uvi = _nz(uvi_list[0], 0.0)
    except Exception:
        pass

    humidity, temp = 50.0, 20.0
    try:
        hourly = fjson.get("hourly", {})
        times = hourly.get("time", [])
        hums  = hourly.get("relative_humidity_2m", [])
        temps = hourly.get("temperature_2m", [])
        if times and hums and temps:
            now = pd.Timestamp.now(tz)
            ts = pd.to_datetime(times)
            idx = int(np.argmin(np.abs((ts - now).astype("timedelta64[m]").astype(int))))
            humidity = _nz(hums[idx], 50.0)
            temp     = _nz(temps[idx], 20.0)
    except Exception:
        pass

    pm25 = 10.0
    try:
        pms = ajson.get("hourly", {}).get("pm2_5", [])
        if pms:
            pm25 = _nz(pms[-1], 10.0)
    except Exception:
        pass

    return {"uvi": uvi, "humidity": humidity, "temp": temp, "pm25": pm25}
