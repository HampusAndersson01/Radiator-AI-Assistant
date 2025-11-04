import requests
from datetime import datetime, timedelta

def get_weather(lat=59.5, lon=16):
    """Return current outside temp and +3h forecast using SMHI API.

    Returns (current, forecast_3h)
    
    SMHI API returns hourly forecasts with 't' (temperature in Celsius).
    Free, no API key required.
    """
    # Round coordinates to SMHI's grid (they use specific points)
    lat_rounded = round(lat, 6)
    lon_rounded = round(lon, 6)
    
    url = f"https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/{lon_rounded}/lat/{lat_rounded}/data.json"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        # SMHI returns timeSeries array with hourly forecasts
        time_series = data.get("timeSeries", [])
        if not time_series:
            return None, None
        
        # Find current temp (first forecast point is usually now or very recent)
        current_temp = None
        forecast_3h_temp = None
        
        now = datetime.utcnow()
        target_3h = now + timedelta(hours=3)
        
        for i, forecast in enumerate(time_series):
            # Parse validTime: "2025-11-04T12:00:00Z"
            valid_time = datetime.fromisoformat(forecast["validTime"].replace("Z", "+00:00"))
            
            # Extract temperature from parameters
            params = forecast.get("parameters", [])
            temp = None
            for param in params:
                if param.get("name") == "t":  # 't' is temperature in Celsius
                    temp = param.get("values", [None])[0]
                    break
            
            if temp is None:
                continue
            
            # First valid temp is current
            if current_temp is None:
                current_temp = temp
            
            # Find closest forecast to +3h
            if forecast_3h_temp is None and valid_time >= target_3h:
                forecast_3h_temp = temp
                break
        
        # Fallback if we didn't find +3h forecast
        if forecast_3h_temp is None and len(time_series) > 3:
            for param in time_series[3].get("parameters", []):
                if param.get("name") == "t":
                    forecast_3h_temp = param.get("values", [None])[0]
                    break
        
        # If still no forecast, use current
        if forecast_3h_temp is None:
            forecast_3h_temp = current_temp
        
        return current_temp, forecast_3h_temp
        
    except Exception as e:
        print(f"SMHI API error: {e}")
        return None, None
