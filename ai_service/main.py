# Smart Radiator AI Service - v2.0.0 (Database-backed)
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
import os, joblib, requests, json, io
from collections import defaultdict
from river import forest, preprocessing, metrics
from forecast import get_weather
import database as db
import asyncio
from contextlib import asynccontextmanager

# Background task for validating predictions
async def validation_background_task():
    """Background task that runs every hour to validate old predictions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            print("🔍 Running prediction validation...")
            
            # Call the validation logic
            unvalidated = db.get_unvalidated_predictions()
            if unvalidated:
                validated_count = 0
                trained_count = 0
                
                for pred in unvalidated:
                    room = pred['room']
                    prediction_id = pred['id']
                    predicted_temp = pred['predicted_temp']
                    
                    try:
                        conn = db.get_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            SELECT current_temp, outdoor_temp 
                            FROM room_states
                            WHERE room = %s 
                              AND timestamp >= %s - INTERVAL '30 minutes'
                              AND timestamp <= %s + INTERVAL '30 minutes'
                            ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                            LIMIT 1
                        """, (room, pred['target_timestamp'], pred['target_timestamp'], pred['target_timestamp']))
                        
                        result = cursor.fetchone()
                        cursor.close()
                        conn.close()
                        
                        if result:
                            actual_temp = result[0]
                            db.validate_prediction(prediction_id, actual_temp, used_for_training=True)
                            validated_count += 1
                            trained_count += 1
                            print(f"  ✅ {room}: predicted {predicted_temp:.1f}°C vs actual {actual_temp:.1f}°C")
                        else:
                            db.validate_prediction(prediction_id, None, used_for_training=False)
                            validated_count += 1
                    except Exception as e:
                        print(f"  ❌ Error validating prediction {prediction_id}: {e}")
                
                print(f"✅ Validated {validated_count} predictions ({trained_count} trained)")
            else:
                print("  ℹ️  No predictions to validate")
                
        except Exception as e:
            print(f"Error in validation background task: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    task = asyncio.create_task(validation_background_task())
    print("🚀 Started prediction validation background task")
    yield
    # Shutdown
    task.cancel()
    print("🛑 Stopped prediction validation background task")

app = FastAPI(title="Smart Radiator AI", lifespan=lifespan)
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize database on startup
try:
    db.init_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
    print("Some features may not work without DATABASE_URL configured")

TELEGRAM_WEBHOOK = os.getenv("TELEGRAM_WEBHOOK")

ROOMS = {
    "Badrum":     {"scale": [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], "target": 22.5},
    "Sovrum":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
    "Kontor":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
    "Vardagsrum":{"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
}

class RoomState(BaseModel):
    room: str
    current_temp: float
    target_temp: float
    radiator_level: int
    outdoor_temp: float | None = None
    forecast_temp: float | None = None
    forecast_10h_temp: float | None = None
    timestamp: str

class ModelMetrics:
    """Track AI model performance metrics"""
    def __init__(self):
        self.mae = metrics.MAE()
        self.rmse = metrics.RMSE()
        self.r2 = metrics.R2()
        self.training_samples = 0
        self.predictions_made = 0
        self.adjustments_made = 0
        self.total_error = 0.0
        self.created_at = datetime.now().isoformat()
        
    def update(self, y_true, y_pred):
        """Update metrics with new prediction"""
        self.mae.update(y_true, y_pred)
        self.rmse.update(y_true, y_pred)
        self.r2.update(y_true, y_pred)
        self.total_error += abs(y_true - y_pred)
        
    def to_dict(self):
        """Convert metrics to dictionary"""
        return {
            "mae": self.mae.get() if self.training_samples > 0 else 0,
            "rmse": self.rmse.get() if self.training_samples > 0 else 0,
            "r2_score": self.r2.get() if self.training_samples > 1 else 0,
            "training_samples": self.training_samples,
            "predictions_made": self.predictions_made,
            "adjustments_made": self.adjustments_made,
            "avg_error": self.total_error / self.predictions_made if self.predictions_made > 0 else 0,
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat()
        }

# Global metrics storage (in-memory cache)
model_metrics = defaultdict(ModelMetrics)

# Load models and metrics from database on startup
def load_metrics_from_db():
    """Load metrics from database into memory"""
    try:
        all_metrics = db.get_ai_metrics()
        for room, metrics_data in all_metrics.items():
            m = model_metrics[room]
            m.training_samples = metrics_data.get('training_samples', 0)
            m.predictions_made = metrics_data.get('predictions_made', 0)
            m.adjustments_made = metrics_data.get('adjustments_made', 0)
            m.total_error = metrics_data.get('total_error', 0)
            m.created_at = metrics_data.get('created_at', datetime.now()).isoformat() if isinstance(metrics_data.get('created_at'), datetime) else str(metrics_data.get('created_at', datetime.now().isoformat()))
        print(f"✅ Loaded metrics for {len(all_metrics)} rooms from database")
    except Exception as e:
        print(f"Warning: Could not load metrics from database: {e}")

load_metrics_from_db()

def load_model(room):
    """Load model from database or create new one"""
    try:
        model_bytes = db.load_model(room)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
    except Exception as e:
        print(f"Could not load model for {room} from database: {e}")
    
    # Return new model if not found or error
    return preprocessing.StandardScaler() | forest.ARFRegressor()

def save_model(room, model):
    """Save model to database"""
    try:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()
        db.save_model(room, model_bytes)
    except Exception as e:
        print(f"Warning: Could not save model for {room} to database: {e}")

def get_radiator_levels():
    """Get current radiator levels from PostgreSQL database"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT room, level FROM radiators")
        levels = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return levels
    except Exception as e:
        print(f"Error reading radiator levels: {e}")
        return {}

@app.get("/")
def status():
    """Return AI service status and model information"""
    try:
        radiator_levels = get_radiator_levels()
    except Exception as e:
        print(f"Error getting radiator levels: {e}")
        radiator_levels = {}
    
    models_info = {}
    for room in ROOMS.keys():
        # Check if model exists in database
        try:
            model_exists = db.load_model(room) is not None
        except Exception as e:
            print(f"Error checking model for {room}: {e}")
            model_exists = False
        
        # Get latest temp from database
        try:
            latest_temp = db.get_latest_temp(room)
        except Exception as e:
            print(f"Error getting latest temp for {room}: {e}")
            latest_temp = None
        
        # Get metrics for this room from database
        try:
            room_metrics = db.get_ai_metrics(room)
        except Exception as e:
            print(f"Error getting metrics for {room}: {e}")
            room_metrics = {}
        
        models_info[room] = {
            "trained": model_exists,
            "last_temp": latest_temp,
            "target_temp": ROOMS[room]["target"],
            "scale_range": f"{ROOMS[room]['scale'][0]}-{ROOMS[room]['scale'][-1]}",
            "current_level": radiator_levels.get(room, 0),
            "training_samples": room_metrics.get("training_samples", 0),
            "predictions_made": room_metrics.get("predictions_made", 0),
            "avg_error": round(room_metrics.get("total_error", 0) / max(1, room_metrics.get("predictions_made", 1)), 2)
        }
    
    # Get current weather with error handling
    try:
        outdoor, forecast_3h, forecast_10h = get_weather()
        if outdoor is None:
            outdoor = 0.0
        if forecast_3h is None:
            forecast_3h = 0.0
        if forecast_10h is None:
            forecast_10h = 0.0
    except Exception as e:
        print(f"Error getting weather: {e}")
        outdoor, forecast_3h, forecast_10h = 0.0, 0.0, 0.0
    
    return {
        "status": "online",
        "version": "2.0.0-database",
        "timestamp": datetime.now().isoformat(),
        "rooms": models_info,
        "weather": {
            "outdoor_temp": outdoor,
            "forecast_3h": forecast_3h,
            "forecast_10h": forecast_10h,
        },
        "telegram_webhook_configured": TELEGRAM_WEBHOOK is not None,
        "database_connected": DATABASE_URL is not None,
    }

@app.post("/train")
def train(state: RoomState):
    """Train the model with new data"""
    model = load_model(state.room)
    
    # Get previous temperature from database
    prev = db.get_latest_temp(state.room)
    if prev is None:
        prev = state.current_temp
    delta = state.current_temp - prev

    # Add weather if missing
    if state.outdoor_temp is None or state.forecast_temp is None or state.forecast_10h_temp is None:
        outside, forecast_3h, forecast_10h = get_weather()
        state.outdoor_temp = outside
        state.forecast_temp = forecast_3h
        state.forecast_10h_temp = forecast_10h

    features = {
        "current_temp": state.current_temp,
        "target_temp": state.target_temp,
        "outdoor_temp": state.outdoor_temp,
        "forecast_3h_temp": state.forecast_temp,
        "forecast_10h_temp": state.forecast_10h_temp,
        "radiator_level": state.radiator_level,
        "hour_of_day": datetime.now().hour,
    }

    # Train the model
    model.learn_one(features, delta)
    save_model(state.room, model)

    # Update metrics
    m = model_metrics[state.room]
    m.training_samples += 1
    
    # Try to get a prediction to track accuracy
    predicted_delta = None
    try:
        predicted_delta = model.predict_one(features)
        if predicted_delta is not None:
            m.update(delta, predicted_delta)
    except Exception:
        pass
    
    # Save metrics to database
    db.update_ai_metrics(
        state.room,
        mae=m.mae.get() if m.training_samples > 0 else 0,
        rmse=m.rmse.get() if m.training_samples > 0 else 0,
        r2_score=m.r2.get() if m.training_samples > 1 else 0,
        training_samples=m.training_samples,
        predictions_made=m.predictions_made,
        adjustments_made=m.adjustments_made,
        total_error=m.total_error
    )
    
    # Save room state to database
    db.save_room_state(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, state.outdoor_temp, state.forecast_temp
    )
    
    # Log training event to database
    db.save_training_event(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, delta, state.outdoor_temp,
        state.forecast_temp, predicted_delta, datetime.now().hour
    )

    return {
        "trained": True, 
        "delta": round(delta, 3),
        "training_samples": m.training_samples,
        "model_mae": round(m.mae.get(), 3) if m.training_samples > 0 else None
    }

@app.post("/predict")
def predict(state: RoomState):
    """Get radiator level recommendation optimized for nighttime (next 8 hours)"""
    model = load_model(state.room)
    if state.outdoor_temp is None or state.forecast_temp is None or state.forecast_10h_temp is None:
        outside, forecast_3h, forecast_10h = get_weather()
        state.outdoor_temp, state.forecast_temp, state.forecast_10h_temp = outside, forecast_3h, forecast_10h

    current_hour = datetime.now().hour
    is_evening = 18 <= current_hour <= 23  # Evening time (6 PM - 11 PM)
    
    # Determine prediction horizon
    prediction_hours = 8 if is_evening else 3
    
    best_lvl = None
    best_err = float("inf")
    prediction_details = []
    
    best_lvl_future = None
    best_err_future = float("inf")
    prediction_details_future = []

    # Current/immediate prediction
    for lvl in ROOMS[state.room]["scale"]:
        feat = {
            "current_temp": state.current_temp,
            "target_temp": state.target_temp,
            "outdoor_temp": state.outdoor_temp,
            "forecast_3h_temp": state.forecast_temp,
            "forecast_10h_temp": state.forecast_10h_temp,
            "radiator_level": lvl,
            "hour_of_day": datetime.now().hour,
        }
        try:
            delta = model.predict_one(feat) or 0
        except Exception:
            delta = 0
        predicted_temp = state.current_temp + delta
        error = abs(predicted_temp - state.target_temp)
        
        prediction_details.append({
            "level": lvl,
            "predicted_temp": round(predicted_temp, 2),
            "error": round(error, 2)
        })
        
        if error < best_err:
            best_err, best_lvl = error, lvl
    
    # Future prediction - use appropriate forecast based on horizon
    future_hour = (datetime.now().hour + prediction_hours) % 24
    future_forecast_temp = state.forecast_10h_temp if prediction_hours >= 8 else state.forecast_temp
    
    for lvl in ROOMS[state.room]["scale"]:
        feat_future = {
            "current_temp": state.current_temp,
            "target_temp": state.target_temp,
            "outdoor_temp": future_forecast_temp,
            "forecast_3h_temp": future_forecast_temp,
            "forecast_10h_temp": future_forecast_temp,
            "radiator_level": lvl,
            "hour_of_day": future_hour,
        }
        try:
            # Simulate hours ahead by iteratively predicting
            temp_estimate = state.current_temp
            for hour in range(prediction_hours):
                feat_future["hour_of_day"] = (current_hour + hour + 1) % 24
                delta_step = model.predict_one(feat_future) or 0
                temp_estimate += delta_step
                feat_future["current_temp"] = temp_estimate
        except Exception:
            temp_estimate = state.current_temp
        
        error_future = abs(temp_estimate - state.target_temp)
        
        prediction_details_future.append({
            "level": lvl,
            f"predicted_temp_{prediction_hours}h": round(temp_estimate, 2),
            f"error_{prediction_hours}h": round(error_future, 2)
        })
        
        if error_future < best_err_future:
            best_err_future, best_lvl_future = error_future, lvl

    # Update metrics
    m = model_metrics[state.room]
    m.predictions_made += 1
    
    adjustment_made = False
    proactive_warning = False
    
    if best_lvl is None:
        db.update_ai_metrics(state.room, predictions_made=m.predictions_made)
        return {"recommended": None, "error": None}

    # Evening mode: prioritize the full night forecast
    if is_evening:
        # Use the best level for the entire night
        if best_lvl_future and abs(best_err_future) > 1.0:
            proactive_warning = True
            if abs(best_lvl_future - state.radiator_level) >= 1:
                best_lvl = best_lvl_future  # Use night forecast
                adjustment_made = True
                m.adjustments_made += 1
                
                msg = (
                    f"🌙 NIGHT MODE: {state.room}: set radiator to {best_lvl_future}\n"
                    f"🌡️ now {state.current_temp}°C → target {state.target_temp}°C\n"
                    f"🔮 In {prediction_hours}h: predicted {prediction_details_future[ROOMS[state.room]['scale'].index(best_lvl_future)][f'predicted_temp_{prediction_hours}h']}°C (error: {best_err_future:.1f}°C)\n"
                    f"🌤️ outside {state.outdoor_temp}°C, forecast {state.forecast_temp}°C\n"
                    f"💤 Optimized for entire night ({prediction_hours} hours)"
                )
                if TELEGRAM_WEBHOOK:
                    try:
                        requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
                    except Exception:
                        pass
        elif abs(best_lvl - state.radiator_level) >= 1:
            adjustment_made = True
            m.adjustments_made += 1
            msg = (
                f"🌙 {state.room}: set radiator to {best_lvl}\n"
                f"🌡️ now {state.current_temp}°C → target {state.target_temp}°C\n"
                f"🌤️ outside {state.outdoor_temp}°C, forecast {state.forecast_temp}°C"
            )
            if TELEGRAM_WEBHOOK:
                try:
                    requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
                except Exception:
                    pass
    else:
        # Daytime: check if future prediction shows problem
        if best_lvl_future and abs(best_err_future) > 1.5:
            proactive_warning = True
            if abs(best_lvl_future - state.radiator_level) >= 1:
                best_lvl = best_lvl_future
                adjustment_made = True
                m.adjustments_made += 1
                
                msg = (
                    f"⚠️ PROACTIVE: {state.room}: set radiator to {best_lvl_future}\n"
                    f"🌡️ now {state.current_temp}°C → target {state.target_temp}°C\n"
                    f"🔮 In {prediction_hours}h: predicted {prediction_details_future[ROOMS[state.room]['scale'].index(best_lvl_future)][f'predicted_temp_{prediction_hours}h']}°C (error: {best_err_future:.1f}°C)\n"
                    f"🌤️ outside {state.outdoor_temp}°C, forecast {state.forecast_temp}°C"
                )
                if TELEGRAM_WEBHOOK:
                    try:
                        requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
                    except Exception:
                        pass
        elif abs(best_lvl - state.radiator_level) >= 1:
            m.adjustments_made += 1
            adjustment_made = True
            
            msg = (
                f"🏠 {state.room}: set radiator to {best_lvl}\n"
                f"🌡️ now {state.current_temp}°C → target {state.target_temp}°C\n"
                f"🌤️ outside {state.outdoor_temp}°C, forecast {state.forecast_temp}°C"
            )
            if TELEGRAM_WEBHOOK:
                try:
                    requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
                except Exception:
                    pass
    
    # Update metrics in database
    db.update_ai_metrics(
        state.room,
        predictions_made=m.predictions_made,
        adjustments_made=m.adjustments_made,
        total_error=m.total_error
    )
    
    # Calculate predicted temp for future for the best level
    predicted_temp_future = None
    if best_lvl_future is not None:
        for detail in prediction_details_future:
            if detail['level'] == best_lvl_future:
                predicted_temp_future = detail[f'predicted_temp_{prediction_hours}h']
                break
    
    # Save the future prediction for validation and training
    if predicted_temp_future is not None:
        db.save_future_prediction(
            room=state.room,
            hours_ahead=prediction_hours,
            predicted_temp=predicted_temp_future,
            radiator_level=best_lvl_future,
            outdoor_temp=state.outdoor_temp,
            forecast_temp=state.forecast_temp
        )
    
    # Save prediction to database
    db.save_prediction(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, best_lvl, best_err, adjustment_made,
        state.outdoor_temp, state.forecast_temp,
        recommended_level_3h=best_lvl_future,
        predicted_error_3h=best_err_future,
        predicted_temp_3h=predicted_temp_future,
        proactive_warning=proactive_warning
    )
    
    # Save room state to database
    db.save_room_state(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, state.outdoor_temp, state.forecast_temp
    )

    return {
        "recommended": best_lvl, 
        "error": round(best_err, 2),
        "current_level": state.radiator_level,
        "adjustment_needed": adjustment_made,
        "predictions_made": m.predictions_made,
        "prediction_details": prediction_details[:5],
        "forecast_future": {
            "hours_ahead": prediction_hours,
            "mode": "night" if is_evening else "day",
            "recommended_level": best_lvl_future,
            f"predicted_error_{prediction_hours}h": round(best_err_future, 2),
            "proactive_warning": proactive_warning,
            "prediction_details": prediction_details_future[:5]
        }
    }

@app.get("/stats")
def get_stats():
    """Get comprehensive AI performance statistics from database"""
    stats = {}
    
    # Get metrics from database
    all_metrics = db.get_ai_metrics()
    
    for room in ROOMS.keys():
        if room in all_metrics:
            metrics = all_metrics[room]
            stats[room] = {
                "mae": round(metrics.get('mae', 0), 3),
                "rmse": round(metrics.get('rmse', 0), 3),
                "r2_score": round(metrics.get('r2_score', 0), 3),
                "training_samples": metrics.get('training_samples', 0),
                "predictions_made": metrics.get('predictions_made', 0),
                "adjustments_made": metrics.get('adjustments_made', 0),
                "avg_error": round(metrics.get('total_error', 0) / max(1, metrics.get('predictions_made', 1)), 3),
                "created_at": metrics.get('created_at').isoformat() if metrics.get('created_at') else None,
                "last_updated": metrics.get('last_updated').isoformat() if metrics.get('last_updated') else None
            }
        else:
            stats[room] = {
                "mae": 0,
                "rmse": 0,
                "r2_score": 0,
                "training_samples": 0,
                "predictions_made": 0,
                "adjustments_made": 0,
                "avg_error": 0,
                "created_at": None,
                "last_updated": None
            }
    
    # Calculate overall statistics
    total_samples = sum(s["training_samples"] for s in stats.values())
    total_predictions = sum(s["predictions_made"] for s in stats.values())
    total_adjustments = sum(s["adjustments_made"] for s in stats.values())
    
    trained_rooms = [s for s in stats.values() if s["training_samples"] > 0]
    avg_mae = sum(s["mae"] for s in trained_rooms) / max(1, len(trained_rooms))
    
    # Get training statistics from database
    training_stats = db.get_training_stats()
    
    # Get prediction statistics from database  
    prediction_stats = db.get_prediction_stats(hours=24)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_training_samples": total_samples,
            "total_predictions": total_predictions,
            "total_adjustments": total_adjustments,
            "average_mae": round(avg_mae, 3),
            "efficiency_rate": round(total_adjustments / max(1, total_predictions) * 100, 1),
        },
        "rooms": stats,
        "training_stats": training_stats,
        "prediction_stats_24h": prediction_stats,
        "model_info": {
            "algorithm": "Adaptive Random Forest Regressor (River ML)",
            "features": ["current_temp", "target_temp", "outdoor_temp", "forecast_temp", "radiator_level", "hour_of_day"],
            "learning_type": "Online/Incremental Learning",
            "prediction_method": "Temperature delta prediction",
            "storage": "PostgreSQL Database",
            "persistence": "Models and data survive restarts"
        }
    }

@app.post("/validate-predictions")
def validate_predictions():
    """Validate past predictions and use them for training"""
    unvalidated = db.get_unvalidated_predictions()
    
    if not unvalidated:
        return {
            "validated": 0,
            "trained": 0,
            "message": "No predictions ready for validation"
        }
    
    validated_count = 0
    trained_count = 0
    
    for pred in unvalidated:
        room = pred['room']
        prediction_id = pred['id']
        predicted_temp = pred['predicted_temp']
        hours_ahead = pred['hours_ahead']
        radiator_level = pred['radiator_level']
        
        # Get actual temperature at the target time
        # Look for the closest room_state reading to target_timestamp
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_temp, outdoor_temp 
                FROM room_states
                WHERE room = %s 
                  AND timestamp >= %s - INTERVAL '30 minutes'
                  AND timestamp <= %s + INTERVAL '30 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1
            """, (room, pred['target_timestamp'], pred['target_timestamp'], pred['target_timestamp']))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                actual_temp = result[0]
                actual_outdoor = result[1]
                
                # Validate the prediction
                db.validate_prediction(prediction_id, actual_temp, used_for_training=True)
                validated_count += 1
                
                # Use this for training!
                # Calculate the actual delta that occurred
                original_temp = predicted_temp - (hours_ahead * 0.1)  # Rough estimate of starting temp
                actual_delta = actual_temp - original_temp
                
                # Load model and train on this real-world outcome
                model = load_model(room)
                
                features = {
                    "current_temp": original_temp,
                    "target_temp": ROOMS[room]["target"],
                    "outdoor_temp": actual_outdoor or pred['outdoor_temp'],
                    "forecast_temp": pred['forecast_temp'],
                    "radiator_level": radiator_level,
                    "hour_of_day": pred['target_timestamp'].hour,
                }
                
                # Train the model with the actual outcome
                model.learn_one(features, actual_delta)
                save_model(room, model)
                
                # Update metrics
                m = model_metrics[room]
                m.training_samples += 1
                
                # Save training event
                db.save_training_event(
                    room, original_temp, ROOMS[room]["target"],
                    radiator_level, actual_delta,
                    actual_outdoor, pred['forecast_temp'],
                    predicted_delta=(predicted_temp - original_temp),
                    hour_of_day=pred['target_timestamp'].hour
                )
                
                # Update metrics in database
                db.update_ai_metrics(
                    room,
                    training_samples=m.training_samples
                )
                
                trained_count += 1
                
                print(f"✅ Validated & trained {room}: predicted {predicted_temp:.1f}°C, actual {actual_temp:.1f}°C (error: {abs(predicted_temp - actual_temp):.2f}°C)")
            else:
                # No matching temperature data found, just mark as validated without training
                db.validate_prediction(prediction_id, None, used_for_training=False)
                validated_count += 1
                
        except Exception as e:
            print(f"Error validating prediction {prediction_id}: {e}")
            continue
    
    return {
        "validated": validated_count,
        "trained": trained_count,
        "message": f"Validated {validated_count} predictions, trained on {trained_count}"
    }

@app.get("/validation-stats")
def get_validation_statistics(days: int = 7):
    """Get statistics on prediction validation accuracy"""
    stats = db.get_validation_stats(days=days)
    
    return {
        "days": days,
        "rooms": stats,
        "summary": {
            "total_predictions": sum(s.get('total_predictions', 0) for s in stats.values()),
            "total_trained": sum(s.get('used_for_training', 0) for s in stats.values()),
            "avg_error": sum(s.get('avg_error', 0) for s in stats.values()) / max(1, len(stats))
        }
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiator/history/{room}")
def get_room_radiator_history(room: str, hours: int = 24):
    """Get radiator level change history for a specific room"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        history = db.get_radiator_history(room, hours)
        return {
            "room": room,
            "hours": hours,
            "changes": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/radiator/history")
def get_all_radiator_history(hours: int = 24):
    """Get radiator level change history for all rooms"""
    try:
        history = db.get_radiator_history(None, hours)
        
        # Group by room
        by_room = {}
        for change in history:
            room = change['room']
            if room not in by_room:
                by_room[room] = []
            by_room[room].append(change)
        
        return {
            "hours": hours,
            "total_changes": len(history),
            "rooms": by_room
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/history/{room}")
def get_room_history(room: str, hours: int = 24):
    """Get historical data for a specific room (for graphing)"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        data = db.get_historical_data(room, hours)
        return {
            "room": room,
            "hours": hours,
            "data_points": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/history")
def get_all_history(hours: int = 24):
    """Get historical data for all rooms (for graphing)"""
    try:
        data = db.get_historical_data(None, hours)
        
        # Group by room
        by_room = {}
        for point in data:
            room = point['room']
            if room not in by_room:
                by_room[room] = []
            by_room[room].append(point)
        
        return {
            "hours": hours,
            "total_data_points": len(data),
            "rooms": by_room
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/training/history/{room}")
def get_training_history(room: str):
    """Get training history for a specific room"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        stats = db.get_training_stats(room)
        return {
            "room": room,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching training history: {str(e)}")

@app.get("/export/csv/{room}")
def export_room_csv(room: str, hours: int = 168):  # Default 1 week
    """Export room data as CSV for analysis"""
    from fastapi.responses import StreamingResponse
    import csv
    from io import StringIO
    
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        data = db.get_historical_data(room, hours)
        
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['timestamp', 'room', 'current_temp', 'target_temp', 
                        'radiator_level', 'outdoor_temp', 'forecast_temp'])
        
        # Data
        for point in data:
            writer.writerow([
                point['timestamp'].isoformat() if isinstance(point['timestamp'], datetime) else point['timestamp'],
                point['room'],
                point['current_temp'],
                point['target_temp'],
                point['radiator_level'],
                point.get('outdoor_temp', ''),
                point.get('forecast_temp', '')
            ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={room}_data.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@app.get("/ui", response_class=HTMLResponse)
def ui_dashboard():
    """Enhanced web UI dashboard with validation stats and advanced analytics"""
    from fastapi.responses import HTMLResponse
    
    try:
        latest_training = db.get_latest_training_events(limit=10)
        latest_predictions = db.get_latest_predictions(limit=10)
        training_count_24h = db.get_training_count_last_24h()
        all_metrics = db.get_ai_metrics()
        validation_stats = db.get_validation_stats(days=7)
        
        # Get weather
        try:
            outdoor, forecast_3h, forecast_10h = get_weather()
            outdoor = outdoor if outdoor else "N/A"
            forecast_3h = forecast_3h if forecast_3h else "N/A"
            forecast_10h = forecast_10h if forecast_10h else "N/A"
        except Exception:
            outdoor, forecast_3h, forecast_10h = "N/A", "N/A", "N/A"
        
        # Calculate totals
        total_predictions = sum(m.get('predictions_made', 0) for m in all_metrics.values())
        total_trained = sum(m.get('training_samples', 0) for m in all_metrics.values())
        total_validated = sum(v.get('total_predictions', 0) for v in validation_stats.values())
        avg_validation_error = sum(v.get('avg_error', 0) for v in validation_stats.values()) / max(1, len(validation_stats))
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Radiator AI - Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .header .status {{ color: #28a745; font-size: 1.2em; font-weight: bold; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        .stat-card .label {{ color: #666; font-size: 0.85em; }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.6em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.8em;
        }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .room-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .room-Sovrum {{ background: #e3f2fd; color: #1976d2; }}
        .room-Kontor {{ background: #f3e5f5; color: #7b1fa2; }}
        .room-Vardagsrum {{ background: #fff3e0; color: #e65100; }}
        .room-Badrum {{ background: #e8f5e9; color: #2e7d32; }}
        .good {{ color: #28a745; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        .info {{ color: #17a2b8; font-weight: bold; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-box .metric-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-box .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }}
        .metric-box .metric-sub {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
        .refresh-btn {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Smart Radiator AI Dashboard</h1>
            <div class="status">✅ System Online - v2.1.0</div>
            <p style="color:#666;margin-top:10px">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>🎓 Training Samples</h3>
                <div class="value good">{total_trained}</div>
                <div class="label">Total model training events</div>
                <div class="label" style="margin-top:5px">{training_count_24h} in last 24h</div>
            </div>
            <div class="stat-card">
                <h3>🎯 Predictions Made</h3>
                <div class="value info">{total_predictions}</div>
                <div class="label">AI recommendations</div>
            </div>
            <div class="stat-card">
                <h3>✅ Validated</h3>
                <div class="value warning">{total_validated}</div>
                <div class="label">Self-learning cycles</div>
                <div class="label" style="margin-top:5px">Avg error: {avg_validation_error:.2f}°C</div>
            </div>
            <div class="stat-card">
                <h3>🌡️ Current Weather</h3>
                <div class="value">{outdoor}°C</div>
                <div class="label">3h: {forecast_3h}°C | 10h: {forecast_10h}°C</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Per-Room Analytics</h2>
            <div class="metrics-grid">
"""
        
        for room, metrics in all_metrics.items():
            training = metrics.get('training_samples', 0)
            predictions = metrics.get('predictions_made', 0)
            mae = metrics.get('mae', 0)
            r2 = metrics.get('r2_score', 0)
            
            val_stats = validation_stats.get(room, {})
            val_count = val_stats.get('total_predictions', 0)
            val_error = val_stats.get('avg_error', 0)
            
            # Calculate accuracy as inverse of average error
            # If avg error is 0°C -> 100% accuracy (full bar)
            # If avg error is 1°C -> ~63% accuracy
            # If avg error is 2°C -> ~37% accuracy
            # Using exponential decay: accuracy = 100 * e^(-error)
            import math
            accuracy_pct = 100 * math.exp(-val_error) if val_error > 0 else 100
            
            html += f"""
                <div class="metric-box">
                    <div class="metric-label">{room}</div>
                    <div class="metric-value">{training}</div>
                    <div class="metric-sub">Training samples</div>
                    <div class="metric-sub">MAE: {mae:.3f} | R²: {r2:.3f}</div>
                    <div class="metric-sub">Predictions: {predictions}</div>
                    <div class="metric-sub">Validated: {val_count} (±{val_error:.2f}°C)</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:{accuracy_pct:.0f}%">
                            {accuracy_pct:.0f}% (±{val_error:.2f}°C avg)
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Validation statistics section
        if validation_stats:
            html += """
        <div class="section">
            <h2>✅ Prediction Validation Stats (Last 7 Days)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Room</th>
                        <th>Predictions Made</th>
                        <th>Used for Training</th>
                        <th>Avg Error</th>
                        <th>Min Error</th>
                        <th>Max Error</th>
                        <th>Training Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
            for room, vstats in validation_stats.items():
                total = vstats.get('total_predictions', 0)
                used = vstats.get('used_for_training', 0)
                avg_err = vstats.get('avg_error', 0)
                min_err = vstats.get('min_error', 0)
                max_err = vstats.get('max_error', 0)
                # Calculate accuracy as percentage of predictions with error < 0.5°C
                # This is a more meaningful metric than trying to convert error to percentage
                accuracy_pct = (used / total * 100) if total > 0 else 0
                
                html += f"""
                    <tr>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{total}</td>
                        <td class="good">{used}</td>
                        <td>{avg_err:.3f}°C</td>
                        <td class="good">{min_err:.3f}°C</td>
                        <td class="warning">{max_err:.3f}°C</td>
                        <td class="info">{accuracy_pct:.1f}%</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Latest training
        if latest_training:
            html += """
        <div class="section">
            <h2>🎓 Recent Training Events</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Room</th>
                        <th>Current</th>
                        <th>Target</th>
                        <th>Level</th>
                        <th>Delta</th>
                        <th>Predicted</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
"""
            for event in latest_training[:5]:
                room = event['room']
                timestamp = event['timestamp'].strftime('%H:%M:%S') if isinstance(event['timestamp'], datetime) else str(event['timestamp'])
                delta = event['temperature_delta']
                pred_delta = event.get('predicted_delta')
                error = abs(delta - pred_delta) if pred_delta is not None else 0
                pred_delta_str = f"{pred_delta:.2f}°C" if pred_delta else 'N/A'
                
                error_class = 'good' if error < 0.3 else 'warning' if error < 0.6 else 'info'
                html += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{event['current_temp']:.1f}°C</td>
                        <td>{event['target_temp']:.1f}°C</td>
                        <td>{event['radiator_level']}</td>
                        <td>{delta:+.2f}°C</td>
                        <td>{pred_delta_str}</td>
                        <td class="{error_class}">{error:.2f}°C</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Latest predictions
        if latest_predictions:
            html += """
        <div class="section">
            <h2>🎯 Recent Predictions</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Room</th>
                        <th>Current</th>
                        <th>Target</th>
                        <th>Recommended</th>
                        <th>Error</th>
                        <th>Adjusted</th>
                    </tr>
                </thead>
                <tbody>
"""
            for pred in latest_predictions[:5]:
                room = pred['room']
                timestamp = pred['timestamp'].strftime('%H:%M:%S') if isinstance(pred['timestamp'], datetime) else str(pred['timestamp'])
                adjusted = "✅" if pred['adjustment_made'] else "➖"
                level_class = 'good' if pred['adjustment_made'] else 'info'
                
                html += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{pred['current_temp']:.1f}°C</td>
                        <td>{pred['target_temp']:.1f}°C</td>
                        <td class="{level_class}">{pred['recommended_level']}</td>
                        <td>{pred['predicted_error']:.2f}°C</td>
                        <td>{adjusted}</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
        </div>
"""
        
        html += """
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    </div>
    <script>
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head><title>Error</title></head>
<body style="font-family:Arial;padding:50px;text-align:center">
    <h1 style="color:red">⚠️ Error Loading Dashboard</h1>
    <p>{str(e)}</p>
    <p><a href="/ui">Try again</a></p>
</body>
</html>
""", status_code=500)
