<div align="center">

# 🏠 Radiator AI Assistant

**Intelligent home heating automation with AI-powered predictions**

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/u/namelesshampus)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Automate your radiator settings with machine learning and smart home integration*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-endpoints)

</div>

---

## 📋 Overview

Smart Radiator Assistant is a complete home automation solution that uses **online machine learning** to optimize your radiator settings. It learns from your preferences and environmental data to maintain perfect room temperature while minimizing energy consumption.

### �️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Home Assistant │      │      n8n         │      │   Telegram Bot  │
│   (Sensors)     │────▶│   (Automation)    │◀──▶│  (Manual Ctrl)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   AI Service     │
                         │   FastAPI + ML   │
                         └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   PostgreSQL     │
                         │   (Model State)  │
                         └──────────────────┘
```

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered
- **Online Machine Learning** using River (ARFRegressor)
- **Dual weather forecasts** (3h + 10h ahead) for better predictions
- **Night mode optimization** - predicts full 8-hour sleep period
- **Self-training predictions** - validates and learns from own forecasts
- Adapts to your heating preferences in real-time

</td>
<td width="50%">

### 🔧 Easy Integration
- **Docker-based deployment** - up and running in minutes
- **Telegram bot** for manual control
- **n8n workflows** for Home Assistant integration
- **RESTful API** with interactive docs
- **Web dashboard** - visual monitoring and analytics

</td>
</tr>
<tr>
<td width="50%">

### 📊 Advanced Analytics
- Real-time AI statistics (MAE, RMSE, R²)
- **Prediction validation tracking** - measures accuracy over time
- Training history and prediction logs
- **Database-backed persistence** - survives restarts
- Export data to CSV for analysis

</td>
<td width="50%">

### 🚀 Production Ready
- Pre-built Docker images on Docker Hub
- Automated build & deployment scripts
- PostgreSQL for data persistence
- **Background validation** - auto-trains every hour
- Scalable microservice architecture

</td>
</tr>
</table>

### 🌙 Night Mode
In the evening (6 PM - 11 PM), the AI automatically switches to **night mode**:
- Predicts temperature for the **next 8 hours** (full sleep period)
- Uses 10-hour weather forecast for overnight planning
- Prevents waking up cold at 3 AM
- Optimizes radiator settings before bedtime

### 🎯 Self-Improving AI
The system learns from its own predictions:
1. Makes a prediction (e.g., "temperature will be 19.5°C in 3 hours")
2. Stores the prediction with timestamp
3. After 3 hours, compares actual temperature to prediction
4. Trains the model on the difference
5. Gets better over time!

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- PostgreSQL database
- Telegram Bot Token ([get one from @BotFather](https://t.me/botfather))
- (Optional) n8n instance for automation

### 1️⃣ Clone & Configure

```bash
git clone https://github.com/HampusAndersson01/smart-radiator-assistant.git
cd smart-radiator-assistant
cp .env.example .env
```

Edit `.env` and add your credentials:
```bash
BOT_TOKEN=your_telegram_bot_token
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=your_db_host
```

### 2️⃣ Deploy with Docker Compose

```bash
# Pull latest images and start services
docker compose pull
docker compose up -d

# Check status
docker compose ps
docker compose logs -f
```

### 3️⃣ Verify Installation

```bash
# Test AI service
curl http://localhost:8000/docs

# Check AI statistics
curl http://localhost:8000/stats

# Send test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "room": "Living Room",
    "current_temp": 20.5,
    "target_temp": 22.0,
    "radiator_level": 3,
    "timestamp": "now"
  }'
```

## 📦 Services

### AI Service
**Port:** `8000`  
**Image:** `namelesshampus/radiator-ai-service:latest`

FastAPI microservice providing ML-powered radiator predictions and training endpoints.

### Telegram Bot
**Image:** `namelesshampus/radiator-bot:latest`

Interactive Telegram bot for manual radiator control and status queries.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `BOT_TOKEN` | Telegram bot authentication token | ✅ | - |
| `DATABASE_URL` | PostgreSQL connection string | ✅ | - |
| `POSTGRES_USER` | Database username | ❌ | `postgres` |
| `POSTGRES_PASSWORD` | Database password | ✅ | - |
| `POSTGRES_HOST` | Database host | ❌ | `host.docker.internal` |
| `TELEGRAM_WEBHOOK` | Webhook URL for notifications | ❌ | - |
| `ALLOWED_CHAT_IDS` | Comma-separated list of allowed Telegram chat IDs | ❌ | - |

## 🌐 API Endpoints

### Core Endpoints

#### Training
```http
POST /train
```
Train the model with new sensor data.

**Body:**
```json
{
  "room": "Bedroom",
  "current_temp": 21.5,
  "target_temp": 22.0,
  "radiator_level": 4,
  "outdoor_temp": 2.0,
  "forecast_temp": -1.0,
  "forecast_10h_temp": -5.0,
  "timestamp": "2025-11-06T18:00:00"
}
```

#### Prediction
```http
POST /predict
```
Get AI-recommended radiator level (with night mode optimization).

**Response:**
```json
{
  "recommended": 4.5,
  "error": 0.3,
  "adjustment_needed": true,
  "forecast_future": {
    "hours_ahead": 8,
    "mode": "night",
    "recommended_level": 5.0,
    "proactive_warning": true
  }
}
```

### Analytics Endpoints

#### Statistics
```http
GET /stats
```
Get comprehensive AI performance metrics including validation accuracy.

#### Validation Stats
```http
GET /validation-stats?days=7
```
View prediction accuracy over time.

**Response:**
```json
{
  "summary": {
    "total_predictions": 150,
    "total_trained": 142,
    "avg_error": 0.34
  },
  "rooms": {
    "Bedroom": {
      "total_predictions": 50,
      "avg_error": 0.28,
      "used_for_training": 48
    }
  }
}
```

#### Validate Predictions
```http
POST /validate-predictions
```
Manually trigger validation of past predictions (also runs hourly automatically).

### Data Export

#### Historical Data
```http
GET /history/{room}?hours=24
```
Get temperature and radiator level history for graphing.

#### CSV Export
```http
GET /export/csv/{room}?hours=168
```
Download CSV file for analysis in Excel/Python.

### Web Dashboard
```http
GET /ui
```
Visual dashboard showing:
- Real-time training events
- Latest predictions
- Per-room performance metrics
- Validation statistics
- Weather conditions

### Interactive Documentation
Visit `http://localhost:8000/docs` for full Swagger UI documentation.

## 🔄 n8n Integration

### Import Workflow

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `n8n/Smart Radiator Training.json`
4. Update HTTP Request nodes with your AI service URL:
   - If using Docker network: `http://ai_service:8000`
   - If using host: `http://<your-server-ip>:8000`

### Network Setup

**Option A: Shared Docker Network (Recommended)**
```bash
docker network create radiator-network
```
Add to both n8n and radiator stacks' compose files:
```yaml
networks:
  radiator-network:
    external: true
```

**Option B: Host Network**
Use `http://<server-ip>:8000` in n8n HTTP Request nodes.

## 🛠️ Development

### Build Images Locally

```bash
# Build and push to Docker Hub
./build-and-push.sh

# Build specific version
./build-and-push.sh v1.0.0
```

### Manual Build

```bash
# Build AI service
docker build -t namelesshampus/radiator-ai-service:latest ./ai_service

# Build bot
docker build -t namelesshampus/radiator-bot:latest ./bot
```

### Update Deployment

```bash
# Pull latest images and restart
docker compose pull && docker compose up -d
```

## 📊 Monitoring & Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f ai_service
docker compose logs -f bot

# Check service status
docker compose ps

# Restart services
docker compose restart
```

## 🐛 Troubleshooting

<details>
<summary><b>Bot not responding</b></summary>

- Verify `BOT_TOKEN` is correct in `.env`
- Check bot logs: `docker compose logs bot`
- Ensure bot is started in Telegram (send `/start`)
</details>

<details>
<summary><b>AI service unreachable</b></summary>

- Verify service is running: `docker compose ps`
- Check port 8000 is not in use: `netstat -tuln | grep 8000`
- Review logs: `docker compose logs ai_service`
</details>

<details>
<summary><b>Database connection errors</b></summary>

- Verify PostgreSQL is running and accessible
- Check `DATABASE_URL` format: `postgresql://user:password@host:5432/smart_radiator_ai`
- Ensure database `smart_radiator_ai` exists
</details>

<details>
<summary><b>n8n can't reach AI service</b></summary>

- Ensure both are on the same Docker network
- Use service name `ai_service` not `localhost`
- Check firewall rules if using host networking
</details>

## 📚 Documentation

- **[Database Schema](DATABASE_SCHEMA.md)** - Complete database structure
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI (when running)
- **[Web Dashboard](http://localhost:8000/ui)** - Visual monitoring interface
- **[n8n Workflows](n8n/)** - Automation templates

## 🔄 Database Management

### Reset Database (Start Fresh)
If you want to clear all training data and start learning from scratch:

```bash
./reset_database.sh
```

This will:
- ❌ Delete all training history
- ❌ Remove all ML models
- ❌ Clear prediction logs
- ✅ Recreate fresh database schema
- ✅ Restart AI service with new feature set

**Use this when:**
- Updating model features (like adding 10h forecast)
- Switching room configurations
- Testing different learning approaches

### Backup Database
```bash
docker compose exec postgres pg_dump -U postgres radiators > backup.sql
```

### Restore Database
```bash
cat backup.sql | docker compose exec -T postgres psql -U postgres radiators
```

## 🎓 Academic Features

This project includes comprehensive metrics for academic evaluation:

- **Performance Metrics**: MAE, RMSE, R² scoring with per-room granularity
- **Prediction Validation**: Compares forecasts to actual outcomes
- **Training Analytics**: Sample counts, model convergence tracking, validation accuracy
- **Self-Learning Loop**: AI improves by validating its own predictions
- **Database Persistence**: PostgreSQL-backed complete audit trail
- **Export Capabilities**: CSV export for external analysis (Excel, Python, R)
- **Time-Series Data**: Historical tracking for trend analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[River](https://riverml.xyz/)** - Online machine learning framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[aiogram](https://aiogram.dev/)** - Telegram bot framework
- **[n8n](https://n8n.io/)** - Workflow automation

---

<div align="center">

**Made with ❤️ for smart home automation**

[⬆ Back to Top](#-smart-radiator-assistant)

</div>
