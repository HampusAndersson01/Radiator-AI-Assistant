import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
AI_URL = os.getenv("AI_URL", "http://ai_service:8000/train")

# Parse allowed chat IDs from comma-separated string
ALLOWED_CHAT_IDS_STR = os.getenv("ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS = [int(id.strip()) for id in ALLOWED_CHAT_IDS_STR.split(",") if id.strip()]

ROOMS = {
    "Badrum":     {"scale": [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], "target": 22.5},
    "Sovrum":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
    "Kontor":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
    "Vardagsrum":{"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
}
