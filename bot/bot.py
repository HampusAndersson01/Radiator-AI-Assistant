from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import requests, os
import psycopg2
from psycopg2.extras import RealDictCursor
from config import ROOMS, BOT_TOKEN, AI_URL

if BOT_TOKEN is None:
    raise SystemExit("BOT_TOKEN environment variable not set")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL is None:
    raise SystemExit("DATABASE_URL environment variable not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

# Initialize database
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS radiators(
        room TEXT PRIMARY KEY, 
        level REAL,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
cur.close()

def get_db():
    """Get a new database connection"""
    return psycopg2.connect(DATABASE_URL)

@dp.message_handler(commands=['set'])
async def set_radiator(msg: types.Message):
    kb = types.InlineKeyboardMarkup()
    for room in ROOMS:
        kb.add(types.InlineKeyboardButton(text=room, callback_data=f"room:{room}"))
    await msg.reply("Select room radiator:", reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("room:"))
async def choose_level(callback: types.CallbackQuery):
    room = callback.data.split(":", 1)[1]
    scale = ROOMS[room]["scale"]
    kb = types.InlineKeyboardMarkup()
    for lvl in scale:
        kb.add(types.InlineKeyboardButton(str(lvl), callback_data=f"set:{room}:{lvl}"))
    await callback.message.edit_text(f"🔧 {room}\nTarget {ROOMS[room]['target']}°C\nChoose level:", reply_markup=kb)

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:"))
async def confirm(callback: types.CallbackQuery):
    _, room, lvl = callback.data.split(":")
    lvl = float(lvl)
    
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO radiators (room, level) VALUES (%s, %s) ON CONFLICT (room) DO UPDATE SET level = %s, updated_at = CURRENT_TIMESTAMP",
        (room, lvl, lvl)
    )
    conn.commit()
    cur.close()
    conn.close()
    
    await callback.message.edit_text(f"✅ {room} radiator set to {lvl} (target {ROOMS[room]['target']}°C)")

    # Send a training ping (manual set). Current temp is left 0: n8n / sensors should provide real values
    try:
        requests.post(AI_URL, json={
            "room": room,
            "target_temp": ROOMS[room]["target"],
            "current_temp": 0,
            "radiator_level": lvl,
            "timestamp": "manual"
        }, timeout=5)
    except Exception:
        pass

if __name__ == "__main__":
    executor.start_polling(dp)
