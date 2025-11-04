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

@dp.message_handler(commands=['start'])
async def start(msg: types.Message):
    """Welcome message with menu button"""
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(types.KeyboardButton("🔧 Set Radiator"))
    keyboard.add(types.KeyboardButton("📊 Status"))
    
    await msg.reply(
        "🏠 *Smart Radiator Assistant*\n\n"
        "Control your radiators with AI-powered temperature management.\n\n"
        "Use the menu below or type /set",
        reply_markup=keyboard,
        parse_mode="Markdown"
    )

@dp.message_handler(lambda msg: msg.text == "🔧 Set Radiator")
@dp.message_handler(commands=['set'])
async def set_radiator(msg: types.Message):
    kb = types.InlineKeyboardMarkup(row_width=2)
    buttons = [
        types.InlineKeyboardButton(text=f"🌡️ {room}", callback_data=f"room:{room}")
        for room in ROOMS
    ]
    kb.add(*buttons)
    await msg.reply("*Select room:*", reply_markup=kb, parse_mode="Markdown")

@dp.message_handler(lambda msg: msg.text == "📊 Status")
async def status(msg: types.Message):
    """Show current radiator levels"""
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT room, level, updated_at FROM radiators ORDER BY room")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            await msg.reply("No radiator levels set yet. Use 🔧 Set Radiator to configure.")
            return
        
        status_text = "📊 *Current Radiator Levels:*\n\n"
        for room, level, updated in rows:
            target = ROOMS.get(room, {}).get("target", "?")
            status_text += f"🌡️ *{room}*: Level {level} (Target {target}°C)\n"
            status_text += f"   _Updated: {updated.strftime('%H:%M %d/%m')}_\n\n"
        
        await msg.reply(status_text, parse_mode="Markdown")
    except Exception as e:
        await msg.reply(f"⚠️ Error fetching status: {str(e)}")

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("room:"))
async def choose_level(callback: types.CallbackQuery):
    room = callback.data.split(":", 1)[1]
    scale = ROOMS[room]["scale"]
    target = ROOMS[room]["target"]
    
    # Get current level from database
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT level FROM radiators WHERE room = %s", (room,))
        result = cur.fetchone()
        current = result[0] if result else 0
        cur.close()
        conn.close()
    except:
        current = 0
    
    # Create keyboard with 3 columns
    kb = types.InlineKeyboardMarkup(row_width=3)
    buttons = []
    for lvl in scale:
        # Highlight current level
        text = f"✓ {lvl}" if lvl == current else str(lvl)
        buttons.append(types.InlineKeyboardButton(text, callback_data=f"set:{room}:{lvl}"))
    kb.add(*buttons)
    
    await callback.message.edit_text(
        f"🌡️ *{room}*\n"
        f"Target: {target}°C\n"
        f"Current: {current}\n\n"
        f"Select level:",
        reply_markup=kb,
        parse_mode="Markdown"
    )

@dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:"))
async def confirm(callback: types.CallbackQuery):
    _, room, lvl = callback.data.split(":")
    lvl = float(lvl)
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO radiators (room, level) VALUES (%s, %s) ON CONFLICT (room) DO UPDATE SET level = %s, updated_at = CURRENT_TIMESTAMP",
            (room, lvl, lvl)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Saved {room} = {lvl} to database")
    except Exception as e:
        print(f"❌ Database error: {e}")
        await callback.message.edit_text(f"⚠️ Database error: {str(e)}")
        return
    
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
    except Exception as e:
        print(f"⚠️ AI training request failed: {e}")

if __name__ == "__main__":
    executor.start_polling(dp)
