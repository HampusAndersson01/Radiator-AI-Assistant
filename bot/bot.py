from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import sqlite3, requests, os
from config import ROOMS, BOT_TOKEN, AI_URL

if BOT_TOKEN is None:
    raise SystemExit("BOT_TOKEN environment variable not set")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

conn = sqlite3.connect("radiators.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS radiators(room TEXT PRIMARY KEY, level INT)")
conn.commit()

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
    lvl = int(lvl)
    cur.execute("INSERT OR REPLACE INTO radiators VALUES(?,?)", (room, lvl))
    conn.commit()
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
