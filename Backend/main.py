import os
import re
import random
import httpx
import json
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId
from google import genai
from google.genai import types

# --- Setup ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME", "todo_ai")
MODEL = "gemini-2.0-flash"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini Tool Definitions ---
tools = [
    {
        "name": "addTask",
        "description": "Add a new task to the user's to-do list.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The title of the task. e.g., 'Buy milk'"},
            },
            "required": ["title"]
        }
    },
    {
        "name": "deleteTask",
        "description": "Delete a task from the user's list based on its title or keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Keywords to find the task to delete. e.g., 'milk'"},
            },
            "required": ["title"]
        }
    },
    {
        "name": "setReminder",
        "description": "Set a reminder for the user. Always requires a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "The subject of the reminder. e.g., 'Call mom'"},
                "delay_minutes": {"type": "integer", "description": "The number of minutes from now to set the reminder."},
            },
            "required": ["title", "delay_minutes"]
        }
    }
]

# --- Helper Functions ---
def task_helper(task) -> dict:
    return {
        "id": str(task["_id"]),
        "title": task["title"],
        "due": task.get("due"),
        "completed": task["completed"]
    }

async def gemini_tool_call(prompt: str):
    if not API_KEY:
        return json.dumps({"name": "fallback", "args": {"reply": "Gemini API key is not set."}})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"functionDeclarations": tools}]
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response_json = response.json()

            if response.status_code == 200 and "candidates" in response_json:
                part = response_json["candidates"][0]["content"]["parts"][0]
                if "functionCall" in part:
                    return json.dumps(part["functionCall"])
                elif "text" in part:
                    return json.dumps({"name": "fallback", "args": {"reply": part["text"]}})
            
            print("--- GEMINI API ERROR ---")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            print("--- END OF ERROR ---")
            return json.dumps({"name": "fallback", "args": {"reply": "Sorry, there was an error processing the request."}})

    except Exception as e:
        print(f"An exception occurred: {e}")
        return json.dumps({"name": "fallback", "args": {"reply": "Sorry, an unexpected error occurred."}})

# --- Tool Handler Functions ---
async def handle_add_task(args):
    await db.tasks.insert_one({"title": args["title"], "due": None, "completed": False})
    return f"Task added: \"{args['title']}\""

async def handle_delete_task(args):
    result = await db.tasks.delete_one({"title": {"$regex": args["title"], "$options": "i"}})
    return f"Deleted task matching '{args['title']}'." if result.deleted_count > 0 else f"Couldn't find a task matching '{args['title']}'."

async def handle_set_reminder(args):
    reminder_time = int(datetime.now(timezone.utc).timestamp() * 1000) + int(args["delay_minutes"]) * 60000
    await db.reminders.insert_one({"title": args["title"], "at": reminder_time, "sent": False})
    return f"OK, I'll remind you about \"{args['title']}\" in {args['delay_minutes']} minutes."

tool_handlers = {
    "addTask": handle_add_task,
    "deleteTask": handle_delete_task,
    "setReminder": handle_set_reminder,
}

# --- API Endpoints ---
@app.get("/chat")
async def chat(msg: str = Query(..., alias="msg")):
    tool_response = json.loads(await gemini_tool_call(msg))
    tool_name = tool_response.get("name")
    
    if handler := tool_handlers.get(tool_name):
        reply = await handler(tool_response.get("args", {}))
    else:
        reply = tool_response.get("args", {}).get("reply", "I'm not sure how to help with that.")
        
    return {"reply": reply}

@app.get("/api/tasks")
async def get_tasks():
    tasks = [task_helper(t) async for t in db.tasks.find()]
    return {"tasks": tasks}

@app.get("/api/due-reminders")
async def get_due_reminders():
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    due_reminders = [r async for r in db.reminders.find({"at": {"$lte": now}, "sent": False})]
    if due_reminders:
        await db.reminders.update_many({"_id": {"$in": [r["_id"] for r in due_reminders]}}, {"$set": {"sent": True}})
    return {"reminders": due_reminders}




# --- Serve Frontend ---
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
