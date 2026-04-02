from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

# Use the environment variable or your direct string
MONGO_URL = os.getenv("MONGO_URL", "MONGO_URL")

client = AsyncIOMotorClient(MONGO_URL)
database = client.medipredict

# Define ALL collections here so they can be imported elsewhere
users_collection = database.get_collection("users")
assessments_collection = database.get_collection("assessments")
health_plans_collection = database.get_collection("health_plans")
lab_reports_collection = database.get_collection("lab_reports") # This was missing!