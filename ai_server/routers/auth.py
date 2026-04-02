from fastapi import APIRouter, HTTPException, status
from models.user import UserRegister, UserLogin, UserProfile # Fixed typos
from config.db import users_collection
from passlib.context import CryptContext
from datetime import datetime # Added missing import
import uuid

router = APIRouter()

# Fixed typo: 'scheme' -> 'schemes'
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register", status_code=201)
async def register_user(user: UserRegister):
    # 1. Check if user already exists
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered.")
    
    # 2. Hash password and generate ID
    hashed_password = pwd_context.hash(user.password)
    user_id = str(uuid.uuid4())

    # 3. Create document
    new_user = {
        "user_id": user_id,
        "email": user.email,
        "full_name": user.full_name,
        "password": hashed_password, # Matched with login key check below
        "created_at": datetime.utcnow()
    }

    # Fixed indentation
    await users_collection.insert_one(new_user)

    return {"message": "User created successfully", "user_id": user_id}

@router.post("/login")
async def login_user(user: UserLogin):
    db_user = await users_collection.find_one({"email": user.email})
    
    # Ensure key matches: db_user["password"] vs "hashed_password"
    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "status": "success",
        "user_id": db_user["user_id"],
        "full_name": db_user["full_name"]
    }

@router.get("/profile/{user_id}", response_model=UserProfile)
async def get_profile(user_id: str):
    user = await users_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # FastAPI handles converting the MongoDB dict to the UserProfile model
    return user