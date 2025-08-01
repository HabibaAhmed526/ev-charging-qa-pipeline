#!/usr/bin/env python3
"""
Model Serving API

FastAPI server for serving the fine-tuned EV charging QA model with authentication and monitoring.
"""

import logging
import os
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
import bcrypt
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    import yaml
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# FastAPI app
app = FastAPI(
    title="EV Charging QA API",
    description="API for EV charging infrastructure Q&A using fine-tuned Mistral-7B",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables
model = None
tokenizer = None
device = None

# Request models
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    max_tokens: int = Field(default=150, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

class GenerateResponse(BaseModel):
    response: str
    tokens_generated: int
    generation_time: float
    model_name: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency: float
    error_rate: float
    model_name: str

# Authentication models
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: str
    expires: datetime

# Mock user database (in production, use proper database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8KqKqKq",  # "password"
        "role": "admin"
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config["api"]["auth"]["access_token_expire_minutes"])
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv("API_SECRET_KEY", "default-secret-key"), algorithm=config["api"]["auth"]["algorithm"])
    return encoded_jwt

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.start_time = time.time()
    
    def record_request(self, latency: float, success: bool = True):
        self.request_count += 1
        self.total_latency += latency
        if not success:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0
        
        return {
            "total_requests": self.request_count,
            "avg_latency": avg_latency,
            "error_rate": error_rate,
            "uptime": uptime
        }

monitor = PerformanceMonitor()

# Rate limiting
request_counts = {}
RATE_LIMIT = config["api"]["rate_limiting"]["requests_per_minute"]

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit"""
    current_time = time.time()
    minute_ago = current_time - 60
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests
    request_counts[client_ip] = [t for t in request_counts[client_ip] if t > minute_ago]
    
    # Check if limit exceeded
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return False
    
    # Add current request
    request_counts[client_ip].append(current_time)
    return True

# Model loading
def load_model():
    """Load the fine-tuned model"""
    global model, tokenizer, device
    
    try:
        model_path = Path("models/finetuned")
        if not model_path.exists():
            logger.warning("No fine-tuned model found, using base model")
            model_path = config["environment"]["base_model"]
        
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials, 
            os.getenv("API_SECRET_KEY", "default-secret-key"), 
            algorithms=[config["api"]["auth"]["algorithm"]]
        )
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API endpoints
@app.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    current_user: Optional[str] = Depends(get_current_user)
):
    """Generate text response"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Rate limiting
    client_ip = "127.0.0.1"  # In production, get from request
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    start_time = time.time()
    
    try:
        # Format prompt
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
        
        # Record metrics
        monitor.record_request(generation_time, True)
        
        return GenerateResponse(
            response=response,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            model_name="mistral-7b-finetuned"
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        monitor.record_request(generation_time, False)
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/generate/public", response_model=GenerateResponse)
async def generate_text_public(request: GenerateRequest):
    """Generate text response without authentication"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Rate limiting
    client_ip = "127.0.0.1"  # In production, get from request
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    start_time = time.time()
    
    try:
        # Format prompt
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(formatted_prompt, "").strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
        
        # Record metrics
        monitor.record_request(generation_time, True)
        
        return GenerateResponse(
            response=response,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            model_name="mistral-7b-finetuned"
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        monitor.record_request(generation_time, False)
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Generation failed")

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint for authentication"""
    user = USERS_DB.get(request.username)
    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=config["api"]["auth"]["access_token_expire_minutes"])
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}, 
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=config["api"]["auth"]["access_token_expire_minutes"] * 60
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get performance metrics"""
    metrics = monitor.get_metrics()
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        avg_latency=metrics["avg_latency"],
        error_rate=metrics["error_rate"],
        model_name="mistral-7b-finetuned"
    )

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        logger.error("Failed to load model on startup")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        workers=config["api"]["workers"]
    ) 