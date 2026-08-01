import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.auth import authenticate_user, verify_token
from backend.utils import trigger_ecs_pipeline, generate_dashboard

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

app = FastAPI(
    title="AWS Sentiment Analysis Pipeline API",
    description="Backend API for AWS ECS sentiment pipeline orchestration and reporting.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "service": "AWS Sentiment Analysis API",
        "status": "online",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login_user(req: LoginRequest):
    try:
        tokens = authenticate_user(req.username, req.password)
        return {"message": "Login successful", "tokens": tokens}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PreprocessRequest(BaseModel):
    filename: str
    user: str

@app.post("/trigger_pipeline")
def trigger_pipeline(req: PreprocessRequest, current_user: str = Depends(verify_token)):
    try:
        print(f"Received pipeline trigger for user={req.user}, file={req.filename}")
        task_arn = trigger_ecs_pipeline(req.filename, req.user)
        print(f"ECS Task Triggered: {task_arn}")
        return {"status": "pipeline_triggered", "task_arn": task_arn}
    except Exception as e:
        print(f"Error in /trigger_pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_dashboard")
def generate_dashboard_api(req: PreprocessRequest, current_user: str = Depends(verify_token)):
    try:
        print(f"Attempting dashboard generation for {req.user}/{req.filename}")
        generate_dashboard(req.user, req.filename)
        return {"status": "dashboard_generated"}

    except FileNotFoundError as e:
        error_msg = f"Required files missing: {str(e)}"
        print(f"Dashboard generation failed (missing file): {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"Error in /generate_dashboard: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)
