# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from run_inference_api import generate, RuntimeError

class InferenceRequest(BaseModel):
    prompt: str
    # if you like you can expose more parameters here:
    # model: str = None
    # threads: int = None
    # etc.

app = FastAPI()

@app.post("/infer")
async def infer(req: InferenceRequest):
    try:
        result = generate(
            prompt=req.prompt,
            # model=req.model or leave default,
            # threads=req.threads, etc.
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"result": result.strip()}

# To run:
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

