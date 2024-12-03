import os
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db import get_db
from app import models, schemas

router = APIRouter()
upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg"]:
        return JSONResponse(content={"error": "Plik nie jest w formacie JPG."}, status_code=400)

    try:
        with open(os.path.join(upload_dir, file.filename), "wb") as f:
            content = await file.read()
            f.write(content)
        return {"filename": file.filename, "message": "Plik został pomyślnie zapisany."}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)