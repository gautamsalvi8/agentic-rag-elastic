from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

from fastapi import FastAPI, UploadFile, File
import os
import shutil

from extract_text import read_pdf, read_txt

app = FastAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


@app.get("/")
def health_check():
    return {"status": "backend running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    save_path = os.path.join(DATA_DIR, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if file.filename.lower().endswith(".pdf"):
        extracted_text = read_pdf(save_path)
    elif file.filename.lower().endswith(".txt"):
        extracted_text = read_txt(save_path)
    else:
        return {"error": "Unsupported file type"}

    return {
        "filename": file.filename,
        "characters_extracted": len(extracted_text)
    }
