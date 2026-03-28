from fastapi import FastAPI, UploadFile, File, Query
from faster_whisper import WhisperModel
import os
import uuid

app = FastAPI()

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8",
    cpu_threads=2   # Railway free tier safer
)

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    translate: bool = Query(False),
    language: str = Query(None)
):
    temp_filename = f"temp_{uuid.uuid4()}.wav"

    with open(temp_filename, "wb") as f:
        f.write(await file.read())

    try:
        segments, info = model.transcribe(
            temp_filename,
            beam_size=3,   # lower for Railway
            language=language,
            task="translate" if translate else "transcribe"
        )

        text = " ".join([seg.text for seg in segments])

        return {
            "language": info.language,
            "text": text
        }

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
